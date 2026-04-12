"""
ML Pipeline 主程式入口點
============================
使用 Four Phase Trainer 架構執行完整的信用風險模型訓練流程

四階段流程：
1. Phase 1: Model Development (18 個月 Rolling Training)
2. Phase 2: Champion Retraining (用完整 development data 重訓 → 產出 Final Champion Artifact)
3. Phase 3: Policy Validation (4 個月，threshold / zone policy tuning)
4. Phase 4: Final Blind Holdout (最後 2 個月，完全不調模型與 threshold)

重要術語：
- Champion Strategy: 模型 + imbalance strategy + 設定組合（Phase 1 選出）
- Final Champion Artifact: 用 Champion Strategy 在完整 development data 重訓的實際模型檔案（Phase 2 產出）
- Policy Validation: threshold / zone policy tuning window（Phase 3，4 個月）
- Final Blind Holdout: 完全不調模型與 threshold 的最終驗證集（Phase 4，2 個月）
- Production Batch Scoring: 上線後的批次推論，沒有即時 label，只輸出 predictions

Usage:
    python main.py                             # 執行 Data + Training（預設）
    python main.py --all                       # 執行完整流程 (Data + Training + Monitoring)
    python main.py --data-only                 # 只執行 Bronze -> Silver -> Gold
    python main.py --train-only                # 只執行四階段訓練
    python main.py --monitor-only              # 只執行 Production Monitoring
    python main.py --lower-threshold 0.3       # 自訂閾值
    python main.py --imbalance scale_weight    # 指定不平衡處理方法
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# 添加專案根目錄到 Python 路徑
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_processing_bronze_table import run_bronze_pipeline
from utils.data_processing_silver_table import run_silver_pipeline
from utils.data_processing_gold_table import run_gold_pipeline
from utils.four_phase_trainer import (
    FourPhaseTrainer,
    run_four_phase_pipeline,
    PhaseConfig,
)
from utils.production_monitor import (
    ProductionMonitor,
    ProductionMonitoringConfig,
    generate_retraining_data_window,
)

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================
# Data Pipeline
# ============================================
def run_data_pipeline(project_root: Path = None):
    """
    執行完整的資料處理流程 Bronze -> Silver -> Gold

    每一層都是 function-based pipeline，對齊 utils 裡的實際實作。
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    print("\n" + "=" * 70)
    print("  資料處理流程 (Bronze → Silver → Gold)")
    print("=" * 70)

    # ----- Bronze Layer -----
    print("\n[1/3] Bronze Layer 處理中...")
    bronze_path = run_bronze_pipeline(project_root)
    print(f"  ✓ Bronze 完成，輸出: {bronze_path}")

    # ----- Silver Layer -----
    print("\n[2/3] Silver Layer 處理中...")
    silver_path = run_silver_pipeline(project_root)
    print(f"  ✓ Silver 完成，輸出: {silver_path}")

    # ----- Gold Layer -----
    print("\n[3/3] Gold Layer 處理中...")
    gold_output_paths = run_gold_pipeline(project_root)
    print(f"  ✓ Gold 完成")
    for name, path in gold_output_paths.items():
        if name not in ("artifact_files",):
            print(f"    - {name}: {path}")

    # Manifest 確認
    if "manifest" in gold_output_paths:
        print(f"\n  📄 Gold Split Manifest 已輸出:")
        print(f"     {gold_output_paths['manifest']}")
        print(f"     說明: 記錄 development / oot 路徑與四階段對應關係")

    print("\n" + "=" * 70)
    print("  資料處理流程完成！")
    print("=" * 70)

    return gold_output_paths


# ============================================
# Training Pipeline
# ============================================
def run_training_pipeline(
    project_root: Path = None,
    lower_threshold: float = None,
    upper_threshold: float = None,
    imbalance_strategy: str = "scale_weight",
    use_calibration: bool = True,
    model_names: list = None,
):
    """
    執行四階段訓練流程

    唯一主訓練入口，呼叫 four_phase_trainer.run_four_phase_pipeline()。
    
    lower/upper_threshold 作為 fallback（Phase 3 推薦優先）。
    若為 None，使用 run_four_phase_pipeline 的預設值 (0.5 / 0.85)。
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    
    # 使用預設值 fallback
    lower_threshold = lower_threshold or 0.5
    upper_threshold = upper_threshold or 0.85

    print("\n" + "=" * 70)
    print("  四階段訓練流程 (Four Phase Training)")
    print("=" * 70)
    print(f"  Imbalance Strategy : {imbalance_strategy}")
    print(f"  Lower Threshold    : {lower_threshold} (fallback, Phase 3 推薦優先)")
    print(f"  Upper Threshold    : {upper_threshold} (fallback, Phase 3 推薦優先)")
    print(f"  Calibration        : {use_calibration}")
    print("=" * 70)

    results = run_four_phase_pipeline(
        project_root=project_root,
        model_names=model_names,
        use_calibration=use_calibration,
        imbalance_strategy=imbalance_strategy,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )

    print("\n" + "=" * 70)
    print("  四階段訓練流程完成！")
    print("=" * 70)

    return results


# ============================================
# Monitoring Pipeline
# ============================================
def run_monitoring_pipeline(
    project_root: Path = None,
    lower_threshold: float = None,
    upper_threshold: float = None,
):
    """
    執行 Production Monitoring

    Threshold 優先順序：
    1. zone_policy_summary.json 中的 selected_lower/upper_threshold (Phase 3 推薦)
    2. champion_summary.json 中的 threshold_config
    3. CLI 傳入的 lower_threshold / upper_threshold
    4. ProductionMonitoringConfig 預設值

    Baseline 來源：
    - 從最新 model_bank 結果中讀取 policy_validation_predictions.csv
    - Policy Validation 是模型部署前最後一次有 label 的評估階段
    - 其 score 分布最接近上線後 production 推論的真實分布
    
    Monitoring 對象（demonstration 模式）：
    - 使用 final_holdout_predictions.csv 模擬 production data
    - 若有真實 production data，應替換為 production 資料
    - Final Blind Holdout 有真實 label，可算 AUC/F1；
      真正的 Production Batch Scoring 沒有 label，只能比 PSI / zone shift
    """
    import json
    import numpy as np
    import pandas as pd

    if project_root is None:
        project_root = PROJECT_ROOT

    print("\n" + "=" * 70)
    print("  Production Monitoring")
    print("=" * 70)

    # 找最新的 model_bank 結果
    model_bank = project_root / "model_bank"
    if not model_bank.exists():
        print("  ⚠️ 找不到 model_bank，請先執行 training pipeline")
        return None

    # 找最新 run（支援 four_phase_ 與 legacy rolling_training_v2_ 兩種前綴）
    run_dirs = sorted(
        [d for d in model_bank.iterdir()
         if d.is_dir() and (d.name.startswith("four_phase_") or d.name.startswith("rolling_training_v2_"))],
        reverse=True
    )
    if not run_dirs:
        print("  ⚠️ 找不到訓練結果目錄（搜尋 four_phase_* 或 rolling_training_v2_*）")
        return None

    latest_dir = run_dirs[0]
    print(f"  使用最新訓練結果: {latest_dir.name}")
    if latest_dir.name.startswith("rolling_training_v2_"):
        print(f"  ⚠️ 注意：此為 legacy 命名格式，新版訓練會使用 four_phase_* 前綴")

    # ★ 自動偵測 Phase 3 推薦的 threshold
    # 優先順序: zone_policy_summary.json → champion_summary.json → CLI 參數 → config 預設值
    resolved_lower = lower_threshold
    resolved_upper = upper_threshold
    threshold_source = "default"
    
    zone_policy_path = latest_dir / "zone_policy_summary.json"
    champion_path = latest_dir / "champion_summary.json"
    
    if zone_policy_path.exists():
        with open(zone_policy_path, "r", encoding="utf-8") as f:
            zone_policy = json.load(f)
        if "selected_lower_threshold" in zone_policy:
            resolved_lower = zone_policy["selected_lower_threshold"]
            resolved_upper = zone_policy["selected_upper_threshold"]
            threshold_source = "zone_policy_summary.json (Phase 3 推薦)"
        elif "recommended_lower_threshold" in zone_policy:
            resolved_lower = zone_policy["recommended_lower_threshold"]
            resolved_upper = zone_policy["recommended_upper_threshold"]
            threshold_source = "zone_policy_summary.json (recommended)"
    elif champion_path.exists():
        with open(champion_path, "r", encoding="utf-8") as f:
            champion_info = json.load(f)
        tc = champion_info.get("threshold_config", {})
        if tc.get("lower_threshold") is not None:
            resolved_lower = tc["lower_threshold"]
            resolved_upper = tc["upper_threshold"]
            threshold_source = "champion_summary.json"
    
    # CLI 傳入的值優先於檔案偵測（若使用者明確指定）
    if lower_threshold is not None:
        resolved_lower = lower_threshold
        threshold_source = "CLI 參數"
    if upper_threshold is not None:
        resolved_upper = upper_threshold
        threshold_source = "CLI 參數"
    
    # 若都沒有，使用 config 預設值
    if resolved_lower is None:
        resolved_lower = ProductionMonitoringConfig().default_lower_threshold
    if resolved_upper is None:
        resolved_upper = ProductionMonitoringConfig().default_upper_threshold
    
    print(f"\n  ★ Monitoring Threshold:")
    print(f"     Lower: {resolved_lower}")
    print(f"     Upper: {resolved_upper}")
    print(f"     來源: {threshold_source}")

    # 讀取 policy validation predictions（作為 baseline）
    pv_pred_path = latest_dir / "policy_validation_predictions.csv"
    holdout_pred_path = latest_dir / "final_holdout_predictions.csv"

    if not pv_pred_path.exists():
        print("  ⚠️ 找不到 policy_validation_predictions.csv")
        return None

    print(f"\n  📊 Baseline 來源:")
    print(f"     檔案: {pv_pred_path}")
    print(f"     說明: Phase 3 Policy Validation 的預測機率")
    print(f"     原因: 部署前最後一次有 label 的評估，score 分布最接近 production")

    # Baseline = policy validation 的 score distribution
    pv_df = pd.read_csv(pv_pred_path)
    baseline_scores = pv_df["pred_prob"].values
    print(f"     樣本數: {len(baseline_scores)}")
    print(f"     Score 平均: {baseline_scores.mean():.4f}, 標準差: {baseline_scores.std():.4f}")

    # 建立 monitor（使用 resolved threshold）
    config = ProductionMonitoringConfig(
        default_lower_threshold=resolved_lower,
        default_upper_threshold=resolved_upper,
    )
    monitor = ProductionMonitor(config=config)
    monitor.set_baseline(baseline_scores, resolved_lower, resolved_upper)

    # 讀取 champion_summary 取得 training date
    champion_path = latest_dir / "champion_summary.json"
    if champion_path.exists():
        with open(champion_path, "r", encoding="utf-8") as f:
            champion_info = json.load(f)
        monitor.set_last_training_date(champion_info.get("created_at", "")[:10])

    # 如果有 holdout predictions → 使用它來做 monitoring demo
    if holdout_pred_path.exists():
        print(f"\n  📋 Monitoring 對象 (demonstration):")
        print(f"     檔案: {holdout_pred_path}")
        print(f"     說明: Phase 4 Final Blind Holdout 預測結果")
        print(f"     ⚠️ 真實 production 環境應替換為 production batch scoring 資料")

        holdout_df = pd.read_csv(holdout_pred_path)
        predictions = holdout_df["pred_prob"].values
        labels = holdout_df["actual_label"].values if "actual_label" in holdout_df.columns else None

        result = monitor.run_production_monitoring(
            predictions=predictions,
            labels=labels,
            model_version=latest_dir.name,
            period_start=str(holdout_df["進件日"].min()) if "進件日" in holdout_df.columns else "",
            period_end=str(holdout_df["進件日"].max()) if "進件日" in holdout_df.columns else "",
            lower_threshold=resolved_lower,
            upper_threshold=resolved_upper,
        )

        # 儲存結果（含 threshold metadata）
        output_dir = project_root / "model_bank" / "monitoring"
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = output_dir / f"monitoring_result_{ts}.json"
        
        # 在 monitoring result 中加入 threshold metadata
        result_dict = result.to_dict()
        result_dict["threshold_config"] = {
            "lower_threshold": resolved_lower,
            "upper_threshold": resolved_upper,
            "source": threshold_source,
        }
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Monitoring 結果儲存至: {result_path}")

        # Retraining Decision
        decision = {
            "timestamp": ts,
            "needs_retraining": result.needs_retraining,
            "trigger_reason": result.retraining_trigger_reason,
            "alerts": result.alerts,
            "warnings": result.warnings,
        }
        if result.needs_retraining:
            # 產生 dynamic retraining window
            window = generate_retraining_data_window(
                current_date=datetime.now().date()
            )
            decision["retraining_window"] = window

        decision_path = output_dir / f"retraining_decision_{ts}.json"
        with open(decision_path, "w", encoding="utf-8") as f:
            json.dump(decision, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Retraining decision 儲存至: {decision_path}")

        return result
    else:
        print("  ⚠️ 找不到 final_holdout_predictions.csv，無法執行 monitoring")
        return None


# ============================================
# Full Pipeline
# ============================================
def run_full_pipeline(
    project_root: Path = None,
    lower_threshold: float = None,
    upper_threshold: float = None,
    imbalance_strategy: str = "scale_weight",
    use_calibration: bool = True,
    model_names: list = None,
    include_monitoring: bool = False,
):
    """
    執行完整流程：資料處理 + 四階段訓練 + (可選) Production Monitoring

    Args:
        lower_threshold: fallback 閾值，Phase 3 推薦優先。None → 使用預設 0.5
        upper_threshold: fallback 閾值，Phase 3 推薦優先。None → 使用預設 0.85
        include_monitoring: 若為 True，訓練完成後自動執行 Monitoring
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    print("\n" + "=" * 80)
    print("  ML PIPELINE - 完整流程")
    print("  Four Phase Training Architecture")
    if include_monitoring:
        print("  (含 Production Monitoring)")
    print("=" * 80)

    # Step 1: 資料處理
    run_data_pipeline(project_root)

    # Step 2: 四階段訓練
    results = run_training_pipeline(
        project_root=project_root,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        imbalance_strategy=imbalance_strategy,
        use_calibration=use_calibration,
        model_names=model_names,
    )

    # Step 3: Production Monitoring（可選）
    # ★ 使用 training 結果中的 selected threshold（Phase 3 推薦），而非 CLI default
    monitoring_result = None
    if include_monitoring:
        selected_lower = None
        selected_upper = None
        if results and isinstance(results, dict):
            selected_lower = results.get("selected_lower_threshold")
            selected_upper = results.get("selected_upper_threshold")
        
        monitoring_result = run_monitoring_pipeline(
            project_root=project_root,
            lower_threshold=selected_lower,
            upper_threshold=selected_upper,
        )

    print("\n" + "=" * 80)
    print("  完整流程執行完畢！")
    if include_monitoring:
        print("  ✓ Data Pipeline → Training → Monitoring 全部完成")
    print("=" * 80)

    return {"training": results, "monitoring": monitoring_result}


# ============================================
# CLI Entry Point
# ============================================
def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description="ML Pipeline - Four Phase Training Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法：
  python main.py                               # 執行 Data + Training（預設）
  python main.py --all                         # 一路到底：Data + Training + Monitoring
  python main.py --data-only                   # 只執行資料處理
  python main.py --train-only                  # 只執行訓練
  python main.py --monitor-only                # 只執行 Production Monitoring
  python main.py --lower-threshold 0.35        # 自訂閾值
  python main.py --imbalance scale_weight      # 指定不平衡處理方法
  python main.py --imbalance smote             # 使用 SMOTE（需安裝 imbalanced-learn）
        """
    )

    # 模式選擇（互斥）
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--data-only",
        action="store_true",
        help="只執行資料處理流程 (Bronze -> Silver -> Gold)"
    )
    mode_group.add_argument(
        "--train-only",
        action="store_true",
        help="只執行四階段訓練流程（需已有 Gold 資料）"
    )
    mode_group.add_argument(
        "--monitor-only",
        action="store_true",
        help="只執行 Production Monitoring（需已有訓練結果）"
    )
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="執行完整流程：Data + Training + Monitoring（一路到底）"
    )

    # Threshold（作為 fallback，Phase 3 推薦值優先）
    parser.add_argument(
        "--lower-threshold",
        type=float,
        default=None,
        help="三區間下限閾值 (不指定則自動使用 Phase 3 推薦值，再 fallback 到 0.5)"
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=None,
        help="三區間上限閾值 (不指定則自動使用 Phase 3 推薦值，再 fallback 到 0.85)"
    )

    # Imbalance
    parser.add_argument(
        "--imbalance",
        type=str,
        default="scale_weight",
        choices=["scale_weight", "class_weight", "smote", "undersample", "none"],
        help="不平衡資料處理方法 (default: scale_weight，推薦)"
    )

    # Calibration
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="不使用 probability calibration"
    )

    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="指定候選模型 (e.g., xgboost random_forest logistic_regression)"
    )

    args = parser.parse_args()

    # 根據參數決定執行模式
    if args.data_only:
        run_data_pipeline()

    elif args.train_only:
        run_training_pipeline(
            lower_threshold=args.lower_threshold,
            upper_threshold=args.upper_threshold,
            imbalance_strategy=args.imbalance,
            use_calibration=not args.no_calibration,
            model_names=args.models,
        )

    elif args.monitor_only:
        run_monitoring_pipeline(
            lower_threshold=args.lower_threshold,
            upper_threshold=args.upper_threshold,
        )

    elif args.all:
        # Data + Training + Monitoring 一路到底
        run_full_pipeline(
            lower_threshold=args.lower_threshold,
            upper_threshold=args.upper_threshold,
            imbalance_strategy=args.imbalance,
            use_calibration=not args.no_calibration,
            model_names=args.models,
            include_monitoring=True,
        )

    else:
        # 預設執行 Data + Training（不含 Monitoring）
        run_full_pipeline(
            lower_threshold=args.lower_threshold,
            upper_threshold=args.upper_threshold,
            imbalance_strategy=args.imbalance,
            use_calibration=not args.no_calibration,
            model_names=args.models,
            include_monitoring=False,
        )


if __name__ == "__main__":
    main()
