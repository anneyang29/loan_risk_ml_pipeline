#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rolling Training CLI
====================

執行 Rolling Window 訓練流程的命令列工具。

Usage:
    # 完整流程
    python rolling_train.py --run-all
    
    # 只執行 rolling training
    python rolling_train.py --rolling-only
    
    # 只執行 OOT inference
    python rolling_train.py --oot-only --model-path model_bank/xgb_calibrated_v1.pkl
    
    # 自訂閾值
    python rolling_train.py --run-all --lower-threshold 0.35 --upper-threshold 0.75
    
    # Production monitoring
    python rolling_train.py --monitor --model-version v_20250701_rolling
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_spark():
    """初始化 PySpark"""
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder \
        .appName("RollingTraining") \
        .config("spark.sql.session.timeZone", "UTC") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


def run_rolling_training(
    dev_data_path: str,
    window_definition_path: str,
    model_bank_path: str,
    random_state: int,
    lower_threshold: float,
    upper_threshold: float,
    strategies: list = None
):
    """
    執行 Rolling Training
    """
    from utils.rolling_trainer import (
        RollingTrainer,
        run_rolling_training as execute_rolling,
        aggregate_rolling_results,
        select_champion_strategy
    )
    
    logger.info("=" * 60)
    logger.info("Rolling Training 開始")
    logger.info("=" * 60)
    
    spark = setup_spark()
    
    try:
        # 建立 trainer
        trainer = RollingTrainer(
            spark=spark,
            dev_data_path=dev_data_path,
            window_definition_path=window_definition_path,
            model_bank_path=model_bank_path,
            random_state=random_state,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold
        )
        
        # 執行 rolling training
        if strategies:
            all_results = execute_rolling(trainer, strategies=strategies)
        else:
            all_results = execute_rolling(trainer)
        
        # 彙總結果
        rolling_summary = aggregate_rolling_results(all_results)
        
        # 選擇 champion
        champion = select_champion_strategy(rolling_summary)
        
        logger.info(f"\n✓ Champion Strategy: {champion}")
        
        return trainer, all_results, rolling_summary, champion
        
    finally:
        spark.stop()


def run_oot_inference(
    trainer,
    oot_data_path: str,
    model_path: str,
    champion_strategy: str,
    output_dir: str
):
    """
    執行 OOT Inference
    """
    from utils.rolling_trainer import (
        retrain_final_champion,
        infer_on_oot
    )
    
    logger.info("=" * 60)
    logger.info("OOT Inference 開始")
    logger.info("=" * 60)
    
    spark = setup_spark()
    
    try:
        # 如果沒有提供已訓練的 model path，重新訓練 final champion
        if model_path is None:
            # 從 window definition 取得最新訓練窗口
            final_model, final_train_range = retrain_final_champion(trainer, champion_strategy)
            
            # 儲存 final model
            import joblib
            model_path = Path(output_dir) / f"final_champion_{champion_strategy}.pkl"
            joblib.dump(final_model, model_path)
            logger.info(f"✓ Final model saved: {model_path}")
        else:
            import joblib
            final_model = joblib.load(model_path)
            final_train_range = {"start": "N/A", "end": "N/A"}
        
        # OOT inference
        oot_results = infer_on_oot(
            spark=spark,
            oot_data_path=oot_data_path,
            model=final_model,
            trainer=trainer,
            output_path=Path(output_dir) / "oot_predictions.csv"
        )
        
        return oot_results
        
    finally:
        spark.stop()


def register_model(
    model_bank_path: str,
    champion_strategy: str,
    rolling_summary: dict,
    oot_metrics: dict = None,
    zone_summary: list = None,
    threshold_config: dict = None
):
    """
    註冊模型到 registry
    """
    from utils.model_registry import ExtendedModelRegistry
    import uuid
    
    registry = ExtendedModelRegistry(Path(model_bank_path))
    
    # 生成版本號
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"v_{timestamp}_rolling_{champion_strategy}"
    run_id = str(uuid.uuid4())[:8]
    
    record = registry.register_rolling_model(
        model_version=model_version,
        run_id=run_id,
        training_date=datetime.now().strftime("%Y-%m-%d"),
        champion_strategy=champion_strategy,
        rolling_summary=rolling_summary,
        oot_metrics=oot_metrics,
        zone_summary=zone_summary,
        threshold_config=threshold_config,
        model_path=str(Path(model_bank_path) / f"final_champion_{champion_strategy}.pkl"),
        rolling_results_path=str(Path(model_bank_path) / "rolling_results.csv")
    )
    
    # 設為 best model
    registry.set_best_model(model_version, "Rolling training champion")
    
    logger.info(f"✓ Model registered: {model_version}")
    return model_version


def run_production_monitoring(
    model_bank_path: str,
    model_version: str,
    current_data_path: str,
    reference_data_path: str
):
    """
    執行 Production Monitoring
    """
    from utils.monitoring import ProductionMonitor, ProductionMonitorConfig, check_retrain_trigger
    from utils.model_registry import ExtendedModelRegistry
    
    logger.info("=" * 60)
    logger.info("Production Monitoring 開始")
    logger.info("=" * 60)
    
    spark = setup_spark()
    
    try:
        # 載入 model record
        registry = ExtendedModelRegistry(Path(model_bank_path))
        record = registry.get_model_record(model_version)
        
        if record is None:
            logger.error(f"模型不存在: {model_version}")
            return None
        
        # 設定 monitoring config
        config = ProductionMonitorConfig(
            auc_min_threshold=0.75,
            f1_reject_min_threshold=0.20,
            psi_max_threshold=0.25,
            max_days_since_training=180
        )
        
        # 建立 monitor
        monitor = ProductionMonitor(spark, config)
        
        # 執行 monitoring
        result = monitor.run_monitoring(
            current_data_path=current_data_path,
            reference_data_path=reference_data_path,
            model_record=record
        )
        
        # 檢查是否需要 retrain
        trigger_result = check_retrain_trigger(result)
        
        logger.info(f"\n監控結果:")
        logger.info(f"  需要 Retrain: {trigger_result['need_retrain']}")
        if trigger_result['triggered_reasons']:
            logger.info(f"  觸發原因: {', '.join(trigger_result['triggered_reasons'])}")
        
        return result, trigger_result
        
    finally:
        spark.stop()


def save_outputs(
    output_dir: str,
    rolling_summary: dict = None,
    champion: str = None,
    oot_metrics: dict = None,
    zone_summary: list = None,
    threshold_config: dict = None
):
    """
    儲存輸出檔案
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # champion_summary.json
    if rolling_summary and champion:
        champion_summary = {
            "champion_strategy": champion,
            "selection_timestamp": datetime.now().isoformat(),
            "strategy_metrics": rolling_summary.get(champion, {}),
            "all_strategies_comparison": {
                strategy: {
                    "avg_cv_auc": data.get("avg_cv_auc", 0),
                    "avg_monitor_auc": data.get("avg_monitor_auc", 0),
                    "avg_monitor_f1_reject": data.get("avg_monitor_f1_reject", 0),
                    "overall_score": data.get("overall_score", 0)
                }
                for strategy, data in rolling_summary.items()
            },
            "threshold_config": threshold_config
        }
        
        with open(output_path / "champion_summary.json", "w", encoding="utf-8") as f:
            json.dump(champion_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {output_path / 'champion_summary.json'}")
    
    # oot_metrics.json
    if oot_metrics:
        with open(output_path / "oot_metrics.json", "w", encoding="utf-8") as f:
            json.dump(oot_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {output_path / 'oot_metrics.json'}")
    
    # zone_summary.json
    if zone_summary:
        with open(output_path / "zone_summary.json", "w", encoding="utf-8") as f:
            json.dump(zone_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {output_path / 'zone_summary.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Rolling Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 完整流程
    python rolling_train.py --run-all
    
    # 只執行 rolling training（不含 OOT inference）
    python rolling_train.py --rolling-only
    
    # 只執行 OOT inference（需提供已訓練模型）
    python rolling_train.py --oot-only --model-path model_bank/final_champion_xgb_calibrated.pkl
    
    # 自訂閾值
    python rolling_train.py --run-all --lower-threshold 0.35 --upper-threshold 0.75
    
    # 只訓練特定策略
    python rolling_train.py --run-all --strategies xgb_calibrated rf_calibrated
    
    # Production monitoring
    python rolling_train.py --monitor --model-version v_20250701_rolling_xgb_calibrated
        """
    )
    
    # 執行模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--run-all", action="store_true",
                           help="執行完整流程（rolling training + OOT inference + register）")
    mode_group.add_argument("--rolling-only", action="store_true",
                           help="只執行 rolling training")
    mode_group.add_argument("--oot-only", action="store_true",
                           help="只執行 OOT inference")
    mode_group.add_argument("--monitor", action="store_true",
                           help="執行 production monitoring")
    
    # 路徑設定
    parser.add_argument("--dev-data-path", type=str,
                       default="datamart/gold/development",
                       help="Development data 路徑")
    parser.add_argument("--oot-data-path", type=str,
                       default="datamart/gold/oot",
                       help="OOT data 路徑")
    parser.add_argument("--window-definition-path", type=str,
                       default="datamart/gold/rolling_window_definition.csv",
                       help="Rolling window definition CSV 路徑")
    parser.add_argument("--model-bank-path", type=str,
                       default="model_bank",
                       help="Model bank 路徑")
    parser.add_argument("--output-dir", type=str,
                       default="model_bank",
                       help="輸出目錄")
    
    # 模型設定
    parser.add_argument("--model-path", type=str, default=None,
                       help="已訓練模型路徑（用於 --oot-only）")
    parser.add_argument("--model-version", type=str, default=None,
                       help="模型版本（用於 --monitor）")
    
    # 訓練參數
    parser.add_argument("--random-state", type=int, default=2022,
                       help="Random state (default: 2022)")
    parser.add_argument("--lower-threshold", type=float, default=0.4,
                       help="Lower threshold for zone assignment (default: 0.4)")
    parser.add_argument("--upper-threshold", type=float, default=0.7,
                       help="Upper threshold for zone assignment (default: 0.7)")
    parser.add_argument("--strategies", nargs="+", default=None,
                       help="要訓練的策略（預設全部）")
    
    # Monitoring 參數
    parser.add_argument("--current-data-path", type=str, default=None,
                       help="當前資料路徑（用於 monitoring）")
    parser.add_argument("--reference-data-path", type=str, default=None,
                       help="參考資料路徑（用於 monitoring）")
    
    args = parser.parse_args()
    
    threshold_config = {
        "lower_threshold": args.lower_threshold,
        "upper_threshold": args.upper_threshold
    }
    
    logger.info("=" * 70)
    logger.info("Rolling Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Random State: {args.random_state}")
    logger.info(f"Threshold Config: {threshold_config}")
    logger.info(f"Model Bank: {args.model_bank_path}")
    
    try:
        if args.run_all:
            # ===== 完整流程 =====
            
            # 1. Rolling Training
            trainer, all_results, rolling_summary, champion = run_rolling_training(
                dev_data_path=args.dev_data_path,
                window_definition_path=args.window_definition_path,
                model_bank_path=args.model_bank_path,
                random_state=args.random_state,
                lower_threshold=args.lower_threshold,
                upper_threshold=args.upper_threshold,
                strategies=args.strategies
            )
            
            # 2. OOT Inference
            oot_results = run_oot_inference(
                trainer=trainer,
                oot_data_path=args.oot_data_path,
                model_path=args.model_path,
                champion_strategy=champion,
                output_dir=args.output_dir
            )
            
            oot_metrics = oot_results.get("metrics", {})
            zone_summary = oot_results.get("zone_summary", [])
            
            # 3. 儲存輸出
            save_outputs(
                output_dir=args.output_dir,
                rolling_summary=rolling_summary,
                champion=champion,
                oot_metrics=oot_metrics,
                zone_summary=zone_summary,
                threshold_config=threshold_config
            )
            
            # 4. 註冊模型
            model_version = register_model(
                model_bank_path=args.model_bank_path,
                champion_strategy=champion,
                rolling_summary=rolling_summary,
                oot_metrics=oot_metrics,
                zone_summary=zone_summary,
                threshold_config=threshold_config
            )
            
            logger.info("\n" + "=" * 70)
            logger.info("✓ Rolling Training Pipeline 完成!")
            logger.info("=" * 70)
            logger.info(f"  Champion Strategy: {champion}")
            logger.info(f"  Model Version: {model_version}")
            if oot_metrics:
                logger.info(f"  OOT AUC: {oot_metrics.get('auc', 'N/A'):.4f}")
                logger.info(f"  OOT F1 (reject): {oot_metrics.get('f1_reject', 'N/A'):.4f}")
            
        elif args.rolling_only:
            # ===== 只執行 Rolling Training =====
            trainer, all_results, rolling_summary, champion = run_rolling_training(
                dev_data_path=args.dev_data_path,
                window_definition_path=args.window_definition_path,
                model_bank_path=args.model_bank_path,
                random_state=args.random_state,
                lower_threshold=args.lower_threshold,
                upper_threshold=args.upper_threshold,
                strategies=args.strategies
            )
            
            # 儲存部分輸出
            save_outputs(
                output_dir=args.output_dir,
                rolling_summary=rolling_summary,
                champion=champion,
                threshold_config=threshold_config
            )
            
            logger.info("\n" + "=" * 70)
            logger.info("✓ Rolling Training 完成!")
            logger.info("=" * 70)
            logger.info(f"  Champion Strategy: {champion}")
            
        elif args.oot_only:
            # ===== 只執行 OOT Inference =====
            if args.model_path is None:
                logger.error("必須提供 --model-path")
                sys.exit(1)
            
            # 需要載入 champion strategy（從 champion_summary.json）
            champion_file = Path(args.output_dir) / "champion_summary.json"
            if champion_file.exists():
                with open(champion_file, "r") as f:
                    champion_data = json.load(f)
                    champion = champion_data.get("champion_strategy", "xgb_calibrated")
            else:
                champion = "xgb_calibrated"  # 預設
                logger.warning(f"找不到 champion_summary.json，使用預設策略: {champion}")
            
            # 建立 minimal trainer for OOT
            from utils.rolling_trainer import RollingTrainer
            spark = setup_spark()
            trainer = RollingTrainer(
                spark=spark,
                dev_data_path=args.dev_data_path,
                window_definition_path=args.window_definition_path,
                model_bank_path=args.model_bank_path,
                random_state=args.random_state,
                lower_threshold=args.lower_threshold,
                upper_threshold=args.upper_threshold
            )
            spark.stop()
            
            oot_results = run_oot_inference(
                trainer=trainer,
                oot_data_path=args.oot_data_path,
                model_path=args.model_path,
                champion_strategy=champion,
                output_dir=args.output_dir
            )
            
            oot_metrics = oot_results.get("metrics", {})
            zone_summary = oot_results.get("zone_summary", [])
            
            save_outputs(
                output_dir=args.output_dir,
                oot_metrics=oot_metrics,
                zone_summary=zone_summary
            )
            
            logger.info("\n" + "=" * 70)
            logger.info("✓ OOT Inference 完成!")
            logger.info("=" * 70)
            if oot_metrics:
                logger.info(f"  OOT AUC: {oot_metrics.get('auc', 'N/A'):.4f}")
            
        elif args.monitor:
            # ===== Production Monitoring =====
            if args.model_version is None:
                logger.error("必須提供 --model-version")
                sys.exit(1)
            
            current_data = args.current_data_path or args.oot_data_path
            reference_data = args.reference_data_path or args.dev_data_path
            
            result, trigger_result = run_production_monitoring(
                model_bank_path=args.model_bank_path,
                model_version=args.model_version,
                current_data_path=current_data,
                reference_data_path=reference_data
            )
            
            logger.info("\n" + "=" * 70)
            logger.info("✓ Production Monitoring 完成!")
            logger.info("=" * 70)
            logger.info(f"  需要 Retrain: {trigger_result['need_retrain']}")
            if trigger_result['triggered_reasons']:
                for reason in trigger_result['triggered_reasons']:
                    logger.info(f"    - {reason}")
    
    except Exception as e:
        logger.error(f"執行失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
