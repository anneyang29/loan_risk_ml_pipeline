"""
Model Training - Legacy Entry Point
=====================================
⚠️ 此檔案為 legacy / backward compatibility 用途。
⚠️ 正式主訓練流程請使用 main.py (Four Phase Training Architecture)。

如果你正在尋找正式的訓練入口：
    python main.py                     # 完整流程
    python main.py --train-only        # 只跑訓練

此檔案保留的原因：
1. 向後相容：已有的 CI/CD 或 script 可能仍引用 train.py
2. Model Registry 操作：--list-models / --compare / --set-best 等指令仍可用

術語提醒：
- 主訓練架構已改用 FourPhaseTrainer（four_phase_trainer.py）
- Champion Strategy = 模型 + imbalance strategy + 設定組合
- Final Champion Artifact = 用 Champion Strategy 在完整 development data 重訓的模型檔案

Version: LEGACY (請改用 main.py)
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# 正式主流程 import
from utils.four_phase_trainer import FourPhaseTrainer, run_four_phase_pipeline
from utils.model_registry import ModelRegistry
from utils.config import ConfigManager, CONFIG_VERSION

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="Credit Scoring Model Training - XGBoost + Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py                             # 預設設定訓練
    python train.py --training-date 2026-04-08  # 指定訓練日期
    python train.py --calibration sigmoid       # Sigmoid calibration
    python train.py --n-splits 10               # 10-fold CV
    python train.py --compare                   # 比較所有模型
    python train.py --list-models               # 列出所有模型
    python train.py --set-best credit_model_2026_04_08  # 手動設定最佳模型

XGBoost 參數調整:
    python train.py --max-depth 6 --learning-rate 0.03
    python train.py --n-estimators 300 --min-child-weight 10
        """
    )
    
    # 基本設定
    parser.add_argument(
        '--training-date',
        type=str,
        default=None,
        help='訓練日期 (YYYY-MM-DD)，用於模型版本命名'
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='實驗 Run ID（預設自動生成時間戳）'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='設定檔路徑 (YAML/JSON)'
    )
    
    # Model Registry 操作
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='列出所有訓練過的模型'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='比較所有模型表現'
    )
    
    parser.add_argument(
        '--set-best',
        type=str,
        default=None,
        help='手動設定最佳模型版本'
    )
    
    parser.add_argument(
        '--no-auto-best',
        action='store_true',
        help='不自動選擇最佳模型'
    )
    
    # Calibration 設定
    parser.add_argument(
        '--calibration',
        type=str,
        default='isotonic',
        choices=['isotonic', 'sigmoid'],
        help='Probability calibration 方法（預設 isotonic）'
    )
    
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Cross-validation 折數（預設 5）'
    )
    
    # XGBoost 參數
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=200,
        help='XGBoost trees 數量（預設 200）'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=5,
        help='Tree 最大深度（預設 5）'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.05,
        help='Learning rate（預設 0.05）'
    )
    
    parser.add_argument(
        '--min-child-weight',
        type=int,
        default=5,
        help='最小葉節點權重（預設 5）'
    )
    
    parser.add_argument(
        '--subsample',
        type=float,
        default=0.8,
        help='Subsample ratio（預設 0.8）'
    )
    
    parser.add_argument(
        '--colsample-bytree',
        type=float,
        default=0.8,
        help='Column subsample ratio（預設 0.8）'
    )
    
    parser.add_argument(
        '--reg-alpha',
        type=float,
        default=0.1,
        help='L1 正則化（預設 0.1）'
    )
    
    parser.add_argument(
        '--reg-lambda',
        type=float,
        default=1.0,
        help='L2 正則化（預設 1.0）'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 專案根目錄
    project_root = Path(__file__).parent
    model_bank_path = project_root / "model_bank"
    
    # ============================================
    # Model Registry 操作（不需要訓練）
    # ============================================
    if args.list_models or args.compare or args.set_best:
        registry = ModelRegistry(model_bank_path)
        
        if args.list_models:
            print("\n📦 Models in Registry:")
            print("-" * 50)
            for v in registry.list_models():
                r = registry.records[v]
                status = []
                if r.is_best:
                    status.append("⭐ BEST")
                if r.is_production:
                    status.append("🚀 PROD")
                status_str = f" {' '.join(status)}" if status else ""
                print(f"  {v}{status_str}")
                print(f"    AUC: {r.auc_test:.4f} | GINI: {r.gini_test:.4f}")
            print()
            return
        
        if args.compare:
            print(registry.compare_models())
            return
        
        if args.set_best:
            registry.set_best_model(args.set_best, reason="Manually selected")
            print(f"✓ Best model set to: {args.set_best}")
            return
    
    # ============================================
    # 模型訓練 → 轉接到 Four Phase Pipeline
    # ============================================
    logger.warning("=" * 70)
    logger.warning("⚠️  train.py 是 legacy entry point")
    logger.warning("⚠️  正式主流程請使用: python main.py")
    logger.warning("⚠️  現在將轉接到 Four Phase Training Pipeline...")
    logger.warning("=" * 70)

    results = run_four_phase_pipeline(
        project_root=project_root,
        imbalance_strategy="scale_weight",
        lower_threshold=0.4,
        upper_threshold=0.7,
    )

    logger.info("\n" + "=" * 70)
    logger.info("🎉 訓練完成！（透過 Four Phase Pipeline）")
    logger.info("=" * 70)
    if results:
        logger.info(f"Champion Strategy: {results.get('champion_strategy', 'N/A')}")
        logger.info(f"Output: {results.get('output_dir', 'N/A')}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
