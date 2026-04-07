"""
Model Training - Entry Point
============================
XGBoost + Calibration 信用評分模型訓練

Usage:
    python train.py                           # 使用預設設定
    python train.py --calibration sigmoid     # 使用 sigmoid calibration
    python train.py --n-splits 10             # 10-fold CV
    python train.py --run-id exp_001          # 指定 run ID
    python train.py --config config/prod.yaml # 使用設定檔
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

from pyspark.sql import SparkSession

from utils.model_train import run_model_training, CreditScoringModelTrainer
from utils.config import ConfigManager, CONFIG_VERSION

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_spark_session() -> SparkSession:
    """建立 Spark Session"""
    return SparkSession.builder \
        .appName("model_training") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="Credit Scoring Model Training - XGBoost + Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py                             # 預設設定訓練
    python train.py --calibration sigmoid       # Sigmoid calibration
    python train.py --n-splits 10               # 10-fold CV
    python train.py --run-id experiment_001     # 指定實驗 ID
    python train.py --config config/prod.yaml   # 使用設定檔

XGBoost 參數調整:
    python train.py --max-depth 6 --learning-rate 0.03
    python train.py --n-estimators 300 --min-child-weight 10
        """
    )
    
    # 基本設定
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
    
    # 生成 run_id
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 載入設定
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            config = ConfigManager(config_path)
            logger.info(f"載入設定檔: {config_path}")
        else:
            logger.warning(f"設定檔不存在: {config_path}，使用預設設定")
    
    # 組合 XGBoost 參數
    xgb_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "use_label_encoder": False,
        "n_jobs": -1,
    }
    
    # 顯示設定
    logger.info("=" * 70)
    logger.info("Credit Scoring Model Training")
    logger.info("=" * 70)
    logger.info(f"Config Version: {CONFIG_VERSION}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Calibration: {args.calibration}")
    logger.info(f"CV Folds: {args.n_splits}")
    logger.info("-" * 70)
    logger.info("XGBoost Parameters:")
    for k, v in xgb_params.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 70)
    
    # 建立 Spark Session
    spark = create_spark_session()
    
    try:
        # 執行訓練
        model_path = run_model_training(
            project_root=project_root,
            spark=spark,
            config=config,
            run_id=run_id,
            n_splits=args.n_splits,
            calibration_method=args.calibration,
            xgb_params=xgb_params,
        )
        
        logger.info("\n" + "=" * 70)
        logger.info("訓練完成！")
        logger.info("=" * 70)
        logger.info(f"模型儲存於: {model_path}")
        logger.info("")
        logger.info("輸出檔案:")
        logger.info(f"  - model.pkl              (Calibrated Model)")
        logger.info(f"  - base_model.pkl         (Base XGBoost for SHAP)")
        logger.info(f"  - model_artifact.json    (Metadata)")
        logger.info(f"  - feature_importance.csv (特徵重要性)")
        logger.info(f"  - training_report.json   (訓練報告)")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"訓練失敗: {e}")
        raise
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
