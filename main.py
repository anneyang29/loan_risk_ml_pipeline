"""
ML Pipeline - Main Entry Point
==============================
執行完整的 Bronze → Silver → Gold Data Pipeline
支援企業級資料治理功能

For model training, use train.py instead.

Usage:
    python main.py                    # 執行全部（含 validation + artifacts）
    python main.py bronze             # 只執行 Bronze
    python main.py silver             # 只執行 Silver
    python main.py gold               # 只執行 Gold
    python main.py bronze silver      # 執行 Bronze + Silver
    python main.py --simple           # 簡化模式（不含 validation）
    python main.py --config config/pipeline_config.yaml  # 指定設定檔
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

from pyspark.sql import SparkSession

from utils.data_processing_bronze_table import run_bronze_pipeline
from utils.data_processing_silver_table import run_silver_pipeline
from utils.data_processing_gold_table import run_gold_pipeline
from utils.config import ConfigManager, CONFIG_VERSION

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_spark_session() -> SparkSession:
    """建立共用的 Spark Session"""
    return SparkSession.builder \
        .appName("ml_pipeline") \
        .getOrCreate()


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="ML Pipeline - Data Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                          # 執行全部 data pipeline
    python main.py silver gold              # 只執行 Silver + Gold
    python main.py --simple                 # 簡化模式（不含 validation）
    python main.py --config config/dev.yaml # 使用指定設定檔
    python main.py --no-drift               # 不執行漂移檢查
    
For model training:
    python train.py                         # 訓練模型
    python train.py --calibration sigmoid   # 使用 sigmoid calibration
        """
    )
    
    parser.add_argument(
        'stages', 
        nargs='*', 
        default=None,  # 改為 None，後面處理
        help='要執行的 pipeline 階段 (bronze, silver, gold)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='設定檔路徑 (YAML/JSON)'
    )
    
    parser.add_argument(
        '--simple', '-s',
        action='store_true',
        help='簡化模式：不執行 validation 和 artifact 儲存'
    )
    
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='不執行 schema validation'
    )
    
    parser.add_argument(
        '--no-artifacts',
        action='store_true',
        help='不儲存 transformation artifacts'
    )
    
    parser.add_argument(
        '--no-drift',
        action='store_true',
        help='不執行 drift check'
    )
    
    parser.add_argument(
        '--fail-on-error',
        action='store_true',
        help='validation 失敗時停止 pipeline'
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='指定 run ID（預設自動生成）'
    )
    
    return parser.parse_args()


def main():
    # 解析參數
    args = parse_args()
    
    # 取得專案根目錄
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
    
    # 決定執行模式
    if args.simple:
        validate_input = False
        save_artifacts = False
        run_drift_check = False
    else:
        validate_input = not args.no_validation
        save_artifacts = not args.no_artifacts
        run_drift_check = not args.no_drift
    
    fail_on_validation_error = args.fail_on_error
    
    # 處理 stages（若未指定則執行全部）
    if args.stages:
        stages = [s.lower() for s in args.stages]
        # 驗證 stages
        valid_stages = {'bronze', 'silver', 'gold'}
        invalid = set(stages) - valid_stages
        if invalid:
            logger.error(f"無效的 stage: {invalid}，有效選項為: {valid_stages}")
            sys.exit(1)
    else:
        stages = ['bronze', 'silver', 'gold']
    
    logger.info("=" * 70)
    logger.info("ML Pipeline - Data Processing")
    logger.info("=" * 70)
    logger.info(f"Config Version: {CONFIG_VERSION}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"專案根目錄: {project_root}")
    logger.info(f"執行階段: {stages}")
    logger.info(f"Validation: {validate_input}")
    logger.info(f"Save Artifacts: {save_artifacts}")
    logger.info(f"Drift Check: {run_drift_check}")
    logger.info("=" * 70)
    
    # 建立共用的 Spark Session
    spark = create_spark_session()
    
    try:
        # Bronze Layer
        if "bronze" in stages:
            logger.info("\n" + "=" * 70)
            logger.info("【Bronze Layer】開始執行...")
            logger.info("=" * 70)
            
            bronze_path = run_bronze_pipeline(project_root, spark=spark)
            logger.info(f"✓ Bronze 完成: {bronze_path}")
        
        # Silver Layer
        if "silver" in stages:
            logger.info("\n" + "=" * 70)
            logger.info("【Silver Layer】開始執行...")
            logger.info("=" * 70)
            
            silver_path = run_silver_pipeline(
                project_root, 
                spark=spark,
                config=config,
                run_id=run_id,
                validate_input=validate_input,
                fail_on_validation_error=fail_on_validation_error
            )
            logger.info(f"✓ Silver 完成: {silver_path}")
        
        # Gold Layer
        if "gold" in stages:
            logger.info("\n" + "=" * 70)
            logger.info("【Gold Layer】開始執行...")
            logger.info("=" * 70)
            
            output_paths = run_gold_pipeline(
                project_root, 
                spark=spark,
                config=config,
                run_id=run_id,
                save_artifacts=save_artifacts,
                run_drift_check=run_drift_check
            )
            logger.info("✓ Gold 完成!")
            for name, path in output_paths.items():
                logger.info(f"  - {name}: {path}")
        
        logger.info("\n" + "=" * 70)
        logger.info(f" Data Pipeline 執行完成！ (Run ID: {run_id})")
        logger.info("=" * 70)
        logger.info("")
        logger.info("下一步：執行模型訓練")
        logger.info("  python train.py --run-id " + run_id)
        logger.info("=" * 70)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
