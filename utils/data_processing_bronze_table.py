"""
Bronze Layer: 原始資料載入與儲存
================================
將 CSV 原始資料轉換為 Parquet 格式，保留所有原始欄位。
"""

import logging
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "bronze_layer") -> SparkSession:
    """建立 Spark Session"""
    return SparkSession.builder.appName(app_name).getOrCreate()


def load_raw_csv(spark: SparkSession, raw_file_path: Path) -> DataFrame:
    """
    載入原始 CSV 檔案
    
    Args:
        spark: SparkSession
        raw_file_path: 原始 CSV 檔案路徑
        
    Returns:
        PySpark DataFrame
    """
    logger.info(f"載入原始資料: {raw_file_path}")
    
    df = (
        spark.read
        .option("header", True)
        .option("encoding", "UTF-8")
        .csv(str(raw_file_path))
    )
    
    # 新增載入時間戳記
    df = df.withColumn("bronze_load_timestamp", F.current_timestamp())
    
    row_count = df.count()
    logger.info(f"原始資料筆數: {row_count}")
    
    return df


def save_bronze_parquet(df: DataFrame, output_path: Path) -> None:
    """
    儲存 Bronze Layer Parquet
    
    Args:
        df: PySpark DataFrame
        output_path: 輸出路徑
    """
    logger.info(f"儲存 Bronze Layer 至: {output_path}")
    df.write.mode("overwrite").parquet(str(output_path))
    logger.info("Bronze Layer 儲存完成")


def run_bronze_pipeline(
    project_root: Path,
    raw_filename: str = "application_data.csv",
    spark: SparkSession = None
) -> Path:
    """
    執行 Bronze Layer Pipeline
    
    Args:
        project_root: 專案根目錄
        raw_filename: 原始檔案名稱
        spark: 可選的 SparkSession（若不提供則自動建立）
        
    Returns:
        Bronze Parquet 輸出路徑
    """
    # 路徑設定
    raw_file_path = project_root / "data" / "raw" / raw_filename
    bronze_output_path = project_root / "datamart" / "bronze" / "application"
    
    # 確保輸出目錄存在
    bronze_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 建立 Spark Session
    should_stop_spark = False
    if spark is None:
        spark = create_spark_session()
        should_stop_spark = True
    
    try:
        # 載入原始資料
        df_bronze = load_raw_csv(spark, raw_file_path)
        
        # 顯示 Schema
        logger.info("Bronze Schema:")
        df_bronze.printSchema()
        
        # 儲存 Bronze
        save_bronze_parquet(df_bronze, bronze_output_path)
        
        # 驗證
        df_check = spark.read.parquet(str(bronze_output_path))
        logger.info(f"驗證讀取筆數: {df_check.count()}")
        
        return bronze_output_path
        
    finally:
        if should_stop_spark:
            spark.stop()


if __name__ == "__main__":
    # 取得專案根目錄
    project_root = Path(__file__).parent.parent
    
    logger.info("=" * 60)
    logger.info("開始執行 Bronze Layer Pipeline")
    logger.info("=" * 60)
    
    output_path = run_bronze_pipeline(project_root)
    
    logger.info("=" * 60)
    logger.info(f"✓ Bronze Layer 完成！輸出: {output_path}")
    logger.info("=" * 60)
