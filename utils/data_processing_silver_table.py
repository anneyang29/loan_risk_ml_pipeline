"""
Silver Layer: 資料清理與特徵工程
================================
從 Bronze 讀取資料，進行：
- Schema Validation（資料契約驗證）
- 資料型別轉換
- 缺失值處理
- 特徵編碼（二元、序位）
- Log Transform
- 去重複

Version: 1.0.0
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType

from .config import ConfigManager, default_config, CONFIG_VERSION
from .schema_validation import (
    DataValidator, ValidationReport, 
    check_schema_compatibility
)
from .monitoring import AuditLogger, AuditRecord
from .transformation_artifacts import ArtifactManager, OrdinalEncodingArtifact

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "silver_layer") -> SparkSession:
    """建立 Spark Session"""
    return SparkSession.builder.appName(app_name).getOrCreate()


def clean_column_names(df: DataFrame) -> DataFrame:
    """清理欄位名稱（去除空白）"""
    for c in df.columns:
        df = df.withColumnRenamed(c, c.strip())
    return df


def convert_date_column(df: DataFrame) -> DataFrame:
    """轉換日期欄位"""
    df = df.withColumn(
        "進件日",
        F.to_date(F.col("進件日").cast("string"), "yyyyMMdd")
    )
    return df


def convert_numeric_columns(df: DataFrame) -> DataFrame:
    """
    轉換數值欄位型別
    - 整數欄位: 申辦期數, 年齡, 所留市內電話數, 內部往來次數, 近半年同業查詢次數
    - 浮點數欄位: 原申辦金額, 車齡
    """
    int_cols = ["申辦期數", "年齡", "所留市內電話數", "內部往來次數", "近半年同業查詢次數"]
    double_cols = ["原申辦金額", "車齡"]
    
    # 處理 "NULL" 字串
    for c in int_cols + double_cols:
        if c in df.columns:
            df = df.withColumn(
                c,
                F.when(
                    (F.upper(F.col(c)) == "NULL") | (F.col(c) == ""),
                    None
                ).otherwise(F.col(c))
            )
    
    # 型別轉換
    for c in int_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(IntegerType()))
    
    for c in double_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))
    
    return df


def clean_category_columns(df: DataFrame) -> DataFrame:
    """清理類別欄位（去空白、空字串轉 NULL）"""
    category_cols = [
        "性別", "教育程度", "婚姻狀況", "職業說明", 
        "居住地", "廠牌車型", "動產設定", "授信結果"
    ]
    
    for c in category_cols:
        if c in df.columns:
            df = df.withColumn(c, F.trim(F.col(c)))
            df = df.withColumn(
                c,
                F.when((F.col(c) == "") | (F.col(c) == " "), None).otherwise(F.col(c))
            )
    
    # 移除「成功案例」欄位（若存在）
    if "成功案例" in df.columns:
        df = df.drop("成功案例")
        logger.info("已移除「成功案例」欄位")
    
    return df


def drop_missing_key_rows(df: DataFrame) -> DataFrame:
    """刪除缺少 key/date/label 的資料"""
    before_count = df.count()
    df = df.dropna(subset=["案件編號", "進件日", "授信結果"])
    after_count = df.count()
    logger.info(f"刪除缺失 key/date/label: {before_count} → {after_count} ({before_count - after_count} 筆)")
    return df


def deduplicate(df: DataFrame) -> DataFrame:
    """去重複：每個案件編號只保留最新一筆"""
    before_count = df.count()
    
    window_spec = Window.partitionBy("案件編號").orderBy(F.col("進件日").desc())
    df = df.withColumn("rn", F.row_number().over(window_spec))
    df = df.filter(F.col("rn") == 1).drop("rn")
    
    after_count = df.count()
    logger.info(f"去重複: {before_count} → {after_count} ({before_count - after_count} 筆)")
    return df


def encode_target(df: DataFrame) -> DataFrame:
    """編碼目標變數：核准=1, 婉拒=0"""
    df = df.withColumn(
        "授信結果_二元",
        F.when(F.col("授信結果") == "APP(核准)", 1)
         .when(F.col("授信結果") == "WTCD(婉拒)", 0)
         .otherwise(None)
    )
    return df


def handle_education_missing(df: DataFrame) -> DataFrame:
    """處理教育程度缺失：用職業推斷"""
    # 缺失旗標
    df = df.withColumn(
        "教育程度_是否缺失",
        F.when(F.col("教育程度").isNull(), 1).otherwise(0)
    )
    
    # 用職業推斷補值
    df = df.withColumn(
        "教育程度_補值後",
        F.when(F.col("教育程度").isNotNull(), F.col("教育程度"))
         .when(F.col("職業說明").contains("學生(大專生)"), F.lit("大學"))
         .when(F.col("職業說明").contains("學生(高中職生)"), F.lit("高中"))
         .otherwise(F.col("教育程度"))
    )
    
    return df


def fill_missing_with_label(df: DataFrame) -> DataFrame:
    """類別欄位缺失填補為 'Missing'"""
    fill_cols = ["教育程度", "婚姻狀況", "職業說明", "居住地", "性別", "月所得"]
    
    for c in fill_cols:
        if c in df.columns:
            df = df.withColumn(c, F.coalesce(F.col(c), F.lit("Missing")))
    
    return df


def process_car_age(df: DataFrame) -> DataFrame:
    """處理車齡：-1=缺失, >100=異常"""
    # 保留原始值
    df = df.withColumn("車齡_原始", F.col("車齡"))
    
    # 缺失旗標
    df = df.withColumn(
        "車齡_是否缺失",
        F.when(F.col("車齡") == -1, 1).otherwise(0)
    )
    
    # 異常旗標
    df = df.withColumn(
        "車齡_異常旗標",
        F.when(F.col("車齡") > 100, 1).otherwise(0)
    )
    
    # 清理後的值
    df = df.withColumn(
        "車齡_清理後",
        F.when(F.col("車齡") == -1, None)
         .when(F.col("車齡") > 100, None)
         .when(F.col("車齡") < 0, None)
         .otherwise(F.col("車齡"))
    )
    
    return df


def encode_binary_features(df: DataFrame) -> DataFrame:
    """二元特徵編碼：性別、婚姻狀況"""
    df = df.withColumn(
        "性別_二元",
        F.when(F.col("性別") == "男", 1)
         .when(F.col("性別") == "女", 0)
         .otherwise(0)
    )
    
    df = df.withColumn(
        "婚姻狀況_二元",
        F.when(F.col("婚姻狀況") == "已婚", 1)
         .when(F.col("婚姻狀況") == "未婚", 0)
         .otherwise(0)
    )
    
    return df


def encode_education_ordinal(df: DataFrame, config: ConfigManager = None) -> DataFrame:
    """
    教育程度序位編碼
    
    使用 config 中定義的映射，確保與 transformation_artifacts 一致
    """
    if config is None:
        config = default_config
    
    # 從 config 取得映射
    edu_mapping = config.feature_encoding.education_ordinal_mapping
    
    # 建立 mapping expression
    mapping_expr = F.lit(0)  # 預設值
    for edu_level, ordinal in edu_mapping.items():
        mapping_expr = F.when(
            F.col("教育程度") == F.lit(edu_level), F.lit(ordinal)
        ).otherwise(mapping_expr)
    
    df = df.withColumn("教育程度_序位", mapping_expr)
    return df


def encode_income_ordinal(df: DataFrame, config: ConfigManager = None) -> DataFrame:
    """
    月所得序位編碼
    
    使用 config 中定義的映射，確保與 transformation_artifacts 一致
    """
    if config is None:
        config = default_config
    
    # 合併細分與粗分映射
    income_mapping = {}
    income_mapping.update(config.feature_encoding.income_ordinal_mapping_detailed)
    income_mapping.update(config.feature_encoding.income_ordinal_mapping_coarse)
    
    # 建立 mapping expression
    mapping_expr = F.lit(0)  # 預設值
    for income_level, ordinal in income_mapping.items():
        mapping_expr = F.when(
            F.col("月所得") == F.lit(income_level), F.lit(ordinal)
        ).otherwise(mapping_expr)
    
    df = df.withColumn("月所得_序位", mapping_expr)
    
    # 缺失旗標
    df = df.withColumn(
        "月所得_是否缺失",
        F.when(F.col("月所得") == "Missing", 1).otherwise(0)
    )
    
    return df


def encode_age_group(df: DataFrame, config: ConfigManager = None) -> DataFrame:
    """
    年齡組編碼
    
    使用 config 中定義的映射，確保一致性
    """
    if config is None:
        config = default_config
    
    # 從 config 取得年齡組映射
    age_mapping = config.feature_encoding.age_group_mapping
    
    # 年齡組分類
    df = df.withColumn(
        "年齡組",
        F.when(F.col("年齡") < 21, "~20")
         .when((F.col("年齡") >= 21) & (F.col("年齡") < 31), "21-30")
         .when((F.col("年齡") >= 31) & (F.col("年齡") < 41), "31-40")
         .when((F.col("年齡") >= 41) & (F.col("年齡") < 51), "41-50")
         .when((F.col("年齡") >= 51) & (F.col("年齡") < 61), "51-60")
         .otherwise("61~")
    )
    
    # 從 config 取得序位映射
    ordinal_mapping_expr = F.lit(0)
    for age_group, info in age_mapping.items():
        ordinal_mapping_expr = F.when(
            F.col("年齡組") == F.lit(age_group), F.lit(info["ordinal"])
        ).otherwise(ordinal_mapping_expr)
    
    df = df.withColumn("年齡組_序位", ordinal_mapping_expr)
    
    return df


def process_count_features(df: DataFrame) -> DataFrame:
    """處理次數類特徵：內部往來次數、近半年同業查詢次數、所留市內電話數"""
    # 內部往來次數
    df = df.withColumn(
        "內部往來次數_是否特殊值",
        F.when(F.col("內部往來次數") == -1, 1).otherwise(0)
    )
    df = df.withColumn(
        "內部往來次數_清理後",
        F.when(F.col("內部往來次數") < 0, 0).otherwise(F.col("內部往來次數"))
    )
    
    # 近半年同業查詢次數
    df = df.withColumn(
        "近半年同業查詢次數_是否缺失",
        F.when(F.col("近半年同業查詢次數") == -1, 1).otherwise(0)
    )
    df = df.withColumn(
        "近半年同業查詢次數_清理後",
        F.when(F.col("近半年同業查詢次數") < 0, 0).otherwise(F.col("近半年同業查詢次數"))
    )
    
    # 所留市內電話數
    df = df.withColumn(
        "所留市內電話數_清理後",
        F.when(F.col("所留市內電話數") < 0, 0).otherwise(F.col("所留市內電話數"))
    )
    
    return df


def apply_log_transform(df: DataFrame) -> DataFrame:
    """Log Transform：處理右偏分佈"""
    # 原申辦金額
    df = df.withColumn("原申辦金額_log", F.log1p(F.col("原申辦金額")))
    
    # 車齡（缺失填 0）
    df = df.withColumn("車齡_log", F.log1p(F.coalesce(F.col("車齡_清理後"), F.lit(0.0))))
    
    # 次數類
    df = df.withColumn("內部往來次數_log", F.log1p(F.col("內部往來次數_清理後")))
    df = df.withColumn("近半年同業查詢次數_log", F.log1p(F.col("近半年同業查詢次數_清理後")))
    df = df.withColumn("所留市內電話數_log", F.log1p(F.col("所留市內電話數_清理後")))
    
    return df


def run_silver_pipeline(
    project_root: Path, 
    spark: SparkSession = None,
    config: ConfigManager = None,
    run_id: str = None,
    validate_input: bool = True,
    fail_on_validation_error: bool = False
) -> Path:
    """
    執行 Silver Layer Pipeline（企業級版本）
    
    Args:
        project_root: 專案根目錄
        spark: 可選的 SparkSession（若不提供則自動建立）
        config: 設定管理器
        run_id: 執行 ID
        validate_input: 是否驗證輸入資料
        fail_on_validation_error: 驗證失敗時是否停止
        
    Returns:
        Silver Parquet 輸出路徑
    """
    # 設定
    if config is None:
        config = default_config
    
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 路徑設定
    bronze_path = project_root / "datamart" / "bronze" / "application"
    silver_output_path = project_root / "datamart" / "silver" / "application"
    audit_path = project_root / "datamart" / "silver" / "audit"
    
    # 確保輸出目錄存在
    silver_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化 Audit Logger
    audit_logger = AuditLogger(audit_path)
    audit_record = AuditRecord(
        run_id=run_id,
        stage="silver",
        action="run_silver_pipeline",
        input_path=str(bronze_path),
        output_path=str(silver_output_path),
        config_version=CONFIG_VERSION,
    )
    
    # 建立 Spark Session
    should_stop_spark = False
    if spark is None:
        spark = create_spark_session()
        should_stop_spark = True
    
    try:
        # ============================================
        # 1. 讀取 Bronze
        # ============================================
        logger.info(f"讀取 Bronze: {bronze_path}")
        df = spark.read.parquet(str(bronze_path))
        initial_count = df.count()
        logger.info(f"Bronze 筆數: {initial_count}")
        
        audit_record.row_count_before = initial_count
        
        # ============================================
        # 2. Schema Validation（資料契約驗證）
        # ============================================
        if validate_input:
            logger.info("執行 Schema Validation...")
            validator = DataValidator(config)
            validation_report = validator.validate_bronze(df)
            
            # 儲存驗證報告
            report_path = project_root / "datamart" / "silver" / "reports" / f"validation_{run_id}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(validation_report.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 記錄 NULL summary
            audit_record.null_summary = validation_report.null_summary
            
            # 檢查是否有 FATAL 錯誤
            if validation_report.overall_status == "FAIL":
                error_msg = f"Validation failed: {len(validation_report.fatal_errors)} fatal errors"
                audit_record.errors.append(error_msg)
                
                if fail_on_validation_error:
                    audit_logger.add_record(audit_record)
                    raise ValueError(error_msg)
                else:
                    logger.warning(f"⚠️ {error_msg}，但繼續處理...")
            
            # 記錄警告
            for warning in validation_report.warnings:
                audit_record.warnings.append(warning.message)
        
        # ============================================
        # 3. 資料清理
        # ============================================
        logger.info("開始資料清理...")
        df = clean_column_names(df)
        df = convert_date_column(df)
        df = convert_numeric_columns(df)
        df = clean_category_columns(df)
        
        # 記錄刪除原因
        before_drop = df.count()
        df = drop_missing_key_rows(df)
        after_drop = df.count()
        audit_record.drop_reasons["missing_key_date_label"] = before_drop - after_drop
        
        before_dedup = df.count()
        df = deduplicate(df)
        after_dedup = df.count()
        audit_record.drop_reasons["duplicate"] = before_dedup - after_dedup
        
        # ============================================
        # 4. 特徵工程
        # ============================================
        logger.info("開始特徵工程...")
        df = encode_target(df)
        df = handle_education_missing(df)
        df = fill_missing_with_label(df)
        df = process_car_age(df)
        df = encode_binary_features(df)
        df = encode_education_ordinal(df, config)  # 使用 config 中的映射
        df = encode_income_ordinal(df, config)      # 使用 config 中的映射
        df = encode_age_group(df, config)           # 使用 config 中的映射
        df = process_count_features(df)
        df = apply_log_transform(df)
        
        # 新增處理時間戳記
        df = df.withColumn("silver_process_timestamp", F.current_timestamp())
        
        # ============================================
        # 5. 輸出前驗證
        # ============================================
        if validate_input:
            logger.info("執行輸出前驗證...")
            validator = DataValidator(config)
            output_report = validator.validate_silver(df)
            
            if output_report.overall_status == "FAIL":
                for error in output_report.errors:
                    audit_record.errors.append(error.message)
                logger.warning("⚠️ Silver 輸出驗證有錯誤")
        
        # ============================================
        # 6. 儲存 Silver
        # ============================================
        logger.info(f"儲存 Silver 至: {silver_output_path}")
        df.write.mode("overwrite").parquet(str(silver_output_path))
        
        # 驗證
        df_check = spark.read.parquet(str(silver_output_path))
        final_count = df_check.count()
        logger.info(f"Silver 總欄位數: {len(df_check.columns)}")
        logger.info(f"Silver 總筆數: {final_count}")
        
        audit_record.row_count_after = final_count
        audit_record.rows_dropped = initial_count - final_count
        
        # ============================================
        # 7. 儲存 Audit Log
        # ============================================
        audit_logger.add_record(audit_record)
        
        return silver_output_path
        
    except Exception as e:
        audit_record.errors.append(str(e))
        audit_logger.add_record(audit_record)
        raise
        
    finally:
        if should_stop_spark:
            spark.stop()


# 向後相容：保留原有簽名
def run_silver_pipeline_simple(project_root: Path, spark: SparkSession = None) -> Path:
    """簡化版本（向後相容）"""
    return run_silver_pipeline(
        project_root, 
        spark=spark, 
        validate_input=False
    )


if __name__ == "__main__":
    # 取得專案根目錄
    project_root = Path(__file__).parent.parent
    
    logger.info("=" * 60)
    logger.info("開始執行 Silver Layer Pipeline")
    logger.info("=" * 60)
    
    output_path = run_silver_pipeline(project_root)
    
    logger.info("=" * 60)
    logger.info(f"✓ Silver Layer 完成！輸出: {output_path}")
    logger.info("=" * 60)
