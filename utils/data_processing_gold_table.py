"""
Gold Layer: 模型特徵準備與資料切分
==================================
從 Silver 讀取資料，進行：
- Dev / OOT 切分
- Frequency Encoding（高基數類別）
- Cross Features（交互特徵）
- MinMax Scaling（標準化）
- Rolling Windows 定義
- Transformation Artifacts 儲存
- Data Drift 監控

Version: 1.0.0
"""

import logging
from pathlib import Path
from datetime import date, datetime
from typing import Dict, List, Tuple, Optional
import uuid

import pandas as pd
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from .config import (
    ConfigManager, default_config, CONFIG_VERSION,
    FeatureDefinitionConfig, TimePeriodConfig
)
from .schema_validation import DataValidator, ValidationReport
from .transformation_artifacts import (
    ArtifactManager, TransformationPackage,
    ScalerArtifact, FrequencyEncodingArtifact
)
from .monitoring import (
    DriftMonitor, DriftReport, AuditLogger, AuditRecord,
    check_unseen_categories, compute_feature_statistics
)

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================
# 從 Config 載入特徵定義（避免 hardcode）
# ============================================
def get_feature_config(config: ConfigManager = None) -> FeatureDefinitionConfig:
    """取得特徵定義設定"""
    if config is None:
        config = default_config
    return config.feature_definition


def get_time_config(config: ConfigManager = None) -> TimePeriodConfig:
    """取得時間區間設定"""
    if config is None:
        config = default_config
    return config.time_period


# 為了向後相容，保留原有常數（但從 config 取值）
_default_feature_config = get_feature_config()

# 數值特徵（需要 MinMaxScaler）
NUMERIC_FEATURES_TO_SCALE = _default_feature_config.numeric_features_to_scale

# 序位特徵（已編碼，不需要 scale）
ORDINAL_FEATURES = _default_feature_config.ordinal_features

# 二元特徵（0/1，不需要 scale）
BINARY_FEATURES = _default_feature_config.binary_features

# 高基數類別（用 Frequency Encoding）
HIGH_CARDINALITY_FEATURES = _default_feature_config.high_cardinality_features

# 目標變數
TARGET_COL = _default_feature_config.target_column

# Key 欄位
KEY_COLS = _default_feature_config.key_columns


def create_spark_session(app_name: str = "gold_layer") -> SparkSession:
    """建立 Spark Session"""
    return SparkSession.builder.appName(app_name).getOrCreate()


def split_dev_oot(
    df: DataFrame,
    oot_start: date = date(2025, 10, 1)
) -> Tuple[DataFrame, DataFrame]:
    """
    切分 Development 和 OOT 資料集
    
    Args:
        df: Silver DataFrame
        oot_start: OOT 開始日期
        
    Returns:
        (df_dev, df_oot)
    """
    df_dev = df.filter(F.col("進件日") < F.lit(oot_start))
    df_oot = df.filter(F.col("進件日") >= F.lit(oot_start))
    
    logger.info(f"Development 筆數: {df_dev.count()}")
    logger.info(f"OOT 筆數: {df_oot.count()}")
    
    return df_dev, df_oot


def define_rolling_windows(
    data_start: date = date(2024, 4, 1),
    oot_start: date = date(2025, 10, 1),
    train_months: int = 4,
    monitor_months: int = 2,
    step_months: int = 2
) -> List[Dict]:
    """
    定義 Rolling Windows
    
    Args:
        data_start: 資料開始日期
        oot_start: OOT 開始日期
        train_months: 訓練期月數
        monitor_months: 監控期月數
        step_months: 滾動步長
        
    Returns:
        Rolling windows 定義列表
    """
    rolling_windows = []
    current_start = data_start
    window_id = 1
    
    while True:
        train_end = current_start + relativedelta(months=train_months) - relativedelta(days=1)
        monitor_start = train_end + relativedelta(days=1)
        monitor_end = monitor_start + relativedelta(months=monitor_months) - relativedelta(days=1)
        
        if monitor_end >= oot_start:
            break
        
        rolling_windows.append({
            "window_id": window_id,
            "train_start": current_start,
            "train_end": train_end,
            "monitor_start": monitor_start,
            "monitor_end": monitor_end
        })
        
        current_start = current_start + relativedelta(months=step_months)
        window_id += 1
    
    logger.info(f"定義了 {len(rolling_windows)} 個 Rolling Windows")
    return rolling_windows


def apply_frequency_encoding(
    df_dev: DataFrame,
    df_oot: DataFrame,
    columns: List[str]
) -> Tuple[DataFrame, DataFrame, List[str]]:
    """
    對高基數類別欄位做 Frequency Encoding
    
    Args:
        df_dev: Development DataFrame
        df_oot: OOT DataFrame
        columns: 要編碼的欄位列表
        
    Returns:
        (df_dev, df_oot, frequency_feature_names)
    """
    frequency_features = []
    
    for col in columns:
        if col in df_dev.columns:
            # 計算 Development 中每個類別的頻率
            freq_df = df_dev.groupBy(col).count()
            total = df_dev.count()
            freq_col_name = f"{col}_頻率"
            
            freq_df = freq_df.withColumn(
                freq_col_name,
                F.col("count") / F.lit(total)
            ).select(col, freq_col_name)
            
            # Join 回 df_dev 和 df_oot
            df_dev = df_dev.join(freq_df, on=col, how="left")
            df_oot = df_oot.join(freq_df, on=col, how="left")
            
            # OOT 中未見過的類別，頻率設為 0
            df_oot = df_oot.withColumn(
                freq_col_name,
                F.coalesce(F.col(freq_col_name), F.lit(0.0))
            )
            
            frequency_features.append(freq_col_name)
            logger.info(f"✓ {col} 完成 Frequency Encoding")
    
    return df_dev, df_oot, frequency_features


def create_cross_features(
    df_dev: DataFrame,
    df_oot: DataFrame,
    config: ConfigManager = None
) -> Tuple[DataFrame, DataFrame, List[str]]:
    """
    建立交互特徵
    
    修正說明：
    - 負債月所得比：使用估計月所得中位數（而非序位值）計算，
      確保財務意義正確
    
    Returns:
        (df_dev, df_oot, cross_feature_names)
    """
    if config is None:
        config = default_config
    
    # 取得月所得中位數映射
    income_midpoint = config.feature_encoding.income_midpoint_mapping
    
    # 1. 負債月所得比（使用估計月所得，而非序位）
    # 先新增估計月所得欄位
    income_mapping_expr = F.lit(income_midpoint.get("Missing", 30000.0))
    for income_level, midpoint in income_midpoint.items():
        income_mapping_expr = F.when(
            F.col("月所得") == F.lit(income_level), F.lit(midpoint)
        ).otherwise(income_mapping_expr)
    
    df_dev = df_dev.withColumn("月所得_估計值", income_mapping_expr)
    df_oot = df_oot.withColumn("月所得_估計值", income_mapping_expr)
    
    # 計算負債月所得比（金額 / 估計月所得）
    df_dev = df_dev.withColumn(
        "負債月所得比",
        F.when(F.col("月所得_估計值") > 0, F.col("原申辦金額") / F.col("月所得_估計值"))
         .otherwise(F.col("原申辦金額") / F.lit(30000.0))  # 預設分母
    )
    df_oot = df_oot.withColumn(
        "負債月所得比",
        F.when(F.col("月所得_估計值") > 0, F.col("原申辦金額") / F.col("月所得_估計值"))
         .otherwise(F.col("原申辦金額") / F.lit(30000.0))
    )
    
    # 2. 年齡 x 婚姻狀況
    df_dev = df_dev.withColumn("年齡_婚姻交互", F.col("年齡") * F.col("婚姻狀況_二元"))
    df_oot = df_oot.withColumn("年齡_婚姻交互", F.col("年齡") * F.col("婚姻狀況_二元"))
    
    # 3. 車齡 x 申辦金額（車齡缺失時用 0）
    df_dev = df_dev.withColumn(
        "車齡_金額交互",
        F.coalesce(F.col("車齡_清理後"), F.lit(0.0)) * F.col("原申辦金額_log")
    )
    df_oot = df_oot.withColumn(
        "車齡_金額交互",
        F.coalesce(F.col("車齡_清理後"), F.lit(0.0)) * F.col("原申辦金額_log")
    )
    
    # 4. 教育程度 x 月所得
    df_dev = df_dev.withColumn("教育_所得交互", F.col("教育程度_序位") * F.col("月所得_序位"))
    df_oot = df_oot.withColumn("教育_所得交互", F.col("教育程度_序位") * F.col("月所得_序位"))
    
    cross_features = [
        "負債月所得比",      # 修正：使用估計月所得
        "年齡_婚姻交互",
        "車齡_金額交互",
        "教育_所得交互"
    ]
    
    logger.info(f"✓ 新增交互特徵: {cross_features}")
    logger.info("  - 負債月所得比：使用估計月所得中位數計算，確保財務意義")
    
    return df_dev, df_oot, cross_features


def apply_minmax_scaling(
    df_dev: DataFrame,
    df_oot: DataFrame,
    columns: List[str]
) -> Tuple[DataFrame, DataFrame, List[str], Dict]:
    """
    MinMax 標準化（用 Development 的 min/max 套用到 OOT）
    
    Returns:
        (df_dev, df_oot, scaled_feature_names, scale_stats)
    """
    scale_stats = {}
    scaled_features = []
    
    # 計算 Development 的 min/max
    for col in columns:
        if col in df_dev.columns:
            stats = df_dev.select(
                F.min(col).alias("min_val"),
                F.max(col).alias("max_val")
            ).collect()[0]
            
            scale_stats[col] = {
                "min": stats["min_val"] if stats["min_val"] is not None else 0,
                "max": stats["max_val"] if stats["max_val"] is not None else 1
            }
    
    # 套用標準化
    for col in columns:
        if col in scale_stats and col in df_dev.columns:
            min_val = scale_stats[col]["min"]
            max_val = scale_stats[col]["max"]
            range_val = max_val - min_val if max_val != min_val else 1
            
            scaled_col = f"{col}_scaled"
            
            df_dev = df_dev.withColumn(
                scaled_col,
                (F.col(col) - F.lit(min_val)) / F.lit(range_val)
            )
            
            df_oot = df_oot.withColumn(
                scaled_col,
                (F.col(col) - F.lit(min_val)) / F.lit(range_val)
            )
            
            scaled_features.append(scaled_col)
    
    logger.info(f"✓ MinMaxScaler 完成，共 {len(scaled_features)} 個標準化特徵")
    return df_dev, df_oot, scaled_features, scale_stats


def prepare_final_features(
    df_dev: DataFrame,
    df_oot: DataFrame,
    ordinal_features: List[str],
    binary_features: List[str],
    scaled_features: List[str],
    key_cols: List[str],
    target_col: str
) -> Tuple[DataFrame, DataFrame]:
    """
    準備最終模型特徵，進行 NULL 檢查與填補
    """
    all_features = ordinal_features + binary_features + scaled_features
    
    # 檢查 NULL
    logger.info("檢查 NULL 數量...")
    null_found = False
    
    for col in all_features:
        if col in df_dev.columns:
            cnt = df_dev.filter(F.col(col).isNull()).count()
            if cnt > 0:
                logger.warning(f"  [Dev] {col}: {cnt} NULLs")
                null_found = True
        if col in df_oot.columns:
            cnt = df_oot.filter(F.col(col).isNull()).count()
            if cnt > 0:
                logger.warning(f"  [OOT] {col}: {cnt} NULLs")
                null_found = True
    
    # 填補 NULL
    if null_found:
        logger.info("填補 NULL 為 0...")
        for col in all_features:
            if col in df_dev.columns:
                df_dev = df_dev.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))
            if col in df_oot.columns:
                df_oot = df_oot.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))
    else:
        logger.info("✓ 所有特徵都沒有 NULL")
    
    # 選取最終欄位
    final_cols = key_cols + all_features + [target_col]
    
    df_dev_final = df_dev.select(*[c for c in final_cols if c in df_dev.columns])
    df_oot_final = df_oot.select(*[c for c in final_cols if c in df_oot.columns])
    
    logger.info(f"最終特徵數: {len(all_features)}")
    logger.info(f"Development 筆數: {df_dev_final.count()}")
    logger.info(f"OOT 筆數: {df_oot_final.count()}")
    
    return df_dev_final, df_oot_final


def save_gold_outputs(
    df_dev_final: DataFrame,
    df_oot_final: DataFrame,
    rolling_windows: List[Dict],
    project_root: Path
) -> Dict[str, Path]:
    """
    儲存 Gold Layer 輸出
    
    Returns:
        輸出路徑字典
    """
    output_paths = {}
    
    # 輸出路徑
    gold_dev_path = project_root / "datamart" / "gold" / "development"
    gold_oot_path = project_root / "datamart" / "gold" / "oot"
    gold_window_csv = project_root / "datamart" / "gold" / "rolling_window_definition.csv"
    
    # 確保目錄存在
    gold_dev_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. 儲存 Development（按月份）
    logger.info("儲存 Development...")
    df_dev_final.write.mode("overwrite").partitionBy("進件年月").parquet(str(gold_dev_path))
    output_paths["development"] = gold_dev_path
    
    # 2. 儲存 OOT（按月份）
    logger.info("儲存 OOT...")
    df_oot_final.write.mode("overwrite").partitionBy("進件年月").parquet(str(gold_oot_path))
    output_paths["oot"] = gold_oot_path
    
    # 3. Rolling Windows 定義
    logger.info("儲存 Rolling Windows 定義...")
    rolling_windows_pd = pd.DataFrame([
        {
            "window_id": w["window_id"],
            "train_start": str(w["train_start"]),
            "train_end": str(w["train_end"]),
            "monitor_start": str(w["monitor_start"]),
            "monitor_end": str(w["monitor_end"])
        }
        for w in rolling_windows
    ])
    rolling_windows_pd.to_csv(str(gold_window_csv), index=False, encoding="utf-8")
    output_paths["rolling_windows"] = gold_window_csv
    
    return output_paths


def run_gold_pipeline(
    project_root: Path, 
    spark: SparkSession = None,
    config: ConfigManager = None,
    run_id: str = None,
    save_artifacts: bool = True,
    run_drift_check: bool = True
) -> Dict[str, Path]:
    """
    執行 Gold Layer Pipeline（企業級版本）
    
    Args:
        project_root: 專案根目錄
        spark: 可選的 SparkSession（若不提供則自動建立）
        config: 設定管理器（若不提供則使用預設）
        run_id: 執行 ID（若不提供則自動生成）
        save_artifacts: 是否儲存 transformation artifacts
        run_drift_check: 是否執行漂移檢查
        
    Returns:
        輸出路徑字典
    """
    # 設定
    if config is None:
        config = default_config
    
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 路徑設定
    silver_path = project_root / "datamart" / "silver" / "application"
    artifacts_path = project_root / "datamart" / "gold" / "artifacts" / run_id
    audit_path = project_root / "datamart" / "gold" / "audit"
    
    # 時間設定（從 config 取得）
    time_config = config.time_period
    data_start = time_config.data_start
    oot_start = time_config.oot_start
    
    # 初始化 Audit Logger
    audit_logger = AuditLogger(audit_path)
    audit_record = AuditRecord(
        run_id=run_id,
        stage="gold",
        action="run_gold_pipeline",
        input_path=str(silver_path),
        config_version=CONFIG_VERSION,
    )
    
    # 建立 Spark Session
    should_stop_spark = False
    if spark is None:
        spark = create_spark_session()
        should_stop_spark = True
    
    try:
        # ============================================
        # 1. 讀取 Silver
        # ============================================
        logger.info(f"讀取 Silver: {silver_path}")
        df = spark.read.parquet(str(silver_path))
        initial_count = df.count()
        logger.info(f"Silver 筆數: {initial_count}")
        
        audit_record.row_count_before = initial_count
        
        # 新增年月欄位
        df = df.withColumn("進件年月", F.date_format(F.col("進件日"), "yyyy-MM"))
        
        # ============================================
        # 2. Dev / OOT 切分
        # ============================================
        logger.info("切分 Dev / OOT...")
        logger.info(f"  - OOT 起始日期: {oot_start} (來自 config)")
        df_dev, df_oot = split_dev_oot(df, oot_start)
        
        # ============================================
        # 3. Rolling Windows 定義
        # ============================================
        rolling_windows = define_rolling_windows(
            data_start, oot_start,
            train_months=time_config.train_months,
            monitor_months=time_config.monitor_months,
            step_months=time_config.step_months
        )
        
        # ============================================
        # 4. Frequency Encoding（with artifact）
        # ============================================
        logger.info("Frequency Encoding...")
        
        # 使用 ArtifactManager 計算並儲存
        artifact_manager = ArtifactManager(config)
        freq_artifact = artifact_manager.compute_frequency_artifact(
            df_dev, 
            config.feature_definition.high_cardinality_features
        )
        
        # 套用 encoding
        df_dev, df_oot, frequency_features = apply_frequency_encoding(
            df_dev, df_oot, 
            config.feature_definition.high_cardinality_features
        )
        
        # ============================================
        # 5. Cross Features
        # ============================================
        logger.info("建立交互特徵...")
        df_dev, df_oot, cross_features = create_cross_features(df_dev, df_oot, config)
        
        # ============================================
        # 6. MinMax Scaling（with artifact）
        # ============================================
        logger.info("MinMax Scaling...")
        scale_cols = (
            config.feature_definition.numeric_features_to_scale + 
            cross_features + 
            frequency_features
        )
        df_dev, df_oot, scaled_features, scale_stats = apply_minmax_scaling(
            df_dev, df_oot, scale_cols
        )
        
        # 建立 Scaler Artifact
        scaler_artifact = ScalerArtifact()
        for col, stats in scale_stats.items():
            scaler_artifact.add_feature(col, stats["min"], stats["max"])
        
        # ============================================
        # 7. 準備最終特徵
        # ============================================
        logger.info("準備最終特徵...")
        df_dev_final, df_oot_final = prepare_final_features(
            df_dev, df_oot,
            config.feature_definition.ordinal_features, 
            config.feature_definition.binary_features, 
            scaled_features,
            config.feature_definition.key_columns, 
            config.feature_definition.target_column
        )
        
        audit_record.row_count_after = df_dev_final.count() + df_oot_final.count()
        
        # ============================================
        # 8. 儲存輸出
        # ============================================
        logger.info("儲存 Gold Layer...")
        output_paths = save_gold_outputs(
            df_dev_final, df_oot_final, rolling_windows, project_root
        )
        
        audit_record.output_path = str(output_paths.get("development", ""))
        
        # ============================================
        # 9. 儲存 Transformation Artifacts
        # ============================================
        if save_artifacts:
            logger.info("儲存 Transformation Artifacts...")
            
            # 建立完整 package
            package = TransformationPackage(
                run_id=run_id,
                scaler=scaler_artifact,
                frequency_encoding=freq_artifact,
                feature_list=scaled_features + 
                             config.feature_definition.ordinal_features + 
                             config.feature_definition.binary_features,
                target_column=config.feature_definition.target_column,
                key_columns=config.feature_definition.key_columns,
                training_row_count=df_dev_final.count(),
            )
            
            # 計算訓練期間
            if "進件日" in df_dev.columns:
                date_range = df_dev.agg(
                    F.min("進件日").alias("min_date"),
                    F.max("進件日").alias("max_date")
                ).collect()[0]
                package.training_period = {
                    "start": str(date_range["min_date"]),
                    "end": str(date_range["max_date"])
                }
            
            # 儲存
            artifacts_path.mkdir(parents=True, exist_ok=True)
            package.save(artifacts_path)
            output_paths["artifacts"] = artifacts_path
            
            logger.info(f"✓ Artifacts 儲存至: {artifacts_path}")
        
        # ============================================
        # 10. 資料漂移檢查
        # ============================================
        if run_drift_check and df_oot_final.count() > 0:
            logger.info("執行資料漂移檢查...")
            
            drift_monitor = DriftMonitor(config)
            drift_report = drift_monitor.generate_drift_report(
                df_dev, df_oot,
                category_columns=config.feature_definition.high_cardinality_features,
                numeric_columns=config.feature_definition.numeric_features_to_scale
            )
            
            # 儲存報告
            drift_report_path = project_root / "datamart" / "gold" / "reports" / f"drift_report_{run_id}.json"
            drift_report_path.parent.mkdir(parents=True, exist_ok=True)
            drift_report.save(drift_report_path)
            output_paths["drift_report"] = drift_report_path
            
            # 未見類別檢查
            unseen_report = check_unseen_categories(
                df_dev, df_oot,
                config.feature_definition.high_cardinality_features
            )
            unseen_report_path = project_root / "datamart" / "gold" / "reports" / f"unseen_categories_{run_id}.json"
            unseen_report.save(unseen_report_path)
            output_paths["unseen_report"] = unseen_report_path
            
            # 記錄警告
            if drift_report.critical_drift_count > 0:
                audit_record.warnings.append(
                    f"Critical drift detected in {drift_report.critical_drift_count} features"
                )
        
        # ============================================
        # 11. 特徵統計
        # ============================================
        logger.info("計算特徵統計...")
        feature_stats = compute_feature_statistics(
            df_dev_final,
            numeric_columns=scaled_features,
            category_columns=config.feature_definition.ordinal_features
        )
        
        stats_path = project_root / "datamart" / "gold" / "reports" / f"feature_stats_{run_id}.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(feature_stats, f, ensure_ascii=False, indent=2)
        output_paths["feature_stats"] = stats_path
        
        # ============================================
        # 12. 儲存 Audit Log
        # ============================================
        audit_logger.add_record(audit_record)
        
        return output_paths
        
    except Exception as e:
        audit_record.errors.append(str(e))
        audit_logger.add_record(audit_record)
        raise
        
    finally:
        if should_stop_spark:
            spark.stop()


# 向後相容：保留原有的簡化版本
def run_gold_pipeline_simple(project_root: Path, spark: SparkSession = None) -> Dict[str, Path]:
    """簡化版本的 Gold Pipeline（向後相容）"""
    return run_gold_pipeline(
        project_root, 
        spark=spark, 
        save_artifacts=False, 
        run_drift_check=False
    )


if __name__ == "__main__":
    # 取得專案根目錄
    project_root = Path(__file__).parent.parent
    
    logger.info("=" * 60)
    logger.info("開始執行 Gold Layer Pipeline")
    logger.info("=" * 60)
    
    output_paths = run_gold_pipeline(project_root)
    
    logger.info("=" * 60)
    logger.info("🎉 Gold Layer 完成！")
    logger.info("=" * 60)
    logger.info("輸出檔案：")
    for name, path in output_paths.items():
        logger.info(f"  {name}: {path}")
