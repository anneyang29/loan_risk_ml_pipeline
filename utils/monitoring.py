"""
Data Monitoring Module
======================
資料品質監控、類別漂移偵測、PSI/CSI 計算

提供：
- Unseen category rate monitoring
- Category drift report (PSI/CSI)
- Distribution shift logging
- Feature statistics summary
- Audit logging
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import ConfigManager, default_config

logger = logging.getLogger(__name__)


# ============================================
# PSI / CSI Calculation
# ============================================
def calculate_psi(
    expected: Dict[str, float],
    actual: Dict[str, float],
    epsilon: float = 1e-10
) -> float:
    """
    計算 Population Stability Index (PSI)
    
    PSI < 0.1: 無顯著變化
    0.1 <= PSI < 0.25: 需要監控
    PSI >= 0.25: 需要調查
    
    Args:
        expected: 預期分布 (value -> ratio)
        actual: 實際分布 (value -> ratio)
        epsilon: 避免除零
    
    Returns:
        PSI 值
    """
    all_keys = set(expected.keys()) | set(actual.keys())
    psi = 0.0
    
    for key in all_keys:
        e = expected.get(key, epsilon)
        a = actual.get(key, epsilon)
        
        # 避免零值
        e = max(e, epsilon)
        a = max(a, epsilon)
        
        psi += (a - e) * math.log(a / e)
    
    return psi


def calculate_numeric_psi(
    expected_values: List[float],
    actual_values: List[float],
    n_bins: int = 10,
    epsilon: float = 1e-10
) -> Tuple[float, Dict]:
    """
    計算數值型特徵的 PSI
    
    Args:
        expected_values: 預期值列表
        actual_values: 實際值列表
        n_bins: 分箱數量
        epsilon: 避免除零
    
    Returns:
        (PSI 值, 詳細資訊)
    """
    import numpy as np
    
    # 計算分位數邊界（基於 expected）
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(expected_values, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # 計算各 bin 的比例
    expected_counts, _ = np.histogram(expected_values, bins=bins)
    actual_counts, _ = np.histogram(actual_values, bins=bins)
    
    expected_ratios = expected_counts / len(expected_values)
    actual_ratios = actual_counts / len(actual_values)
    
    # 計算 PSI
    psi = 0.0
    bin_details = []
    
    for i in range(n_bins):
        e = max(expected_ratios[i], epsilon)
        a = max(actual_ratios[i], epsilon)
        bin_psi = (a - e) * math.log(a / e)
        psi += bin_psi
        
        bin_details.append({
            "bin_index": i,
            "bin_range": f"[{bins[i]:.4f}, {bins[i+1]:.4f})",
            "expected_ratio": float(expected_ratios[i]),
            "actual_ratio": float(actual_ratios[i]),
            "bin_psi": float(bin_psi)
        })
    
    return psi, {"bins": bin_details, "total_psi": psi}


# ============================================
# Data Drift Monitor
# ============================================
@dataclass
class DriftResult:
    """漂移檢測結果"""
    column: str
    drift_type: str  # "category", "numeric", "label"
    psi_value: float
    is_significant: bool  # PSI >= 0.1
    is_critical: bool     # PSI >= 0.25
    details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "column": self.column,
            "drift_type": self.drift_type,
            "psi_value": self.psi_value,
            "is_significant": self.is_significant,
            "is_critical": self.is_critical,
            "details": self.details
        }


@dataclass
class DriftReport:
    """漂移報告"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dev_period: str = ""
    oot_period: str = ""
    dev_row_count: int = 0
    oot_row_count: int = 0
    
    results: List[DriftResult] = field(default_factory=list)
    
    # 摘要
    total_features_checked: int = 0
    significant_drift_count: int = 0
    critical_drift_count: int = 0
    
    def add_result(self, result: DriftResult):
        self.results.append(result)
        self.total_features_checked += 1
        if result.is_significant:
            self.significant_drift_count += 1
        if result.is_critical:
            self.critical_drift_count += 1
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "dev_period": self.dev_period,
            "oot_period": self.oot_period,
            "dev_row_count": self.dev_row_count,
            "oot_row_count": self.oot_row_count,
            "summary": {
                "total_features_checked": self.total_features_checked,
                "significant_drift_count": self.significant_drift_count,
                "critical_drift_count": self.critical_drift_count,
            },
            "results": [r.to_dict() for r in self.results]
        }
    
    def save(self, path: Path):
        """儲存報告"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Drift report saved to: {path}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """轉換為 DataFrame"""
        records = []
        for r in self.results:
            records.append({
                "column": r.column,
                "drift_type": r.drift_type,
                "psi_value": r.psi_value,
                "is_significant": r.is_significant,
                "is_critical": r.is_critical,
            })
        return pd.DataFrame(records)
    
    def log_summary(self):
        """輸出摘要"""
        logger.info("=" * 60)
        logger.info("Drift Report Summary")
        logger.info("=" * 60)
        logger.info(f"Dev Period: {self.dev_period}")
        logger.info(f"OOT Period: {self.oot_period}")
        logger.info(f"Dev Rows: {self.dev_row_count:,}")
        logger.info(f"OOT Rows: {self.oot_row_count:,}")
        logger.info(f"Features Checked: {self.total_features_checked}")
        logger.info(f"Significant Drift (PSI >= 0.1): {self.significant_drift_count}")
        logger.info(f"Critical Drift (PSI >= 0.25): {self.critical_drift_count}")
        
        if self.critical_drift_count > 0:
            logger.warning("CRITICAL DRIFT detected in:")
            for r in self.results:
                if r.is_critical:
                    logger.warning(f"  - {r.column}: PSI = {r.psi_value:.4f}")


class DriftMonitor:
    """資料漂移監控器"""
    
    def __init__(self, config: ConfigManager = None):
        self.config = config or default_config
    
    def compute_category_distribution(
        self, 
        df: DataFrame, 
        column: str
    ) -> Dict[str, float]:
        """計算類別分布"""
        total = df.count()
        if total == 0:
            return {}
        
        dist = df.groupBy(column).count().collect()
        return {
            str(row[column]): row["count"] / total
            for row in dist
        }
    
    def check_category_drift(
        self,
        df_dev: DataFrame,
        df_oot: DataFrame,
        column: str
    ) -> DriftResult:
        """檢查類別特徵漂移"""
        dev_dist = self.compute_category_distribution(df_dev, column)
        oot_dist = self.compute_category_distribution(df_oot, column)
        
        psi = calculate_psi(dev_dist, oot_dist)
        
        # 計算未見類別
        dev_categories = set(dev_dist.keys())
        oot_categories = set(oot_dist.keys())
        unseen_categories = oot_categories - dev_categories
        unseen_ratio = sum(oot_dist.get(c, 0) for c in unseen_categories)
        
        return DriftResult(
            column=column,
            drift_type="category",
            psi_value=psi,
            is_significant=psi >= 0.1,
            is_critical=psi >= 0.25,
            details={
                "dev_distribution": dev_dist,
                "oot_distribution": oot_dist,
                "dev_category_count": len(dev_categories),
                "oot_category_count": len(oot_categories),
                "unseen_categories": list(unseen_categories),
                "unseen_ratio": unseen_ratio,
            }
        )
    
    def check_numeric_drift(
        self,
        df_dev: DataFrame,
        df_oot: DataFrame,
        column: str,
        n_bins: int = 10
    ) -> DriftResult:
        """檢查數值特徵漂移"""
        # 取得值列表
        dev_values = [
            row[column] for row in df_dev.select(column).collect()
            if row[column] is not None
        ]
        oot_values = [
            row[column] for row in df_oot.select(column).collect()
            if row[column] is not None
        ]
        
        if not dev_values or not oot_values:
            return DriftResult(
                column=column,
                drift_type="numeric",
                psi_value=0.0,
                is_significant=False,
                is_critical=False,
                details={"error": "Empty values"}
            )
        
        psi, details = calculate_numeric_psi(dev_values, oot_values, n_bins)
        
        # 基本統計
        import numpy as np
        details["dev_stats"] = {
            "mean": float(np.mean(dev_values)),
            "std": float(np.std(dev_values)),
            "min": float(np.min(dev_values)),
            "max": float(np.max(dev_values)),
        }
        details["oot_stats"] = {
            "mean": float(np.mean(oot_values)),
            "std": float(np.std(oot_values)),
            "min": float(np.min(oot_values)),
            "max": float(np.max(oot_values)),
        }
        
        return DriftResult(
            column=column,
            drift_type="numeric",
            psi_value=psi,
            is_significant=psi >= 0.1,
            is_critical=psi >= 0.25,
            details=details
        )
    
    def check_label_drift(
        self,
        df_dev: DataFrame,
        df_oot: DataFrame,
        label_column: str = "授信結果_二元"
    ) -> DriftResult:
        """檢查 Label 分布漂移"""
        dev_dist = self.compute_category_distribution(df_dev, label_column)
        oot_dist = self.compute_category_distribution(df_oot, label_column)
        
        psi = calculate_psi(dev_dist, oot_dist)
        
        return DriftResult(
            column=label_column,
            drift_type="label",
            psi_value=psi,
            is_significant=psi >= 0.1,
            is_critical=psi >= 0.25,
            details={
                "dev_distribution": dev_dist,
                "oot_distribution": oot_dist,
                "dev_positive_ratio": dev_dist.get("1", dev_dist.get(1, 0)),
                "oot_positive_ratio": oot_dist.get("1", oot_dist.get(1, 0)),
            }
        )
    
    def generate_drift_report(
        self,
        df_dev: DataFrame,
        df_oot: DataFrame,
        category_columns: List[str] = None,
        numeric_columns: List[str] = None,
        check_label: bool = True
    ) -> DriftReport:
        """
        生成完整漂移報告
        """
        report = DriftReport()
        report.dev_row_count = df_dev.count()
        report.oot_row_count = df_oot.count()
        
        # 計算時間區間
        if "進件日" in df_dev.columns:
            dev_dates = df_dev.agg(
                F.min("進件日").alias("min"),
                F.max("進件日").alias("max")
            ).collect()[0]
            report.dev_period = f"{dev_dates['min']} ~ {dev_dates['max']}"
        
        if "進件日" in df_oot.columns:
            oot_dates = df_oot.agg(
                F.min("進件日").alias("min"),
                F.max("進件日").alias("max")
            ).collect()[0]
            report.oot_period = f"{oot_dates['min']} ~ {oot_dates['max']}"
        
        # 使用預設欄位
        if category_columns is None:
            category_columns = self.config.feature_definition.high_cardinality_features
        
        if numeric_columns is None:
            numeric_columns = self.config.feature_definition.numeric_features_to_scale
        
        # 1. 類別特徵漂移
        for col in category_columns:
            if col in df_dev.columns and col in df_oot.columns:
                logger.info(f"Checking category drift: {col}")
                result = self.check_category_drift(df_dev, df_oot, col)
                report.add_result(result)
        
        # 2. 數值特徵漂移
        for col in numeric_columns:
            if col in df_dev.columns and col in df_oot.columns:
                logger.info(f"Checking numeric drift: {col}")
                result = self.check_numeric_drift(df_dev, df_oot, col)
                report.add_result(result)
        
        # 3. Label 漂移
        if check_label and "授信結果_二元" in df_dev.columns:
            logger.info("Checking label drift")
            result = self.check_label_drift(df_dev, df_oot)
            report.add_result(result)
        
        report.log_summary()
        return report


# ============================================
# Unseen Category Monitor
# ============================================
@dataclass
class UnseenCategoryReport:
    """未見類別報告"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    results: Dict[str, Dict] = field(default_factory=dict)
    
    def add_column_result(
        self,
        column: str,
        dev_categories: set,
        oot_categories: set,
        unseen_count: int,
        unseen_ratio: float
    ):
        self.results[column] = {
            "dev_category_count": len(dev_categories),
            "oot_category_count": len(oot_categories),
            "unseen_categories": list(oot_categories - dev_categories),
            "unseen_count": unseen_count,
            "unseen_ratio": unseen_ratio,
            "is_warning": unseen_ratio > 0.05,
        }
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "results": self.results
        }
    
    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    def log_summary(self):
        logger.info("=" * 60)
        logger.info("Unseen Category Report")
        logger.info("=" * 60)
        
        for col, info in self.results.items():
            status = "⚠️ WARNING" if info["is_warning"] else "✓ OK"
            logger.info(
                f"{col}: {status} - "
                f"unseen_ratio={info['unseen_ratio']:.2%}, "
                f"unseen_count={info['unseen_count']}"
            )
            
            if info["unseen_categories"]:
                logger.info(f"  Unseen values: {info['unseen_categories'][:10]}...")


def check_unseen_categories(
    df_dev: DataFrame,
    df_oot: DataFrame,
    columns: List[str]
) -> UnseenCategoryReport:
    """
    檢查 OOT 中的未見類別
    """
    report = UnseenCategoryReport()
    
    for col in columns:
        if col not in df_dev.columns or col not in df_oot.columns:
            continue
        
        # 取得 distinct values
        dev_values = {
            row[col] for row in df_dev.select(col).distinct().collect()
            if row[col] is not None
        }
        oot_values = {
            row[col] for row in df_oot.select(col).distinct().collect()
            if row[col] is not None
        }
        
        unseen = oot_values - dev_values
        
        # 計算 unseen 的筆數比例
        if unseen:
            unseen_count = df_oot.filter(F.col(col).isin(list(unseen))).count()
        else:
            unseen_count = 0
        
        oot_total = df_oot.count()
        unseen_ratio = unseen_count / oot_total if oot_total > 0 else 0
        
        report.add_column_result(col, dev_values, oot_values, unseen_count, unseen_ratio)
    
    report.log_summary()
    return report


# ============================================
# Audit Logger
# ============================================
@dataclass
class AuditRecord:
    """審計記錄"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    run_id: str = ""
    stage: str = ""  # bronze, silver, gold
    action: str = ""
    
    input_path: str = ""
    output_path: str = ""
    
    row_count_before: int = 0
    row_count_after: int = 0
    rows_dropped: int = 0
    drop_reasons: Dict[str, int] = field(default_factory=dict)
    
    null_summary: Dict[str, float] = field(default_factory=dict)
    feature_summary: Dict[str, Dict] = field(default_factory=dict)
    
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    config_version: str = ""
    code_version: str = "unknown"
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "stage": self.stage,
            "action": self.action,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "row_counts": {
                "before": self.row_count_before,
                "after": self.row_count_after,
                "dropped": self.rows_dropped,
            },
            "drop_reasons": self.drop_reasons,
            "null_summary": self.null_summary,
            "feature_summary": self.feature_summary,
            "warnings": self.warnings,
            "errors": self.errors,
            "config_version": self.config_version,
            "code_version": self.code_version,
        }


class AuditLogger:
    """審計日誌管理器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[AuditRecord] = []
    
    def add_record(self, record: AuditRecord):
        self.records.append(record)
        
        # 即時儲存
        log_file = self.output_dir / f"audit_{record.run_id}_{record.stage}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Audit record saved: {log_file}")
    
    def save_all(self, run_id: str):
        """儲存所有記錄"""
        all_records = [r.to_dict() for r in self.records]
        
        output_file = self.output_dir / f"audit_log_{run_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "run_id": run_id,
                "total_records": len(all_records),
                "records": all_records
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"All audit records saved: {output_file}")


# ============================================
# Feature Statistics
# ============================================
def compute_feature_statistics(
    df: DataFrame,
    numeric_columns: List[str] = None,
    category_columns: List[str] = None
) -> Dict[str, Dict]:
    """
    計算特徵統計摘要
    """
    stats = {}
    
    # 數值特徵
    if numeric_columns:
        for col in numeric_columns:
            if col not in df.columns:
                continue
            
            col_stats = df.select(
                F.count(col).alias("count"),
                F.sum(F.when(F.col(col).isNull(), 1).otherwise(0)).alias("null_count"),
                F.mean(col).alias("mean"),
                F.stddev(col).alias("std"),
                F.min(col).alias("min"),
                F.max(col).alias("max"),
                F.expr(f"percentile_approx({col}, 0.25)").alias("q25"),
                F.expr(f"percentile_approx({col}, 0.50)").alias("median"),
                F.expr(f"percentile_approx({col}, 0.75)").alias("q75"),
            ).collect()[0]
            
            total = col_stats["count"] + col_stats["null_count"]
            
            stats[col] = {
                "type": "numeric",
                "count": int(col_stats["count"]),
                "null_count": int(col_stats["null_count"]),
                "null_ratio": col_stats["null_count"] / total if total > 0 else 0,
                "mean": float(col_stats["mean"]) if col_stats["mean"] else None,
                "std": float(col_stats["std"]) if col_stats["std"] else None,
                "min": float(col_stats["min"]) if col_stats["min"] else None,
                "max": float(col_stats["max"]) if col_stats["max"] else None,
                "q25": float(col_stats["q25"]) if col_stats["q25"] else None,
                "median": float(col_stats["median"]) if col_stats["median"] else None,
                "q75": float(col_stats["q75"]) if col_stats["q75"] else None,
            }
    
    # 類別特徵
    if category_columns:
        total_rows = df.count()
        
        for col in category_columns:
            if col not in df.columns:
                continue
            
            dist = df.groupBy(col).count().collect()
            value_counts = {str(row[col]): row["count"] for row in dist}
            
            null_count = value_counts.pop("None", 0)
            
            stats[col] = {
                "type": "category",
                "unique_count": len(value_counts),
                "null_count": null_count,
                "null_ratio": null_count / total_rows if total_rows > 0 else 0,
                "top_values": dict(
                    sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                ),
            }
    
    return stats


def save_feature_statistics(
    stats: Dict[str, Dict],
    output_path: Path
):
    """儲存特徵統計"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Feature statistics saved to: {output_path}")
