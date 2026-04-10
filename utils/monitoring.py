"""
Data Monitoring Module
======================
資料品質監控、漂移偵測、Production Monitoring

提供：
- PSI / CSI 計算
- Category / Numeric drift 檢測
- Label drift 檢測
- Production monitoring (metric + time triggers)
- Audit logging

Version: 2.0.0
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import ConfigManager, default_config

logger = logging.getLogger(__name__)


# ============================================
# PSI Calculation
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
    """
    all_keys = set(expected.keys()) | set(actual.keys())
    psi = 0.0
    
    for key in all_keys:
        e = max(expected.get(key, epsilon), epsilon)
        a = max(actual.get(key, epsilon), epsilon)
        psi += (a - e) * math.log(a / e)
    
    return psi


def calculate_numeric_psi(
    expected_values: List[float],
    actual_values: List[float],
    n_bins: int = 10,
    epsilon: float = 1e-10
) -> Tuple[float, Dict]:
    """計算數值型特徵的 PSI"""
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(expected_values, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    expected_counts, _ = np.histogram(expected_values, bins=bins)
    actual_counts, _ = np.histogram(actual_values, bins=bins)
    
    expected_ratios = expected_counts / len(expected_values)
    actual_ratios = actual_counts / len(actual_values)
    
    psi = 0.0
    bin_details = []
    
    for i in range(n_bins):
        e = max(expected_ratios[i], epsilon)
        a = max(actual_ratios[i], epsilon)
        bin_psi = (a - e) * math.log(a / e)
        psi += bin_psi
        bin_details.append({
            "bin_index": i,
            "expected_ratio": float(expected_ratios[i]),
            "actual_ratio": float(actual_ratios[i]),
            "bin_psi": float(bin_psi)
        })
    
    return psi, {"bins": bin_details, "total_psi": psi}


# ============================================
# Drift Detection
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
            "psi_value": float(self.psi_value) if self.psi_value is not None else None,
            "is_significant": bool(self.is_significant),
            "is_critical": bool(self.is_critical),
            "details": self._convert_numpy_types(self.details)
        }
    
    def _convert_numpy_types(self, obj):
        """遞迴轉換 numpy 類型"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


@dataclass
class DriftReport:
    """漂移報告"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dev_period: str = ""
    oot_period: str = ""
    dev_row_count: int = 0
    oot_row_count: int = 0
    results: List[DriftResult] = field(default_factory=list)
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
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Drift report saved: {path}")
    
    def log_summary(self):
        logger.info("=" * 60)
        logger.info("Drift Report Summary")
        logger.info(f"  Features Checked: {self.total_features_checked}")
        logger.info(f"  Significant (PSI >= 0.1): {self.significant_drift_count}")
        logger.info(f"  Critical (PSI >= 0.25): {self.critical_drift_count}")
        if self.critical_drift_count > 0:
            logger.warning("Critical drift detected:")
            for r in self.results:
                if r.is_critical:
                    logger.warning(f"  - {r.column}: PSI = {r.psi_value:.4f}")


class DriftMonitor:
    """資料漂移監控器"""
    
    def __init__(self, config: ConfigManager = None):
        self.config = config or default_config
    
    def compute_category_distribution(self, df: DataFrame, column: str) -> Dict[str, float]:
        total = df.count()
        if total == 0:
            return {}
        dist = df.groupBy(column).count().collect()
        return {str(row[column]): row["count"] / total for row in dist}
    
    def check_category_drift(self, df_dev: DataFrame, df_oot: DataFrame, column: str) -> DriftResult:
        dev_dist = self.compute_category_distribution(df_dev, column)
        oot_dist = self.compute_category_distribution(df_oot, column)
        psi = calculate_psi(dev_dist, oot_dist)
        
        # 計算未見類別
        dev_categories = set(dev_dist.keys())
        oot_categories = set(oot_dist.keys())
        unseen = oot_categories - dev_categories
        
        return DriftResult(
            column=column,
            drift_type="category",
            psi_value=psi,
            is_significant=psi >= 0.1,
            is_critical=psi >= 0.25,
            details={
                "dev_distribution": dev_dist,
                "oot_distribution": oot_dist,
                "unseen_categories": list(unseen),
                "unseen_ratio": sum(oot_dist.get(c, 0) for c in unseen),
            }
        )
    
    def check_numeric_drift(self, df_dev: DataFrame, df_oot: DataFrame, column: str) -> DriftResult:
        dev_values = [r[column] for r in df_dev.select(column).collect() if r[column] is not None]
        oot_values = [r[column] for r in df_oot.select(column).collect() if r[column] is not None]
        
        if not dev_values or not oot_values:
            return DriftResult(column=column, drift_type="numeric", psi_value=0.0,
                             is_significant=False, is_critical=False, details={"error": "Empty"})
        
        psi, details = calculate_numeric_psi(dev_values, oot_values)
        details["dev_stats"] = {"mean": float(np.mean(dev_values)), "std": float(np.std(dev_values))}
        details["oot_stats"] = {"mean": float(np.mean(oot_values)), "std": float(np.std(oot_values))}
        
        return DriftResult(
            column=column, drift_type="numeric", psi_value=psi,
            is_significant=psi >= 0.1, is_critical=psi >= 0.25, details=details
        )
    
    def check_label_drift(self, df_dev: DataFrame, df_oot: DataFrame, label_column: str = "授信結果_二元") -> DriftResult:
        dev_dist = self.compute_category_distribution(df_dev, label_column)
        oot_dist = self.compute_category_distribution(df_oot, label_column)
        psi = calculate_psi(dev_dist, oot_dist)
        
        return DriftResult(
            column=label_column, drift_type="label", psi_value=psi,
            is_significant=psi >= 0.1, is_critical=psi >= 0.25,
            details={"dev_distribution": dev_dist, "oot_distribution": oot_dist}
        )
    
    def generate_drift_report(
        self, df_dev: DataFrame, df_oot: DataFrame,
        category_columns: List[str] = None, numeric_columns: List[str] = None, check_label: bool = True
    ) -> DriftReport:
        report = DriftReport()
        report.dev_row_count = df_dev.count()
        report.oot_row_count = df_oot.count()
        
        if category_columns is None:
            category_columns = self.config.feature_definition.high_cardinality_features
        if numeric_columns is None:
            numeric_columns = self.config.feature_definition.numeric_features_to_scale
        
        for col in category_columns:
            if col in df_dev.columns and col in df_oot.columns:
                report.add_result(self.check_category_drift(df_dev, df_oot, col))
        
        for col in numeric_columns:
            if col in df_dev.columns and col in df_oot.columns:
                report.add_result(self.check_numeric_drift(df_dev, df_oot, col))
        
        if check_label and "授信結果_二元" in df_dev.columns:
            report.add_result(self.check_label_drift(df_dev, df_oot))
        
        report.log_summary()
        return report


# ============================================
# Unseen Category Check (保留向後相容)
# ============================================
def check_unseen_categories(df_dev: DataFrame, df_oot: DataFrame, columns: List[str]) -> Dict[str, Dict]:
    """檢查 OOT 中的未見類別"""
    results = {}
    for col in columns:
        if col not in df_dev.columns or col not in df_oot.columns:
            continue
        dev_values = {r[col] for r in df_dev.select(col).distinct().collect() if r[col] is not None}
        oot_values = {r[col] for r in df_oot.select(col).distinct().collect() if r[col] is not None}
        unseen = oot_values - dev_values
        oot_total = df_oot.count()
        unseen_count = df_oot.filter(F.col(col).isin(list(unseen))).count() if unseen else 0
        
        results[col] = {
            "dev_category_count": len(dev_values),
            "oot_category_count": len(oot_values),
            "unseen_categories": list(unseen),
            "unseen_count": unseen_count,
            "unseen_ratio": unseen_count / oot_total if oot_total > 0 else 0,
        }
    return results


# ============================================
# Audit Logger
# ============================================
@dataclass
class AuditRecord:
    """審計記錄"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    run_id: str = ""
    stage: str = ""
    action: str = ""
    input_path: str = ""
    output_path: str = ""
    row_count_before: int = 0
    row_count_after: int = 0
    rows_dropped: int = 0
    drop_reasons: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    config_version: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "stage": self.stage,
            "action": self.action,
            "row_counts": {"before": self.row_count_before, "after": self.row_count_after, "dropped": self.rows_dropped},
            "drop_reasons": self.drop_reasons,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class AuditLogger:
    """審計日誌管理器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[AuditRecord] = []
    
    def add_record(self, record: AuditRecord):
        self.records.append(record)
        log_file = self.output_dir / f"audit_{record.run_id}_{record.stage}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Audit record saved: {log_file}")
    
    def save_all(self, run_id: str):
        output_file = self.output_dir / f"audit_log_{run_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"run_id": run_id, "records": [r.to_dict() for r in self.records]}, f, ensure_ascii=False, indent=2)


# ============================================
# Feature Statistics
# ============================================
def compute_feature_statistics(
    df: DataFrame,
    numeric_columns: List[str] = None,
    category_columns: List[str] = None
) -> Dict[str, Dict]:
    """計算特徵統計摘要"""
    stats = {}
    
    if numeric_columns:
        for col in numeric_columns:
            if col not in df.columns:
                continue
            quoted_col = f"`{col}`"
            col_stats = df.select(
                F.count(F.col(col)).alias("count"),
                F.mean(F.col(col)).alias("mean"),
                F.stddev(F.col(col)).alias("std"),
                F.min(F.col(col)).alias("min"),
                F.max(F.col(col)).alias("max"),
            ).collect()[0]
            stats[col] = {
                "type": "numeric",
                "count": int(col_stats["count"]),
                "mean": float(col_stats["mean"]) if col_stats["mean"] else None,
                "std": float(col_stats["std"]) if col_stats["std"] else None,
            }
    
    if category_columns:
        total_rows = df.count()
        for col in category_columns:
            if col not in df.columns:
                continue
            dist = df.groupBy(col).count().collect()
            value_counts = {str(row[col]): row["count"] for row in dist}
            stats[col] = {
                "type": "category",
                "unique_count": len(value_counts),
                "top_values": dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            }
    
    return stats


# ============================================
# Production Monitoring
# ============================================
@dataclass
class ProductionMonitorConfig:
    """Production Monitoring 設定"""
    min_auc: float = 0.75
    min_f1_reject: float = 0.20
    max_psi: float = 0.25
    retrain_interval_months: int = 6
    warning_auc: float = 0.80
    warning_psi: float = 0.10


@dataclass
class MonitoringResult:
    """Production Monitoring 結果"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    period_start: str = ""
    period_end: str = ""
    total_predictions: int = 0
    total_with_labels: int = 0
    
    # Metrics
    auc: Optional[float] = None
    f1_reject: Optional[float] = None
    brier_score: Optional[float] = None
    
    # Score distribution
    score_mean: float = 0.0
    score_std: float = 0.0
    
    # Zone distribution
    zone_high_ratio: float = 0.0
    zone_manual_ratio: float = 0.0
    zone_low_ratio: float = 0.0
    
    # Drift
    score_psi: Optional[float] = None
    
    # Alerts
    alerts: List[str] = field(default_factory=list)
    needs_retraining: bool = False
    retraining_reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "period": f"{self.period_start} ~ {self.period_end}",
            "total_predictions": self.total_predictions,
            "metrics": {"auc": self.auc, "f1_reject": self.f1_reject, "brier_score": self.brier_score},
            "score_distribution": {"mean": self.score_mean, "std": self.score_std},
            "zone_distribution": {"high": self.zone_high_ratio, "manual": self.zone_manual_ratio, "low": self.zone_low_ratio},
            "score_psi": self.score_psi,
            "alerts": self.alerts,
            "needs_retraining": self.needs_retraining,
            "retraining_reason": self.retraining_reason,
        }


class ProductionMonitor:
    """Production Monitoring"""
    
    def __init__(self, config: ProductionMonitorConfig = None, baseline_scores: np.ndarray = None):
        self.config = config or ProductionMonitorConfig()
        self.baseline_scores = baseline_scores
        self.history: List[MonitoringResult] = []
    
    def set_baseline(self, scores: np.ndarray):
        self.baseline_scores = scores
        logger.info(f"Baseline set: {len(scores)} samples")
    
    def monitor(
        self,
        predictions: np.ndarray,
        labels: np.ndarray = None,
        period_start: str = "",
        period_end: str = "",
        lower_threshold: float = 0.4,
        upper_threshold: float = 0.7
    ) -> MonitoringResult:
        result = MonitoringResult(
            period_start=period_start, period_end=period_end, total_predictions=len(predictions)
        )
        
        # Score distribution
        result.score_mean = float(np.mean(predictions))
        result.score_std = float(np.std(predictions))
        
        # Zone distribution
        result.zone_high_ratio = float(np.mean(predictions >= upper_threshold))
        result.zone_manual_ratio = float(np.mean((predictions >= lower_threshold) & (predictions < upper_threshold)))
        result.zone_low_ratio = float(np.mean(predictions < lower_threshold))
        
        # Score PSI
        if self.baseline_scores is not None:
            psi_value, _ = calculate_numeric_psi(self.baseline_scores.tolist(), predictions.tolist())
            result.score_psi = psi_value
            if psi_value >= self.config.max_psi:
                result.alerts.append(f"CRITICAL: Score PSI={psi_value:.4f} >= {self.config.max_psi}")
        
        # Metrics (if labels available)
        if labels is not None:
            valid_mask = ~np.isnan(labels)
            if np.sum(valid_mask) > 0:
                y_true = labels[valid_mask].astype(int)
                y_pred_proba = predictions[valid_mask]
                y_pred = (y_pred_proba >= 0.5).astype(int)
                result.total_with_labels = len(y_true)
                
                try:
                    from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
                    result.auc = roc_auc_score(y_true, y_pred_proba)
                    result.f1_reject = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
                    result.brier_score = brier_score_loss(y_true, y_pred_proba)
                    
                    if result.auc < self.config.min_auc:
                        result.alerts.append(f"CRITICAL: AUC={result.auc:.4f} < {self.config.min_auc}")
                    if result.f1_reject < self.config.min_f1_reject:
                        result.alerts.append(f"CRITICAL: F1_reject={result.f1_reject:.4f} < {self.config.min_f1_reject}")
                except Exception as e:
                    logger.warning(f"Metrics calculation error: {e}")
        
        # Check retrain trigger
        result.needs_retraining, result.retraining_reason = self._check_retrain_trigger(result)
        self.history.append(result)
        return result
    
    def _check_retrain_trigger(self, result: MonitoringResult) -> Tuple[bool, str]:
        reasons = []
        if result.auc is not None and result.auc < self.config.min_auc:
            reasons.append(f"AUC={result.auc:.4f} < {self.config.min_auc}")
        if result.f1_reject is not None and result.f1_reject < self.config.min_f1_reject:
            reasons.append(f"F1_reject={result.f1_reject:.4f} < {self.config.min_f1_reject}")
        if result.score_psi is not None and result.score_psi >= self.config.max_psi:
            reasons.append(f"PSI={result.score_psi:.4f} >= {self.config.max_psi}")
        return (True, "; ".join(reasons)) if reasons else (False, "")
    
    def check_time_trigger(self, last_training_date: str) -> Tuple[bool, str]:
        last_date = pd.to_datetime(last_training_date).date()
        current_date = datetime.now().date()
        months_since = (current_date.year - last_date.year) * 12 + (current_date.month - last_date.month)
        if months_since >= self.config.retrain_interval_months:
            return True, f"{months_since} months since last training"
        return False, ""
    
    def save_history(self, output_path: Path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"history": [r.to_dict() for r in self.history]}, f, ensure_ascii=False, indent=2)


def check_retrain_trigger(
    current_metrics: Dict[str, float],
    last_training_date: str = None,
    config: ProductionMonitorConfig = None
) -> Tuple[bool, str]:
    """便利函式：檢查是否需要 retraining"""
    config = config or ProductionMonitorConfig()
    reasons = []
    
    auc = current_metrics.get('auc')
    if auc is not None and auc < config.min_auc:
        reasons.append(f"AUC={auc:.4f} < {config.min_auc}")
    
    f1_reject = current_metrics.get('f1_reject')
    if f1_reject is not None and f1_reject < config.min_f1_reject:
        reasons.append(f"F1_reject={f1_reject:.4f} < {config.min_f1_reject}")
    
    score_psi = current_metrics.get('score_psi')
    if score_psi is not None and score_psi >= config.max_psi:
        reasons.append(f"PSI={score_psi:.4f} >= {config.max_psi}")
    
    if last_training_date:
        last_date = pd.to_datetime(last_training_date).date()
        current_date = datetime.now().date()
        months_since = (current_date.year - last_date.year) * 12 + (current_date.month - last_date.month)
        if months_since >= config.retrain_interval_months:
            reasons.append(f"Time: {months_since} months")
    
    return (True, "; ".join(reasons)) if reasons else (False, "")
