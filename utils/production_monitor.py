"""
Production Monitoring Module v2
================================
完整的 Production Monitoring、Retraining Trigger、Production Batch Scoring 支援

功能：
1. Production Monitoring Pipeline
   - Score PSI（與 baseline 比較分布偏移）
   - Score mean / std
   - Zone distribution shift（High / Manual Review / Low）
   - AUC / F1_reject (若 label 已回流)

2. Retraining Trigger
   - Metric trigger (AUC/F1 低於門檻)
   - Time trigger (每 6 個月 review / retrain)

3. Production Batch Scoring
   - score_production_batch(): 只輸出預測，不算指標（因為沒有 label）
   - ⚠️ 這不是 Final Blind Holdout（Phase 4 有 label 可算指標）
   - ⚠️ Production Batch Scoring 沒有 ground truth，只輸出 probability + zone

4. Dynamic Data Window for Retraining
   - 以當前日期往前推 18 個月作為 development

Baseline 設計原則：
- baseline 預設來自 Phase 3 Policy Validation 的預測機率
- 路徑: model_bank/<latest_run>/policy_validation_predictions.csv
- 欄位: pred_prob
- 原因：Policy Validation 是部署前最後一次有 label 的評估階段，
  其 score 分布最接近上線後的真實推論分布

Version: 2.0.0
"""

import json
import logging
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================
# PSI Calculation
# ============================================
def calculate_numeric_psi(
    expected_values: List[float],
    actual_values: List[float],
    n_bins: int = 10,
    epsilon: float = 1e-10
) -> Tuple[float, Dict]:
    """計算數值型特徵的 PSI"""
    if len(expected_values) == 0 or len(actual_values) == 0:
        return 0.0, {}
    
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(expected_values, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # 去重複
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0, {}
    
    expected_counts, _ = np.histogram(expected_values, bins=bins)
    actual_counts, _ = np.histogram(actual_values, bins=bins)
    
    expected_ratios = expected_counts / len(expected_values)
    actual_ratios = actual_counts / len(actual_values)
    
    psi = 0.0
    bin_details = []
    
    for i in range(len(bins) - 1):
        e = max(expected_ratios[i], epsilon)
        a = max(actual_ratios[i], epsilon)
        bin_psi = (a - e) * np.log(a / e)
        psi += bin_psi
        bin_details.append({
            "bin_index": i,
            "expected_ratio": float(expected_ratios[i]),
            "actual_ratio": float(actual_ratios[i]),
            "bin_psi": float(bin_psi)
        })
    
    return float(psi), {"bins": bin_details, "total_psi": float(psi)}


# ============================================
# Production Monitoring Config
# ============================================
@dataclass
class ProductionMonitoringConfig:
    """
    Production Monitoring 設定
    
    Retraining Trigger:
    A. Metric Trigger: 指標低於門檻
    B. Time Trigger: 距離上次訓練超過指定月數
    """
    # Metric Triggers (指標門檻)
    min_auc: float = 0.85              # AUC 低於此值觸發 retraining（提高自 0.75）
    min_f1_reject: float = 0.30        # F1_reject 低於此值觸發 retraining（提高自 0.20）
    max_score_psi: float = 0.25        # Score PSI 超過此值觸發 retraining
    
    # Time Trigger (時間門檻)
    retrain_interval_months: int = 6   # 每 6 個月強制 review / retrain
    
    # Warning Thresholds (警示門檻，不立即觸發 retraining)
    warning_auc: float = 0.88          # 提高自 0.80
    warning_score_psi: float = 0.10
    warning_zone_shift: float = 0.10   # 降低自 0.15，更敏感偵測 zone 變化
    
    # Zone Thresholds
    default_lower_threshold: float = 0.5   # 提高自 0.4
    default_upper_threshold: float = 0.85  # 提高自 0.7
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================
# Production Monitoring Result
# ============================================
@dataclass
class ProductionMonitoringResult:
    """
    Production Monitoring 單次結果
    
    區分兩種情況：
    1. 有 Label 回流：可計算 AUC / F1 / Brier
    2. 無 Label：只能計算 Score Distribution / Zone Distribution / PSI
    """
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = ""
    monitoring_period_start: str = ""
    monitoring_period_end: str = ""
    
    # Counts
    total_predictions: int = 0
    total_with_labels: int = 0        # 已回流 label 數
    label_coverage_ratio: float = 0.0  # 有 label 的比例
    
    # Score Distribution
    score_mean: float = 0.0
    score_std: float = 0.0
    score_median: float = 0.0
    score_min: float = 0.0
    score_max: float = 0.0
    score_q25: float = 0.0
    score_q75: float = 0.0
    
    # Zone Distribution
    zone_high_count: int = 0
    zone_high_ratio: float = 0.0
    zone_manual_count: int = 0
    zone_manual_ratio: float = 0.0
    zone_low_count: int = 0
    zone_low_ratio: float = 0.0
    
    # Baseline Comparison (Zone Shift)
    baseline_zone_high_ratio: float = 0.0
    baseline_zone_manual_ratio: float = 0.0
    baseline_zone_low_ratio: float = 0.0
    zone_high_shift: float = 0.0      # current - baseline
    zone_manual_shift: float = 0.0
    zone_low_shift: float = 0.0
    
    # Score PSI (vs baseline)
    score_psi: Optional[float] = None
    
    # Metrics (若有 label)
    auc: Optional[float] = None
    f1_reject: Optional[float] = None
    brier_score: Optional[float] = None
    ks: Optional[float] = None
    positive_ratio: Optional[float] = None  # 實際正樣本比例
    
    # Alerts & Triggers
    alerts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    needs_retraining: bool = False
    retraining_trigger_reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "period": {
                "start": self.monitoring_period_start,
                "end": self.monitoring_period_end,
            },
            "counts": {
                "total_predictions": self.total_predictions,
                "total_with_labels": self.total_with_labels,
                "label_coverage_ratio": self.label_coverage_ratio,
            },
            "score_distribution": {
                "mean": self.score_mean,
                "std": self.score_std,
                "median": self.score_median,
                "min": self.score_min,
                "max": self.score_max,
                "q25": self.score_q25,
                "q75": self.score_q75,
            },
            "zone_distribution": {
                "high": {"count": self.zone_high_count, "ratio": self.zone_high_ratio},
                "manual": {"count": self.zone_manual_count, "ratio": self.zone_manual_ratio},
                "low": {"count": self.zone_low_count, "ratio": self.zone_low_ratio},
            },
            "zone_shift": {
                "high_shift": self.zone_high_shift,
                "manual_shift": self.zone_manual_shift,
                "low_shift": self.zone_low_shift,
            },
            "score_psi": self.score_psi,
            "metrics": {
                "auc": self.auc,
                "f1_reject": self.f1_reject,
                "brier_score": self.brier_score,
                "ks": self.ks,
                "positive_ratio": self.positive_ratio,
            },
            "alerts": self.alerts,
            "warnings": self.warnings,
            "retraining": {
                "needs_retraining": self.needs_retraining,
                "trigger_reason": self.retraining_trigger_reason,
            }
        }


# ============================================
# Production Batch Scoring Result
# ============================================
@dataclass
class ProductionBatchResult:
    """
    Production Batch Scoring 結果
    
    注意：Production Batch Scoring 沒有真實 label，
    不能算 AUC / F1，只輸出 predictions + zone assignment
    """
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = ""
    batch_id: str = ""
    
    total_records: int = 0
    
    # Zone counts
    zone_high_count: int = 0
    zone_manual_count: int = 0
    zone_low_count: int = 0
    
    # Score distribution
    score_mean: float = 0.0
    score_std: float = 0.0
    
    # Predictions (通常不直接存在 dataclass 裡，而是存成 CSV)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "batch_id": self.batch_id,
            "total_records": self.total_records,
            "zone_distribution": {
                "high": self.zone_high_count,
                "manual": self.zone_manual_count,
                "low": self.zone_low_count,
            },
            "score_distribution": {
                "mean": self.score_mean,
                "std": self.score_std,
            }
        }


# ============================================
# Production Scoring
# ============================================
def score_production_batch(
    model,
    X: np.ndarray,
    case_ids: List[str] = None,
    model_version: str = "",
    lower_threshold: float = 0.5,
    upper_threshold: float = 0.85
) -> Tuple[pd.DataFrame, ProductionBatchResult]:
    """
    Production Batch Scoring
    
    注意：
    - 這是 Production Batch Scoring（上線後的批次推論）
    - 沒有真實 label（或 label 延遲回流）
    - 只輸出 prediction probability、zone、timestamp、model_version
    - 不算 AUC / F1（因為沒有 ground truth）
    
    ⚠️ 與 Final Blind Holdout (Phase 4) 的區別：
    - Phase 4 有真實 label，可算 AUC / F1 / KS
    - Production Batch Scoring 沒有 label，僅做推論
    
    Args:
        model: 訓練好的模型
        X: 特徵矩陣
        case_ids: 案件編號列表
        model_version: 模型版本
        lower_threshold: 三區間低門檻
        upper_threshold: 三區間高門檻
        
    Returns:
        predictions_df: 預測結果 DataFrame
        batch_result: 批次統計摘要
    """
    timestamp = datetime.now().isoformat()
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Inference
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Zone assignment
    zones = np.ones(len(y_pred_proba), dtype=int)  # 預設人工審核
    zones[y_pred_proba >= upper_threshold] = 2     # 高通過機率區
    zones[y_pred_proba < lower_threshold] = 0      # 低通過機率區
    
    zone_names = {2: '高通過機率區', 1: '人工審核區', 0: '低通過機率區'}
    
    # 建立結果 DataFrame
    predictions_df = pd.DataFrame({
        '案件編號': case_ids if case_ids else [f"CASE_{i}" for i in range(len(X))],
        'pred_prob': y_pred_proba,
        'pred_zone': zones,
        'zone_name': [zone_names[z] for z in zones],
        'model_version': model_version,
        'scoring_timestamp': timestamp,
    })
    
    # 統計摘要
    batch_result = ProductionBatchResult(
        timestamp=timestamp,
        model_version=model_version,
        batch_id=batch_id,
        total_records=len(X),
        zone_high_count=int(np.sum(zones == 2)),
        zone_manual_count=int(np.sum(zones == 1)),
        zone_low_count=int(np.sum(zones == 0)),
        score_mean=float(np.mean(y_pred_proba)),
        score_std=float(np.std(y_pred_proba)),
    )
    
    logger.info(f"Production Batch Scoring 完成（無 label，僅輸出 predictions）")
    logger.info(f"  Total: {batch_result.total_records} records")
    logger.info(f"  Zone Distribution: High={batch_result.zone_high_count}, "
                f"Manual={batch_result.zone_manual_count}, Low={batch_result.zone_low_count}")
    logger.info(f"  Score: mean={batch_result.score_mean:.4f}, std={batch_result.score_std:.4f}")
    
    return predictions_df, batch_result


# ============================================
# Production Monitoring Pipeline
# ============================================
class ProductionMonitor:
    """
    Production Monitoring Pipeline
    
    功能：
    1. 監控 score 分布變化 (PSI)
    2. 監控 zone 分布變化
    3. 若 label 回流，計算 AUC / F1_reject
    4. 檢查 retraining trigger
    
    Baseline 來源：
    - baseline_scores 應來自 Phase 3 (Policy Validation) 的預測機率
    - 路徑：model_bank/<latest_run>/policy_validation_predictions.csv
    - 欄位：pred_prob
    - 原因：Policy Validation 是模型部署前最後一次有 label 的評估階段，
      其 score 分布最接近上線時的真實推論分布
    - 在 main.py 中，run_monitoring_pipeline() 會讀取該檔作為 baseline
    """
    
    def __init__(
        self,
        config: ProductionMonitoringConfig = None,
        baseline_scores: np.ndarray = None,
        baseline_zone_distribution: Dict[str, float] = None
    ):
        self.config = config or ProductionMonitoringConfig()
        self.baseline_scores = baseline_scores
        self.baseline_zone_distribution = baseline_zone_distribution or {}
        self.monitoring_history: List[ProductionMonitoringResult] = []
        self.last_training_date: Optional[str] = None
    
    def set_baseline(
        self,
        scores: np.ndarray,
        lower_threshold: float = None,
        upper_threshold: float = None
    ):
        """
        設定 baseline score 分布
        
        建議使用 Phase 3 Policy Validation 的預測機率作為 baseline：
        - 路徑：model_bank/<latest_run>/policy_validation_predictions.csv
        - 欄位：pred_prob
        
        為什麼用 Policy Validation？
        - 這是部署前最後一次有真實 label 的評估資料
        - 其 score 分布最接近上線後的 production 分布
        - 與 development data 相比，更能反映 production 真實狀態
        
        此 baseline 用於：
        - PSI 計算（與新進 production 資料比較分布偏移）
        - Zone 分布監控（High / Manual Review / Low 三區間比例）
        """
        self.baseline_scores = scores
        
        lower = lower_threshold or self.config.default_lower_threshold
        upper = upper_threshold or self.config.default_upper_threshold
        
        self.baseline_zone_distribution = {
            'high': float(np.mean(scores >= upper)),
            'manual': float(np.mean((scores >= lower) & (scores < upper))),
            'low': float(np.mean(scores < lower)),
        }
        
        logger.info(f"Baseline 設定完成: {len(scores)} samples")
        logger.info(f"  Zone Distribution: {self.baseline_zone_distribution}")
    
    def set_last_training_date(self, date_str: str):
        """設定上次訓練日期（用於 time trigger）"""
        self.last_training_date = date_str
    
    def run_production_monitoring(
        self,
        predictions: np.ndarray,
        labels: np.ndarray = None,
        model_version: str = "",
        period_start: str = "",
        period_end: str = "",
        lower_threshold: float = None,
        upper_threshold: float = None
    ) -> ProductionMonitoringResult:
        """
        執行 Production Monitoring
        
        Args:
            predictions: 預測機率
            labels: 真實標籤（若有回流）
            model_version: 模型版本
            period_start: 監控期間開始
            period_end: 監控期間結束
            lower_threshold: 三區間低門檻
            upper_threshold: 三區間高門檻
            
        Returns:
            ProductionMonitoringResult
        """
        lower = lower_threshold or self.config.default_lower_threshold
        upper = upper_threshold or self.config.default_upper_threshold
        
        result = ProductionMonitoringResult(
            model_version=model_version,
            monitoring_period_start=period_start,
            monitoring_period_end=period_end,
            total_predictions=len(predictions),
        )
        
        # ===== Score Distribution =====
        result.score_mean = float(np.mean(predictions))
        result.score_std = float(np.std(predictions))
        result.score_median = float(np.median(predictions))
        result.score_min = float(np.min(predictions))
        result.score_max = float(np.max(predictions))
        result.score_q25 = float(np.percentile(predictions, 25))
        result.score_q75 = float(np.percentile(predictions, 75))
        
        # ===== Zone Distribution =====
        zones = np.ones(len(predictions), dtype=int)
        zones[predictions >= upper] = 2
        zones[predictions < lower] = 0
        
        result.zone_high_count = int(np.sum(zones == 2))
        result.zone_manual_count = int(np.sum(zones == 1))
        result.zone_low_count = int(np.sum(zones == 0))
        
        result.zone_high_ratio = result.zone_high_count / len(predictions) if len(predictions) > 0 else 0
        result.zone_manual_ratio = result.zone_manual_count / len(predictions) if len(predictions) > 0 else 0
        result.zone_low_ratio = result.zone_low_count / len(predictions) if len(predictions) > 0 else 0
        
        # ===== Zone Shift (vs Baseline) =====
        if self.baseline_zone_distribution:
            result.baseline_zone_high_ratio = self.baseline_zone_distribution.get('high', 0)
            result.baseline_zone_manual_ratio = self.baseline_zone_distribution.get('manual', 0)
            result.baseline_zone_low_ratio = self.baseline_zone_distribution.get('low', 0)
            
            result.zone_high_shift = result.zone_high_ratio - result.baseline_zone_high_ratio
            result.zone_manual_shift = result.zone_manual_ratio - result.baseline_zone_manual_ratio
            result.zone_low_shift = result.zone_low_ratio - result.baseline_zone_low_ratio
            
            # Zone shift warning
            max_shift = max(abs(result.zone_high_shift), abs(result.zone_manual_shift), abs(result.zone_low_shift))
            if max_shift >= self.config.warning_zone_shift:
                result.warnings.append(f"Zone distribution shift detected: max_shift={max_shift:.2%}")
        
        # ===== Score PSI =====
        if self.baseline_scores is not None:
            psi, _ = calculate_numeric_psi(
                self.baseline_scores.tolist(),
                predictions.tolist()
            )
            result.score_psi = psi
            
            if psi >= self.config.max_score_psi:
                result.alerts.append(f"CRITICAL: Score PSI={psi:.4f} >= {self.config.max_score_psi}")
            elif psi >= self.config.warning_score_psi:
                result.warnings.append(f"WARNING: Score PSI={psi:.4f} >= {self.config.warning_score_psi}")
        
        # ===== Metrics (if labels available) =====
        if labels is not None:
            valid_mask = ~np.isnan(labels)
            n_valid = np.sum(valid_mask)
            
            result.total_with_labels = int(n_valid)
            result.label_coverage_ratio = n_valid / len(predictions) if len(predictions) > 0 else 0
            
            if n_valid > 0:
                y_true = labels[valid_mask].astype(int)
                y_pred_proba = predictions[valid_mask]
                y_pred = (y_pred_proba >= 0.5).astype(int)
                
                result.positive_ratio = float(np.mean(y_true))
                
                try:
                    from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
                    from scipy import stats
                    
                    result.auc = float(roc_auc_score(y_true, y_pred_proba))
                    result.f1_reject = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
                    result.brier_score = float(brier_score_loss(y_true, y_pred_proba))
                    
                    # KS
                    pos_score = y_pred_proba[y_true == 1]
                    neg_score = y_pred_proba[y_true == 0]
                    if len(pos_score) > 0 and len(neg_score) > 0:
                        ks_stat, _ = stats.ks_2samp(pos_score, neg_score)
                        result.ks = float(ks_stat)
                    
                    # Metric alerts
                    if result.auc < self.config.min_auc:
                        result.alerts.append(f"CRITICAL: AUC={result.auc:.4f} < {self.config.min_auc}")
                    elif result.auc < self.config.warning_auc:
                        result.warnings.append(f"WARNING: AUC={result.auc:.4f} < {self.config.warning_auc}")
                    
                    if result.f1_reject < self.config.min_f1_reject:
                        result.alerts.append(f"CRITICAL: F1_reject={result.f1_reject:.4f} < {self.config.min_f1_reject}")
                        
                except Exception as e:
                    logger.warning(f"Metrics calculation error: {e}")
        
        # ===== Check Retraining Trigger =====
        result.needs_retraining, result.retraining_trigger_reason = self._check_retrain_trigger(result)
        
        # Store history
        self.monitoring_history.append(result)
        
        # Log summary
        self._log_monitoring_summary(result)
        
        return result
    
    def _check_retrain_trigger(
        self,
        result: ProductionMonitoringResult
    ) -> Tuple[bool, str]:
        """
        檢查是否需要 Retraining
        
        Trigger A: Metric Trigger
        - AUC < min_auc
        - F1_reject < min_f1_reject
        - Score PSI >= max_score_psi
        
        Trigger B: Time Trigger
        - 距離上次訓練超過 retrain_interval_months
        """
        reasons = []
        
        # Metric Triggers
        if result.auc is not None and result.auc < self.config.min_auc:
            reasons.append(f"AUC={result.auc:.4f} < {self.config.min_auc}")
        
        if result.f1_reject is not None and result.f1_reject < self.config.min_f1_reject:
            reasons.append(f"F1_reject={result.f1_reject:.4f} < {self.config.min_f1_reject}")
        
        if result.score_psi is not None and result.score_psi >= self.config.max_score_psi:
            reasons.append(f"Score PSI={result.score_psi:.4f} >= {self.config.max_score_psi}")
        
        # Time Trigger
        if self.last_training_date:
            time_trigger, time_reason = self.check_time_trigger()
            if time_trigger:
                reasons.append(f"Time: {time_reason}")
        
        if reasons:
            return True, "; ".join(reasons)
        return False, ""
    
    def check_time_trigger(self) -> Tuple[bool, str]:
        """檢查 Time Trigger"""
        if not self.last_training_date:
            return False, ""
        
        last_date = pd.to_datetime(self.last_training_date).date()
        current_date = datetime.now().date()
        
        months_since = (current_date.year - last_date.year) * 12 + (current_date.month - last_date.month)
        
        if months_since >= self.config.retrain_interval_months:
            return True, f"{months_since} months since last training"
        
        return False, ""
    
    def _log_monitoring_summary(self, result: ProductionMonitoringResult):
        """輸出監控摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("Production Monitoring Summary")
        logger.info("=" * 60)
        logger.info(f"Period: {result.monitoring_period_start} ~ {result.monitoring_period_end}")
        logger.info(f"Total Predictions: {result.total_predictions}")
        logger.info(f"Labels Available: {result.total_with_labels} ({result.label_coverage_ratio:.1%})")
        
        logger.info(f"\nScore Distribution:")
        logger.info(f"  Mean: {result.score_mean:.4f}, Std: {result.score_std:.4f}")
        
        logger.info(f"\nZone Distribution:")
        logger.info(f"  High: {result.zone_high_count} ({result.zone_high_ratio:.1%})")
        logger.info(f"  Manual: {result.zone_manual_count} ({result.zone_manual_ratio:.1%})")
        logger.info(f"  Low: {result.zone_low_count} ({result.zone_low_ratio:.1%})")
        
        if result.score_psi is not None:
            logger.info(f"\nScore PSI: {result.score_psi:.4f}")
        
        if result.auc is not None:
            logger.info(f"\nMetrics (from {result.total_with_labels} labeled samples):")
            logger.info(f"  AUC: {result.auc:.4f}")
            logger.info(f"  F1_reject: {result.f1_reject:.4f}")
            if result.ks is not None:
                logger.info(f"  KS: {result.ks:.4f}")
        
        if result.warnings:
            logger.warning(f"\nWarnings:")
            for w in result.warnings:
                logger.warning(f"  ⚠️ {w}")
        
        if result.alerts:
            logger.error(f"\nAlerts:")
            for a in result.alerts:
                logger.error(f"  🚨 {a}")
        
        if result.needs_retraining:
            logger.warning(f"\n🔄 RETRAINING TRIGGERED: {result.retraining_trigger_reason}")
        
        logger.info("=" * 60)
    
    def save_monitoring_history(self, output_path: Path):
        """儲存監控歷史"""
        data = {
            "config": self.config.to_dict(),
            "last_training_date": self.last_training_date,
            "history": [r.to_dict() for r in self.monitoring_history]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Monitoring history saved: {output_path}")


# ============================================
# Check Retrain Trigger (Convenience Function)
# ============================================
def check_retrain_trigger(
    current_metrics: Dict[str, float],
    last_training_date: str = None,
    config: ProductionMonitoringConfig = None
) -> Tuple[bool, str]:
    """
    便利函式：檢查是否需要 Retraining
    
    Args:
        current_metrics: 當前指標 dict (auc, f1_reject, score_psi)
        last_training_date: 上次訓練日期 (YYYY-MM-DD)
        config: Monitoring 設定
        
    Returns:
        (needs_retraining, reason)
    """
    config = config or ProductionMonitoringConfig()
    reasons = []
    
    # Metric Triggers
    auc = current_metrics.get('auc')
    if auc is not None and auc < config.min_auc:
        reasons.append(f"AUC={auc:.4f} < {config.min_auc}")
    
    f1_reject = current_metrics.get('f1_reject')
    if f1_reject is not None and f1_reject < config.min_f1_reject:
        reasons.append(f"F1_reject={f1_reject:.4f} < {config.min_f1_reject}")
    
    score_psi = current_metrics.get('score_psi')
    if score_psi is not None and score_psi >= config.max_score_psi:
        reasons.append(f"Score PSI={score_psi:.4f} >= {config.max_score_psi}")
    
    # Time Trigger
    if last_training_date:
        last_date = pd.to_datetime(last_training_date).date()
        current_date = datetime.now().date()
        months_since = (current_date.year - last_date.year) * 12 + (current_date.month - last_date.month)
        
        if months_since >= config.retrain_interval_months:
            reasons.append(f"Time: {months_since} months since last training")
    
    if reasons:
        return True, "; ".join(reasons)
    return False, ""


# ============================================
# Dynamic Retraining Data Window
# ============================================
def generate_retraining_data_window(
    current_date: Union[str, date],
    development_months: int = 18,
    policy_validation_months: int = 4,
    final_holdout_months: int = 2
) -> Dict[str, str]:
    """
    動態產生 Retraining 資料窗
    
    為什麼要用「當下日期往前推」？
    - 確保模型總是用最新的資料訓練
    - 避免模型陳舊（staleness）
    - 每次 retraining 都能學到最新的 pattern
    
    Args:
        current_date: 當前日期（觸發 retraining 的日期）
        development_months: Development period 長度（月）
        policy_validation_months: Policy Validation period 長度（月）
        final_holdout_months: Final Holdout period 長度（月）
        
    Returns:
        Dict with date ranges for each phase
    """
    if isinstance(current_date, str):
        current_date = pd.to_datetime(current_date).date()
    
    total_months = development_months + policy_validation_months + final_holdout_months
    
    # 往前推
    data_start = current_date - relativedelta(months=total_months)
    
    # Development: 前 18 個月
    dev_start = data_start
    dev_end = data_start + relativedelta(months=development_months) - relativedelta(days=1)
    
    # Policy Validation: 接下來 4 個月
    pv_start = dev_end + relativedelta(days=1)
    pv_end = pv_start + relativedelta(months=policy_validation_months) - relativedelta(days=1)
    
    # Final Holdout: 最後 2 個月
    holdout_start = pv_end + relativedelta(days=1)
    holdout_end = current_date
    
    return {
        "data_start": data_start.strftime("%Y-%m-%d"),
        "data_end": current_date.strftime("%Y-%m-%d"),
        "development_start": dev_start.strftime("%Y-%m-%d"),
        "development_end": dev_end.strftime("%Y-%m-%d"),
        "policy_validation_start": pv_start.strftime("%Y-%m-%d"),
        "policy_validation_end": pv_end.strftime("%Y-%m-%d"),
        "final_holdout_start": holdout_start.strftime("%Y-%m-%d"),
        "final_holdout_end": holdout_end.strftime("%Y-%m-%d"),
    }


def retrain_with_dynamic_window(
    project_root: Path,
    current_date: Union[str, date] = None,
    development_months: int = 18
) -> Dict:
    """
    使用動態資料窗執行 Retraining
    
    Args:
        project_root: 專案根目錄
        current_date: 當前日期（預設今天）
        development_months: Development period 長度
        
    Returns:
        Retraining 結果
    """
    from .four_phase_trainer import FourPhaseTrainer, PhaseConfig
    
    if current_date is None:
        current_date = datetime.now().date()
    
    # 產生資料窗
    window = generate_retraining_data_window(
        current_date=current_date,
        development_months=development_months
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Dynamic Retraining")
    logger.info("=" * 60)
    logger.info(f"Current Date: {current_date}")
    logger.info(f"Data Window: {window['data_start']} ~ {window['data_end']}")
    logger.info(f"  Development: {window['development_start']} ~ {window['development_end']}")
    logger.info(f"  Policy Val: {window['policy_validation_start']} ~ {window['policy_validation_end']}")
    logger.info(f"  Holdout: {window['final_holdout_start']} ~ {window['final_holdout_end']}")
    
    # 執行 retraining（使用 FourPhaseTrainer）
    trainer = FourPhaseTrainer(project_root=project_root)
    
    result = trainer.run_full_pipeline()
    
    return {
        "window": window,
        "result": result
    }


# ============================================
# Production Monitor Sample Output
# ============================================
def generate_production_monitor_sample(output_path: Path):
    """產生 production_monitor_sample.json 範例"""
    sample = {
        "description": "Production Monitoring 結果範例",
        "monitoring_config": ProductionMonitoringConfig().to_dict(),
        "sample_result": ProductionMonitoringResult(
            model_version="xgboost_v20260412",
            monitoring_period_start="2026-04-01",
            monitoring_period_end="2026-04-12",
            total_predictions=5000,
            total_with_labels=3500,
            label_coverage_ratio=0.70,
            score_mean=0.85,
            score_std=0.12,
            zone_high_count=4200,
            zone_high_ratio=0.84,
            zone_manual_count=500,
            zone_manual_ratio=0.10,
            zone_low_count=300,
            zone_low_ratio=0.06,
            score_psi=0.08,
            auc=0.88,
            f1_reject=0.35,
            brier_score=0.05,
            warnings=["WARNING: Score PSI=0.08 approaching threshold"],
            needs_retraining=False,
        ).to_dict()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Production monitor sample saved: {output_path}")
