"""
Four-Phase Training Module v2.0
================================
完整實作四階段 ML Pipeline for Credit Scoring（唯一主訓練架構）

==================================================
四階段流程設計
==================================================

Phase 1: Model Development (前18個月)
- Rolling Window Training
- 每個 cycle = 4 個月 train + 2 個月 monitor
- 目的：Model Selection + Stability Testing
- Rolling 內的模型不直接部署

Phase 2: Champion Retraining
- 使用完整 18 個月 development dataset
- 依照 Phase 1 選出的 champion strategy 重訓
- 此 Final Champion Model 才是後續使用的模型

【重要區分】
- Champion Strategy: 模型+參數的組合方案（如 "xgboost + scale_pos_weight"）
- Final Champion Artifact: 用 champion strategy 在完整資料上訓練出的實際模型檔案

Phase 3: Policy Validation (4個月)
- 不再調整模型權重
- 只做 Threshold / Zone Policy Validation
- 輸出: threshold_policy_comparison.csv

Phase 4: Final Blind Holdout (最後2個月)
- 完全不調模型、不調 threshold
- 真正的 untouched final evaluation
- 這才是最終的 holdout test

==================================================
Production vs Final Blind Holdout 區分
==================================================

Final Blind Holdout Evaluation (Phase 4):
- 有真實 label
- 可以算 AUC / F1 / KS 等指標
- 用於驗證模型在 blind data 上的表現

Production Batch Scoring (上線後):
- 可能沒有真實 label（或 label 延遲回流）
- 只輸出 probability, zone, timestamp, model_version
- 不算 AUC / F1（因為沒有 ground truth）

==================================================
術語定義（全專案統一）
==================================================

- Champion Strategy: 模型 + imbalance strategy + 設定組合（Phase 1 選出）
- Final Champion Artifact: 用 Champion Strategy 在完整 development data 重訓的實際模型檔案（Phase 2 產出）
- Policy Validation: threshold / zone policy tuning window（Phase 3，4 個月）
- Final Blind Holdout: 完全不調模型與 threshold 的最終驗證集（Phase 4，2 個月）
- Production Batch Scoring: 上線後的批次推論，沒有即時 label，只輸出 predictions
- OOT (legacy): Gold Layer 的 legacy 儲存命名，在此模組中拆為 policy_validation + final_holdout

Version: 2.0.0
"""

import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss,
    f1_score, precision_score, recall_score, log_loss
)
import xgboost as xgb

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from .config import (
    ConfigManager, default_config, CONFIG_VERSION,
    BusinessConstraintConfig, ChampionSelectionConfig, TuningConfig
)

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 忽略部分警告
warnings.filterwarnings('ignore', category=UserWarning)

# Random seed
RANDOM_STATE = 2022


# ============================================
# Phase 定義
# ============================================
@dataclass
class PhaseConfig:
    """
    四階段時間配置
    
    預設配置（總共 24 個月）:
    - Phase 1 Development: 前 18 個月 (rolling training)
    - Phase 3 Policy Validation: 接下來 4 個月 (threshold tuning)
    - Phase 4 Final Holdout: 最後 2 個月 (untouched evaluation)
    
    注意：Phase 2 (Champion Retraining) 沒有獨立時間區間，
          它是用 Phase 1 的全部資料重訓。
    """
    development_months: int = 18
    policy_validation_months: int = 4
    final_holdout_months: int = 2
    
    # Rolling window 參數
    rolling_train_months: int = 4
    rolling_monitor_months: int = 2
    rolling_step_months: int = 2  # 每個 cycle 往前推進的月數
    
    def total_months(self) -> int:
        return (self.development_months + 
                self.policy_validation_months + 
                self.final_holdout_months)


# ============================================
# Data Classes
# ============================================
@dataclass
class WindowDefinition:
    """Rolling Window 定義"""
    window_id: int
    train_start: str
    train_end: str
    monitor_start: str
    monitor_end: str
    
    def __post_init__(self):
        """轉換日期格式"""
        if isinstance(self.train_start, str):
            self.train_start_date = pd.to_datetime(self.train_start).date()
        if isinstance(self.train_end, str):
            self.train_end_date = pd.to_datetime(self.train_end).date()
        if isinstance(self.monitor_start, str):
            self.monitor_start_date = pd.to_datetime(self.monitor_start).date()
        if isinstance(self.monitor_end, str):
            self.monitor_end_date = pd.to_datetime(self.monitor_end).date()


@dataclass
class CycleResult:
    """
    單一 Rolling Cycle 的結果
    
    Development-Stage Monitoring Metrics:
    - 這是 rolling 內每個 cycle 的 monitor window 表現
    - 用於 model selection 和 stability testing
    - 不是 production monitoring
    """
    window_id: int
    model_name: str
    
    # Training CV metrics
    cv_auc_mean: float = 0.0
    cv_auc_std: float = 0.0
    cv_f1_mean: float = 0.0
    cv_f1_std: float = 0.0
    cv_f1_reject_mean: float = 0.0
    cv_f1_reject_std: float = 0.0
    cv_precision_mean: float = 0.0
    cv_recall_mean: float = 0.0
    
    # Development-stage Monitor window metrics
    monitor_auc: float = 0.0
    monitor_f1: float = 0.0
    monitor_f1_reject: float = 0.0
    monitor_precision: float = 0.0
    monitor_recall: float = 0.0
    monitor_ks: float = 0.0
    monitor_brier: float = 0.0
    
    # Score distribution
    monitor_score_mean: float = 0.0
    monitor_score_std: float = 0.0
    monitor_score_median: float = 0.0
    
    # Data counts
    train_rows: int = 0
    monitor_rows: int = 0
    train_positive_ratio: float = 0.0
    monitor_positive_ratio: float = 0.0
    
    # Imbalance handling
    imbalance_strategy: str = "scale_weight"
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StrategyResult:
    """單一候選策略的彙總結果（跨所有 cycles）"""
    model_name: str
    
    # Aggregated CV metrics
    avg_cv_auc: float = 0.0
    std_cv_auc: float = 0.0
    avg_cv_f1_reject: float = 0.0
    std_cv_f1_reject: float = 0.0
    
    # Aggregated monitor metrics
    avg_monitor_auc: float = 0.0
    std_monitor_auc: float = 0.0
    avg_monitor_f1: float = 0.0
    std_monitor_f1: float = 0.0
    avg_monitor_f1_reject: float = 0.0
    std_monitor_f1_reject: float = 0.0
    avg_monitor_ks: float = 0.0
    avg_monitor_brier: float = 0.0
    
    # Stability score (lower is better)
    stability_score: float = 0.0
    
    # Overall score (for ranking)
    overall_score: float = 0.0
    
    # Ranking reason (traceability)
    ranking_reason: str = ""
    
    # Individual cycle results
    cycle_results: List[CycleResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['cycle_results'] = [c.to_dict() for c in self.cycle_results]
        return result


@dataclass
class ZoneSummary:
    """三區間分析結果"""
    zone_name: str  # "高通過機率區", "人工審核區", "低通過機率區"
    
    count: int = 0
    ratio: float = 0.0
    
    avg_prob: float = 0.0
    min_prob: float = 0.0
    max_prob: float = 0.0
    
    actual_approve_rate: float = 0.0
    actual_reject_rate: float = 0.0
    
    # Performance in this zone
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ThresholdPolicyResult:
    """
    單一 Threshold 組合的評估結果
    
    用於 Phase 3 Policy Validation：
    讓業務/風控人員可以比較多組 threshold 組合，選擇最適合的門檻。
    
    欄位說明：
    - auto_decision_rate: (高通過區 + 低通過區) / 總筆數，代表可自動決策的比例
    - manual_review_load: 人工審核區 / 總筆數，代表需人工審查的負擔
    """
    lower_threshold: float
    upper_threshold: float
    
    # 高通過機率區（prob >= upper_threshold -> 建議自動核准）
    high_zone_count: int = 0
    high_zone_ratio: float = 0.0
    high_zone_avg_prob: float = 0.0
    high_zone_actual_approve_rate: float = 0.0
    
    # 人工審核區（lower <= prob < upper -> 需人工審查）
    review_zone_count: int = 0
    review_zone_ratio: float = 0.0
    review_zone_avg_prob: float = 0.0
    review_zone_actual_approve_rate: float = 0.0
    
    # 低通過機率區（prob < lower_threshold -> 建議自動婉拒）
    low_zone_count: int = 0
    low_zone_ratio: float = 0.0
    low_zone_avg_prob: float = 0.0
    low_zone_actual_reject_rate: float = 0.0
    
    # 自動決策率 = (high + low) / total
    auto_decision_rate: float = 0.0
    
    # 人工審查負擔 = review / total
    manual_review_load: float = 0.0
    
    # 預期效能
    expected_precision_high: float = 0.0  # 高區域的核准精確度
    expected_precision_low: float = 0.0   # 低區域的婉拒精確度
    
    # Composite threshold score (Phase 3 用於排名)
    threshold_score: float = 0.0
    
    # 是否滿足 business constraints
    passes_hard_constraints: bool = False
    constraint_violations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class DiagnosticsSummary:
    """
    Overfitting / Robustness 診斷摘要
    
    比較 train / monitor / holdout 三個階段的指標差距
    """
    # In-sample (development) metrics
    train_auc: float = 0.0
    train_f1_reject: float = 0.0
    train_ks: float = 0.0
    train_brier: float = 0.0
    
    # Rolling monitor average metrics
    avg_monitor_auc: float = 0.0
    avg_monitor_f1_reject: float = 0.0
    avg_monitor_ks: float = 0.0
    avg_monitor_brier: float = 0.0
    std_monitor_auc: float = 0.0
    
    # Policy validation metrics
    policy_val_auc: float = 0.0
    policy_val_f1_reject: float = 0.0
    policy_val_ks: float = 0.0
    
    # Final holdout metrics
    final_holdout_auc: float = 0.0
    final_holdout_f1_reject: float = 0.0
    final_holdout_ks: float = 0.0
    final_holdout_brier: float = 0.0
    
    # Gap analysis
    gap_train_vs_monitor_auc: float = 0.0
    gap_train_vs_holdout_auc: float = 0.0
    gap_monitor_vs_holdout_auc: float = 0.0
    gap_train_vs_monitor_brier: float = 0.0  # Brier calibration gap
    
    # Overfitting flags
    is_overfitting: bool = False
    overfitting_severity: str = "none"  # "none", "mild", "moderate", "severe"
    
    # Calibration & Reject detection quality
    has_calibration_issue: bool = False     # Brier score 在 monitor 遠大於 train
    has_reject_detection_issue: bool = False  # F1_reject 過低
    
    def compute_gaps(self):
        """
        計算各階段差距
        
        判斷面向：
        1. AUC gap -> 典型 overfitting（Train >> Monitor/Holdout）
        2. Brier gap -> Calibration 偏移（Train Brier << Monitor Brier）
        3. F1_reject -> 少數類辨識力不足
        """
        self.gap_train_vs_monitor_auc = self.train_auc - self.avg_monitor_auc
        self.gap_train_vs_holdout_auc = self.train_auc - self.final_holdout_auc
        self.gap_monitor_vs_holdout_auc = self.avg_monitor_auc - self.final_holdout_auc
        
        # Brier calibration gap
        if self.train_brier > 0 and self.avg_monitor_brier > 0:
            self.gap_train_vs_monitor_brier = self.avg_monitor_brier - self.train_brier
        
        # 判斷是否 overfitting（AUC gap）
        max_gap = max(abs(self.gap_train_vs_monitor_auc), 
                      abs(self.gap_train_vs_holdout_auc))
        
        if max_gap < 0.02:
            self.is_overfitting = False
            self.overfitting_severity = "none"
        elif max_gap < 0.05:
            self.is_overfitting = True
            self.overfitting_severity = "mild"
        elif max_gap < 0.10:
            self.is_overfitting = True
            self.overfitting_severity = "moderate"
        else:
            self.is_overfitting = True
            self.overfitting_severity = "severe"
        
        # 判斷 Calibration 問題：monitor Brier 是 train 的 2 倍以上
        if self.train_brier > 0 and self.avg_monitor_brier > 0:
            brier_ratio = self.avg_monitor_brier / self.train_brier
            self.has_calibration_issue = bool(brier_ratio > 2.0)
        
        # 判斷 Reject detection 問題
        # F1_reject < 0.35 且 holdout F1_reject < 0.45 -> 模型幾乎無法辨識 reject
        if self.final_holdout_f1_reject > 0:
            self.has_reject_detection_issue = bool(self.final_holdout_f1_reject < 0.45)
        
        # 確保所有 boolean / float 欄位為 Python 原生型別（避免 numpy.bool_ / numpy.float64 導致 JSON 序列化失敗）
        self.is_overfitting = bool(self.is_overfitting)
        self.has_calibration_issue = bool(self.has_calibration_issue)
        self.has_reject_detection_issue = bool(self.has_reject_detection_issue)
        self.gap_train_vs_monitor_auc = float(self.gap_train_vs_monitor_auc)
        self.gap_train_vs_holdout_auc = float(self.gap_train_vs_holdout_auc)
        self.gap_monitor_vs_holdout_auc = float(self.gap_monitor_vs_holdout_auc)
        self.gap_train_vs_monitor_brier = float(self.gap_train_vs_monitor_brier)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DecileSummary:
    """Score Band / Decile 分析"""
    decile: int  # 1-10
    score_min: float = 0.0
    score_max: float = 0.0
    count: int = 0
    ratio: float = 0.0
    predicted_avg_prob: float = 0.0
    actual_approve_rate: float = 0.0
    calibration_gap: float = 0.0  # predicted - actual
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================
# Imbalance Handling
# ============================================
class ImbalanceHandler:
    """
    處理 Label Imbalance 的策略
    
    重要：
    - Resampling 只能在 training folds 內做
    - Monitor / Policy Validation / Final Holdout / Production 必須保留真實分布
    
    支援策略：
    1. scale_pos_weight (XGBoost) - 推薦
    2. class_weight (sklearn models)
    3. SMOTE (僅在 training folds 內)
    4. Threshold moving (後處理)
    """
    
    AVAILABLE_STRATEGIES = ["scale_weight", "class_weight", "smote", "undersample", "none"]
    
    def __init__(
        self,
        strategy: str = "scale_weight",
        random_state: int = RANDOM_STATE
    ):
        if strategy not in self.AVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {self.AVAILABLE_STRATEGIES}")
        self.strategy = strategy
        self.random_state = random_state
        
    def calculate_scale_pos_weight(self, y: np.ndarray) -> float:
        """計算 XGBoost scale_pos_weight"""
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        
        if n_positive == 0:
            return 1.0
        
        return n_negative / n_positive
    
    def calculate_class_weight(self, y: np.ndarray) -> Dict[int, float]:
        """計算 sklearn class_weight"""
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    def get_positive_ratio(self, y: np.ndarray) -> float:
        """取得正樣本比例"""
        return float(np.mean(y))
    
    def resample_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        對訓練資料進行 resampling
        
        WARNING: 只能在 training folds 內使用！
        WARNING: Monitor / Validation / Production data 必須保留原始分布！
        """
        if self.strategy == "smote":
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=self.random_state)
                X_res, y_res = smote.fit_resample(X, y)
                logger.info(f"SMOTE resampling: {len(y)} -> {len(y_res)} samples")
                return X_res, y_res
            except ImportError:
                logger.warning("imblearn not installed, using scale_weight instead")
                return X, y
                
        elif self.strategy == "undersample":
            from sklearn.utils import resample
            
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            
            # 將多數類下採樣到少數類的 2 倍
            n_target = min(len(pos_idx), len(neg_idx) * 2)
            
            if len(pos_idx) > n_target:
                pos_idx_resampled = resample(
                    pos_idx,
                    replace=False,
                    n_samples=n_target,
                    random_state=self.random_state
                )
                all_idx = np.concatenate([pos_idx_resampled, neg_idx])
                np.random.shuffle(all_idx)
                
                logger.info(f"Undersample: {len(y)} -> {len(all_idx)} samples")
                return X[all_idx], y[all_idx]
            
            return X, y
        
        else:
            # scale_weight / class_weight / none 不需要 resample
            return X, y


# ============================================
# Time-Based Cross Validation
# ============================================
class TimeBasedCV:
    """
    Time-Based Cross Validation (Walk-Forward / Expanding Window)
    
    確保訓練時只用「過去」預測「未來」，避免 data leakage
    """
    
    def __init__(
        self,
        n_splits: int = 2,
        min_train_periods: int = 2
    ):
        self.n_splits = n_splits
        self.min_train_periods = min_train_periods
    
    def split(
        self,
        df: pd.DataFrame,
        date_column: str = "進件日"
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """產生 time-based CV splits"""
        df = df.sort_values(date_column).reset_index(drop=True)
        
        df['_year_month'] = pd.to_datetime(df[date_column]).dt.to_period('M')
        unique_months = sorted(df['_year_month'].unique())
        
        n_months = len(unique_months)
        splits = []
        
        for i in range(self.min_train_periods, n_months):
            train_months = unique_months[:i]
            val_months = [unique_months[i]]
            
            train_mask = df['_year_month'].isin(train_months)
            val_mask = df['_year_month'].isin(val_months)
            
            train_idx = df[train_mask].index.values
            val_idx = df[val_mask].index.values
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
                if len(splits) >= self.n_splits:
                    break
        
        df.drop(columns=['_year_month'], inplace=True, errors='ignore')
        
        logger.info(f"Time-based CV: 產生 {len(splits)} 個 folds")
        for i, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"  Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        return splits


# ============================================
# Candidate Models
# ============================================
class CandidateModels:
    """
    候選模型工廠
    
    設計原則（針對 ~95% 正樣本的高度不平衡資料）：
    - 偏向 conservative / 強正則化，避免模型把所有案件都預測為正類
    - XGBoost: 淺樹 + 高正則化 + 低學習率 -> 防止過度擬合多數類
    - Random Forest: 限制深度 + 增加葉節點最小樣本 -> 提升泛化
    - Logistic Regression: 適當正則化 -> 作為 baseline
    """
    
    @staticmethod
    def get_logistic_regression(class_weight: Dict = None) -> LogisticRegression:
        return LogisticRegression(
            penalty='l2',
            C=0.5,               # <- 加強正則化（原 1.0），避免對多數類過度自信
            class_weight=class_weight or 'balanced',
            solver='lbfgs',
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
    @staticmethod
    def get_random_forest(class_weight: Dict = None) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=300,     # <- 增加樹數量（原 200），降低 variance
            max_depth=6,          # <- 降低深度（原 10），防止過度擬合少數類的雜訊
            min_samples_split=50, # <- 提高（原 20），節點切分需更多樣本
            min_samples_leaf=20,  # <- 提高（原 10），葉節點至少 20 筆
            max_features='sqrt',  # <- 限制每棵樹可用特徵數
            class_weight=class_weight or 'balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
    @staticmethod
    def get_xgboost(scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
        """
        注意：early_stopping_rounds 不在此處設定。
        因為 CalibratedClassifierCV 等 sklearn wrapper 內部 fit 時不會傳 eval_set，
        若模型自帶 early_stopping_rounds 會直接報錯。
        early stopping 改由呼叫端在 fit() 時手動設定（見 _train_window / _run_time_based_cv）。
        """
        return xgb.XGBClassifier(
            n_estimators=300,         # <- 增加（原 200），搭配更低的 learning_rate
            max_depth=3,              # <- 降低（原 5），淺樹 = 更強泛化
            learning_rate=0.03,       # <- 降低（原 0.05），學得更慢更穩
            min_child_weight=20,      # <- 提高（原 5），每個葉節點需更多樣本
            subsample=0.7,            # <- 降低（原 0.8），增加隨機性
            colsample_bytree=0.7,     # <- 降低（原 0.8），增加特徵隨機性
            reg_alpha=1.0,            # <- 提高（原 0.1），L1 正則化
            reg_lambda=5.0,           # <- 提高（原 1.0），L2 正則化
            gamma=1.0,                # <- 新增，節點切分最小 gain 門檻
            max_delta_step=1,         # <- 新增，限制葉節點權重變化（對 imbalance 有幫助）
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
    @staticmethod
    def get_all_candidates(
        scale_pos_weight: float = 1.0,
        class_weight: Dict = None
    ) -> Dict[str, Any]:
        return {
            "logistic_regression": CandidateModels.get_logistic_regression(class_weight),
            "random_forest": CandidateModels.get_random_forest(class_weight),
            "xgboost": CandidateModels.get_xgboost(scale_pos_weight),
        }


# ============================================
# Metrics Calculator
# ============================================
class MetricsCalculator:
    """計算各種評估指標"""
    
    @staticmethod
    def calculate_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """計算 KS Statistic"""
        from scipy import stats
        pos_score = y_score[y_true == 1]
        neg_score = y_score[y_true == 0]
        
        if len(pos_score) == 0 or len(neg_score) == 0:
            return 0.0
            
        ks_stat, _ = stats.ks_2samp(pos_score, neg_score)
        return float(ks_stat)
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """計算所有評估指標"""
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {}
        
        # AUC
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['auc'] = 0.0
        
        # AUC-PR
        try:
            metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba)
        except:
            metrics['auc_pr'] = 0.0
        
        # Brier Score
        metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
        
        # F1, Precision, Recall (for positive class)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        
        # F1, Precision, Recall (for reject class = 0)
        metrics['f1_reject'] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        metrics['precision_reject'] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        metrics['recall_reject'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        
        # KS
        metrics['ks'] = MetricsCalculator.calculate_ks(y_true, y_pred_proba)
        
        # Confusion Matrix
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['true_negative'] = int(tn)
            metrics['false_positive'] = int(fp)
            metrics['false_negative'] = int(fn)
            metrics['true_positive'] = int(tp)
        except:
            metrics['true_negative'] = 0
            metrics['false_positive'] = 0
            metrics['false_negative'] = 0
            metrics['true_positive'] = 0
        
        # Positive ratio
        metrics['positive_ratio'] = float(np.mean(y_true))
        
        return metrics
    
    @staticmethod
    def calculate_score_distribution(y_pred_proba: np.ndarray) -> Dict[str, float]:
        """計算 score distribution 統計"""
        return {
            'mean': float(np.mean(y_pred_proba)),
            'std': float(np.std(y_pred_proba)),
            'median': float(np.median(y_pred_proba)),
            'min': float(np.min(y_pred_proba)),
            'max': float(np.max(y_pred_proba)),
            'q25': float(np.percentile(y_pred_proba, 25)),
            'q75': float(np.percentile(y_pred_proba, 75)),
        }
    
    @staticmethod
    def calculate_decile_summary(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_deciles: int = 10
    ) -> List[DecileSummary]:
        """
        計算 Score Band / Decile 分析
        
        用於 Calibration 檢查：predicted probability 是否接近 actual rate
        """
        # 按 probability 分組
        percentiles = np.linspace(0, 100, n_deciles + 1)
        bins = np.percentile(y_pred_proba, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        # 去重複
        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.array([-np.inf, np.inf])
        
        digitized = np.digitize(y_pred_proba, bins) - 1
        digitized = np.clip(digitized, 0, len(bins) - 2)
        
        summaries = []
        total = len(y_true)
        
        for i in range(len(bins) - 1):
            mask = (digitized == i)
            count = np.sum(mask)
            
            if count == 0:
                continue
            
            y_decile = y_true[mask]
            prob_decile = y_pred_proba[mask]
            
            predicted_avg = float(np.mean(prob_decile))
            actual_rate = float(np.mean(y_decile))
            
            summary = DecileSummary(
                decile=i + 1,
                score_min=float(bins[i]) if bins[i] != -np.inf else 0.0,
                score_max=float(bins[i + 1]) if bins[i + 1] != np.inf else 1.0,
                count=int(count),
                ratio=count / total,
                predicted_avg_prob=predicted_avg,
                actual_approve_rate=actual_rate,
                calibration_gap=predicted_avg - actual_rate
            )
            summaries.append(summary)
        
        return summaries


# ============================================
# Score Zone Assignment
# ============================================
def assign_score_zone(
    prob: np.ndarray,
    lower_threshold: float = 0.5,
    upper_threshold: float = 0.85
) -> np.ndarray:
    """
    分配三區間
    
    - 2: 高通過機率區 (prob >= upper_threshold) -> 自動核准
    - 1: 人工審核區 (lower_threshold <= prob < upper_threshold)
    - 0: 低通過機率區 (prob < lower_threshold) -> 自動拒絕
    """
    zone = np.ones(len(prob), dtype=int)  # 預設人工審核
    zone[prob >= upper_threshold] = 2
    zone[prob < lower_threshold] = 0
    
    return zone


def evaluate_zone_performance(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    lower_threshold: float = 0.5,
    upper_threshold: float = 0.85
) -> List[ZoneSummary]:
    """評估三區間的表現"""
    zones = assign_score_zone(y_pred_proba, lower_threshold, upper_threshold)
    total = len(y_true)
    
    zone_names = {
        2: "高通過機率區",
        1: "人工審核區",
        0: "低通過機率區"
    }
    
    results = []
    
    for zone_id in [2, 1, 0]:
        mask = (zones == zone_id)
        count = np.sum(mask)
        
        if count == 0:
            summary = ZoneSummary(zone_name=zone_names[zone_id], count=0, ratio=0.0)
        else:
            y_zone = y_true[mask]
            prob_zone = y_pred_proba[mask]
            
            actual_approve_rate = np.mean(y_zone) if len(y_zone) > 0 else 0
            actual_reject_rate = 1 - actual_approve_rate
            
            summary = ZoneSummary(
                zone_name=zone_names[zone_id],
                count=int(count),
                ratio=count / total,
                avg_prob=float(np.mean(prob_zone)),
                min_prob=float(np.min(prob_zone)),
                max_prob=float(np.max(prob_zone)),
                actual_approve_rate=float(actual_approve_rate),
                actual_reject_rate=float(actual_reject_rate),
            )
        
        results.append(summary)
    
    return results


# ============================================
# Threshold Grid Evaluation
# ============================================
def evaluate_threshold_grid(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    lower_thresholds: List[float] = None,
    upper_thresholds: List[float] = None
) -> List[ThresholdPolicyResult]:
    """
    評估多組 threshold 組合
    
    用於 Phase 3 Policy Validation
    讓業務/風控人員可以依據結果選擇 threshold
    """
    if lower_thresholds is None:
        # 針對高 imbalance 場景（正樣本 ~95%）擴大搜索範圍
        # 包含更高的 lower_threshold 以便在分數壓縮時仍能有效區分
        lower_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    if upper_thresholds is None:
        # 包含更高的 upper_threshold（0.90, 0.95）以適應分數集中在高位的情況
        upper_thresholds = [0.70, 0.80, 0.85, 0.90, 0.95]
    
    results = []
    total = len(y_true)
    
    for lower in lower_thresholds:
        for upper in upper_thresholds:
            if lower >= upper:
                continue
            
            zones = assign_score_zone(y_pred_proba, lower, upper)
            
            # 高通過機率區
            high_mask = (zones == 2)
            high_count = np.sum(high_mask)
            
            # 人工審核區
            review_mask = (zones == 1)
            review_count = np.sum(review_mask)
            
            # 低通過機率區
            low_mask = (zones == 0)
            low_count = np.sum(low_mask)
            
            # 計算各區域的指標
            result = ThresholdPolicyResult(
                lower_threshold=lower,
                upper_threshold=upper,
                high_zone_count=int(high_count),
                high_zone_ratio=high_count / total if total > 0 else 0,
                high_zone_avg_prob=float(np.mean(y_pred_proba[high_mask])) if high_count > 0 else 0,
                high_zone_actual_approve_rate=float(np.mean(y_true[high_mask])) if high_count > 0 else 0,
                review_zone_count=int(review_count),
                review_zone_ratio=review_count / total if total > 0 else 0,
                review_zone_avg_prob=float(np.mean(y_pred_proba[review_mask])) if review_count > 0 else 0,
                review_zone_actual_approve_rate=float(np.mean(y_true[review_mask])) if review_count > 0 else 0,
                low_zone_count=int(low_count),
                low_zone_ratio=low_count / total if total > 0 else 0,
                low_zone_avg_prob=float(np.mean(y_pred_proba[low_mask])) if low_count > 0 else 0,
                low_zone_actual_reject_rate=float(1 - np.mean(y_true[low_mask])) if low_count > 0 else 0,
                auto_decision_rate=(high_count + low_count) / total if total > 0 else 0,
                manual_review_load=review_count / total if total > 0 else 0,
                expected_precision_high=float(np.mean(y_true[high_mask])) if high_count > 0 else 0,
                expected_precision_low=float(1 - np.mean(y_true[low_mask])) if low_count > 0 else 0,
            )
            
            results.append(result)
    
    return results


def score_threshold_policy(
    results: List[ThresholdPolicyResult],
    constraints: BusinessConstraintConfig = None
) -> List[ThresholdPolicyResult]:
    """
    對 threshold 組合進行 business constraint 過濾 + composite scoring
    
    流程：
    1. Hard Constraint 過濾：不符合的標記 passes_hard_constraints=False
    2. Composite Scoring：對每個組合計算 threshold_score
    3. 回傳按 threshold_score 排序的結果
    
    Scoring 公式:
      threshold_score = w_precision * precision_component
                      + w_auto_rate * auto_rate_component
                      + w_zone_balance * zone_balance_component
    
    其中：
    - precision_component = (precision_high + precision_low) / 2
    - auto_rate_component = auto_decision_rate（越高越好）
    - zone_balance_component = 1 - |high_ratio - target|（避免極端分布）
    """
    if constraints is None:
        constraints = BusinessConstraintConfig()
    
    for r in results:
        # ── 1. Hard Constraint Check ──
        violations = []
        
        if r.manual_review_load > constraints.max_manual_review_ratio:
            violations.append(
                f"manual_review={r.manual_review_load:.2%} > max={constraints.max_manual_review_ratio:.2%}"
            )
        
        if r.auto_decision_rate < constraints.min_auto_decision_rate:
            violations.append(
                f"auto_decision_rate={r.auto_decision_rate:.2%} < min={constraints.min_auto_decision_rate:.2%}"
            )
        
        if r.low_zone_ratio < constraints.min_low_zone_ratio:
            violations.append(
                f"low_zone_ratio={r.low_zone_ratio:.2%} < min={constraints.min_low_zone_ratio:.2%}"
            )
        
        r.constraint_violations = violations
        r.passes_hard_constraints = (len(violations) == 0)
        
        # ── 2. Composite Scoring ──
        # Precision component: avg of high and low zone precision
        precision_component = (r.expected_precision_high + r.expected_precision_low) / 2
        
        # Auto decision rate component: directly use the rate
        auto_rate_component = r.auto_decision_rate
        
        # Zone balance component:
        # Penalize if auto_decision_rate deviates too far from target
        # Also penalize if low_zone_ratio is too small (model lacks reject power)
        deviation_from_target = abs(r.auto_decision_rate - constraints.target_auto_decision_rate)
        zone_balance_component = max(0.0, 1.0 - deviation_from_target * 2)
        
        # Bonus for having sufficient low zone (reject power)
        if r.low_zone_ratio >= constraints.min_low_zone_ratio:
            zone_balance_component += 0.1
        
        # Soft target bonus/penalty
        soft_bonus = 0.0
        if r.expected_precision_high >= constraints.min_high_zone_precision:
            soft_bonus += 0.05
        if r.expected_precision_low >= constraints.min_low_zone_reject_precision:
            soft_bonus += 0.05
        
        r.threshold_score = (
            constraints.w_precision * precision_component +
            constraints.w_auto_rate * auto_rate_component +
            constraints.w_zone_balance * zone_balance_component +
            soft_bonus
        )
    
    # Sort by: passes_hard_constraints first (True > False), then threshold_score desc
    results.sort(key=lambda x: (x.passes_hard_constraints, x.threshold_score), reverse=True)
    
    return results


# ============================================
# Data Splitter (Phase-Based)
# ============================================
def split_development_policy_holdout(
    df: pd.DataFrame,
    date_column: str = "進件日",
    phase_config: PhaseConfig = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    將資料切分為三個 phase
    
    Args:
        df: 包含所有資料的 DataFrame
        date_column: 日期欄位
        phase_config: 時間配置
        
    Returns:
        (development_df, policy_validation_df, final_holdout_df)
    """
    if phase_config is None:
        phase_config = PhaseConfig()
    
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)
    
    # 取得日期範圍
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    
    total_months = phase_config.total_months()
    
    # 計算切分點
    dev_end = min_date + relativedelta(months=phase_config.development_months)
    policy_end = dev_end + relativedelta(months=phase_config.policy_validation_months)
    
    # 切分
    development_df = df[df[date_column] < dev_end].copy()
    policy_validation_df = df[(df[date_column] >= dev_end) & (df[date_column] < policy_end)].copy()
    final_holdout_df = df[df[date_column] >= policy_end].copy()
    
    logger.info(f"資料切分完成：")
    logger.info(f"  Development (Phase 1): {len(development_df)} 筆 ({min_date.date()} ~ {dev_end.date()})")
    logger.info(f"  Policy Validation (Phase 3): {len(policy_validation_df)} 筆 ({dev_end.date()} ~ {policy_end.date()})")
    logger.info(f"  Final Holdout (Phase 4): {len(final_holdout_df)} 筆 ({policy_end.date()} ~ {max_date.date()})")
    
    return development_df, policy_validation_df, final_holdout_df


# ============================================
# Four Phase Trainer (主要類別)
# ============================================
class FourPhaseTrainer:
    """
    Four Phase Training Framework
    
    四階段流程：
    1. Phase 1: Model Development (Rolling Training)
    2. Phase 2: Champion Retraining
    3. Phase 3: Policy Validation
    4. Phase 4: Final Blind Holdout
    """
    
    def __init__(
        self,
        project_root: Path,
        phase_config: PhaseConfig = None,
        imbalance_strategy: str = "scale_weight",
        random_state: int = RANDOM_STATE,
        config: ConfigManager = None
    ):
        self.project_root = Path(project_root)
        self.phase_config = phase_config or PhaseConfig()
        self.random_state = random_state
        self.config = config or default_config
        
        # Paths
        self.gold_path = self.project_root / "datamart" / "gold"
        self.development_path = self.gold_path / "development"
        self.oot_path = self.gold_path / "oot"  # Gold Layer legacy storage name
        self.rolling_def_path = self.gold_path / "rolling_window_definition.csv"
        self.output_path = self.project_root / "model_bank"
        
        # Handlers
        self.imbalance_handler = ImbalanceHandler(strategy=imbalance_strategy)
        self.time_cv = TimeBasedCV(n_splits=2)
        self.metrics_calc = MetricsCalculator()
        
        # Results storage
        self.window_definitions: List[WindowDefinition] = []
        self.all_cycle_results: Dict[str, List[CycleResult]] = {}
        self.strategy_results: Dict[str, StrategyResult] = {}
        self.champion_strategy: Optional[str] = None
        self.final_champion_model = None
        self.feature_names: List[str] = []
        
        # Phase data
        self.development_df: Optional[pd.DataFrame] = None
        self.policy_validation_df: Optional[pd.DataFrame] = None
        self.final_holdout_df: Optional[pd.DataFrame] = None
        
        # Diagnostics
        self.diagnostics: Optional[DiagnosticsSummary] = None
        
        # Selected threshold (Phase 3 推薦 -> Phase 4 / Monitoring 使用)
        self.selected_lower_threshold: Optional[float] = None
        self.selected_upper_threshold: Optional[float] = None
        
        # Tuning state
        self.tuning_results: Optional[List[Dict]] = None
        self.tuning_best: Optional[Dict] = None
        
        # Spark session
        self.spark: Optional[SparkSession] = None
        
    def _get_spark(self) -> SparkSession:
        if self.spark is None:
            self.spark = SparkSession.builder \
                .appName("FourPhaseTrainer") \
                .getOrCreate()
        return self.spark
    
    def _stop_spark(self):
        if self.spark is not None:
            self.spark.stop()
            self.spark = None
    
    # ============================================
    # 資料載入
    # ============================================
    def load_all_data(self) -> pd.DataFrame:
        """
        載入所有資料（development + oot 合併）
        
        注意："oot" 是 Gold Layer 的 legacy 儲存命名。
        合併後會由 split_development_policy_holdout() 重新切分為
        development / policy_validation / final_holdout 三段。
        """
        logger.info("載入所有資料...")
        spark = self._get_spark()
        
        dfs = []
        
        # Development
        if self.development_path.exists():
            df_dev = spark.read.parquet(str(self.development_path)).toPandas()
            dfs.append(df_dev)
            logger.info(f"  Development: {len(df_dev)} 筆")
        
        # OOT (Gold Layer legacy storage name -> 後續拆為 policy_validation + final_holdout)
        if self.oot_path.exists():
            df_oot = spark.read.parquet(str(self.oot_path)).toPandas()
            dfs.append(df_oot)
            logger.info(f"  OOT (legacy storage): {len(df_oot)} 筆 -> 後續拆為 policy_validation + final_holdout")
        
        if not dfs:
            raise ValueError("找不到任何資料！")
        
        all_data = pd.concat(dfs, ignore_index=True)
        logger.info(f"  總共: {len(all_data)} 筆")
        
        return all_data
    
    def load_development_data(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """載入 development 資料"""
        logger.info(f"載入 Development 資料...")
        spark = self._get_spark()
        
        df_spark = spark.read.parquet(str(self.development_path))
        
        if start_date:
            df_spark = df_spark.filter(F.col("進件日") >= start_date)
        if end_date:
            df_spark = df_spark.filter(F.col("進件日") <= end_date)
        
        df = df_spark.toPandas()
        logger.info(f"Development 資料: {len(df)} 筆")
        
        return df
    
    def load_rolling_window_definition(self) -> List[WindowDefinition]:
        """載入 rolling window 定義"""
        logger.info("載入 Rolling Window Definition...")
        
        if not self.rolling_def_path.exists():
            raise FileNotFoundError(f"找不到 rolling window 定義: {self.rolling_def_path}")
        
        df = pd.read_csv(self.rolling_def_path)
        
        self.window_definitions = []
        for _, row in df.iterrows():
            window = WindowDefinition(
                window_id=int(row['window_id']),
                train_start=row['train_start'],
                train_end=row['train_end'],
                monitor_start=row['monitor_start'],
                monitor_end=row['monitor_end']
            )
            self.window_definitions.append(window)
        
        logger.info(f"載入 {len(self.window_definitions)} 個 rolling windows")
        for w in self.window_definitions:
            logger.info(f"  Window {w.window_id}: Train [{w.train_start} ~ {w.train_end}], "
                       f"Monitor [{w.monitor_start} ~ {w.monitor_end}]")
        
        return self.window_definitions
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """取得特徵欄位"""
        exclude_cols = [
            '案件編號', '進件日', '進件年月', '授信結果', '授信結果_二元',
            '性別', '教育程度', '婚姻狀況', '月所得', '職業說明', '居住地', '廠牌車型', '動產設定',
            'bronze_load_timestamp', 'silver_process_timestamp', 'gold_process_timestamp',
            '年', '月',
        ]
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        return feature_cols
    
    def _prepare_xy(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """準備 X, y"""
        if feature_cols is None:
            feature_cols = self._get_feature_columns(df)
        
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[feature_cols].values.astype(np.float32)
        y = df['授信結果_二元'].values.astype(int)
        
        X = np.nan_to_num(X, nan=0.0)
        
        return X, y, feature_cols
    
    # ============================================
    # Phase 1: Rolling Training
    # ============================================
    def run_phase1_rolling_training(
        self,
        model_names: List[str] = None
    ) -> Dict[str, List[CycleResult]]:
        """
        Phase 1: Model Development
        
        目的：
        - Model Selection
        - Stability Testing
        - Development-stage Monitoring
        
        Rolling 內每個 cycle 的模型不直接部署
        """
        logger.info("\n" + "=" * 80)
        logger.info("Phase 1: Model Development (Rolling Training)")
        logger.info("=" * 80)
        
        if not self.window_definitions:
            self.load_rolling_window_definition()
        
        if model_names is None:
            model_names = ["logistic_regression", "random_forest", "xgboost"]
        
        self.all_cycle_results = {name: [] for name in model_names}
        
        for window in self.window_definitions:
            cycle_results = self._train_window(window, model_names)
            
            for model_name, result in cycle_results.items():
                self.all_cycle_results[model_name].append(result)
        
        logger.info("\n" + "=" * 80)
        logger.info("Phase 1 完成：Rolling Training")
        logger.info("=" * 80)
        
        return self.all_cycle_results
    
    def _train_window(
        self,
        window: WindowDefinition,
        model_names: List[str]
    ) -> Dict[str, CycleResult]:
        """訓練單一 window"""
        logger.info("=" * 60)
        logger.info(f"Window {window.window_id}: 訓練候選模型")
        logger.info(f"  Training: {window.train_start} ~ {window.train_end}")
        logger.info(f"  Monitor: {window.monitor_start} ~ {window.monitor_end}")
        logger.info("=" * 60)
        
        # 載入資料
        train_df = self.load_development_data(
            start_date=window.train_start,
            end_date=window.train_end
        )
        
        if len(train_df) == 0:
            logger.warning(f"Window {window.window_id}: 無訓練資料！")
            return {}
        
        monitor_df = self.load_development_data(
            start_date=window.monitor_start,
            end_date=window.monitor_end
        )
        
        if len(monitor_df) == 0:
            logger.warning(f"Window {window.window_id}: 無監控資料！")
            return {}
        
        # 準備特徵
        X_train, y_train, feature_cols = self._prepare_xy(train_df)
        X_monitor, y_monitor, _ = self._prepare_xy(monitor_df, feature_cols)
        
        self.feature_names = feature_cols
        
        train_pos_ratio = np.mean(y_train)
        monitor_pos_ratio = np.mean(y_monitor)
        
        logger.info(f"訓練資料: {len(y_train)} 筆, 正樣本比例: {train_pos_ratio:.2%}")
        logger.info(f"監控資料: {len(y_monitor)} 筆, 正樣本比例: {monitor_pos_ratio:.2%}")
        
        # Imbalance handling
        scale_pos_weight = self.imbalance_handler.calculate_scale_pos_weight(y_train)
        class_weight = self.imbalance_handler.calculate_class_weight(y_train)
        
        candidates = CandidateModels.get_all_candidates(scale_pos_weight, class_weight)
        cv_splits = self.time_cv.split(train_df.reset_index(drop=True))
        
        results = {}
        
        for model_name in model_names:
            if model_name not in candidates:
                continue
                
            logger.info(f"\n--- 訓練 {model_name} ---")
            
            model_template = candidates[model_name]
            
            # Time-based CV
            cv_metrics = self._run_time_based_cv(
                model_template, X_train, y_train,
                train_df.reset_index(drop=True), cv_splits
            )
            
            # 用完整 window 訓練
            model = deepcopy(model_template)
            
            # XGBoost: 使用 early stopping 避免過度訓練
            # early_stopping_rounds 在此手動設定（模型 template 不帶此參數）
            if model_name == 'xgboost' and len(y_monitor) > 0:
                model.set_params(early_stopping_rounds=30)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_monitor, y_monitor)],
                    verbose=False
                )
                if hasattr(model, 'best_iteration'):
                    logger.info(f"  XGBoost early stopping at iteration {model.best_iteration}")
            else:
                model.fit(X_train, y_train)
            
            # Monitor 評估（保留真實分布，不做 resample）
            monitor_pred_proba = model.predict_proba(X_monitor)[:, 1]
            monitor_metrics = self.metrics_calc.calculate_all_metrics(y_monitor, monitor_pred_proba)
            score_dist = self.metrics_calc.calculate_score_distribution(monitor_pred_proba)
            
            result = CycleResult(
                window_id=window.window_id,
                model_name=model_name,
                cv_auc_mean=cv_metrics['auc_mean'],
                cv_auc_std=cv_metrics['auc_std'],
                cv_f1_mean=cv_metrics['f1_mean'],
                cv_f1_std=cv_metrics['f1_std'],
                cv_f1_reject_mean=cv_metrics['f1_reject_mean'],
                cv_f1_reject_std=cv_metrics['f1_reject_std'],
                cv_precision_mean=cv_metrics['precision_mean'],
                cv_recall_mean=cv_metrics['recall_mean'],
                monitor_auc=monitor_metrics['auc'],
                monitor_f1=monitor_metrics['f1'],
                monitor_f1_reject=monitor_metrics['f1_reject'],
                monitor_precision=monitor_metrics['precision'],
                monitor_recall=monitor_metrics['recall'],
                monitor_ks=monitor_metrics['ks'],
                monitor_brier=monitor_metrics['brier_score'],
                monitor_score_mean=score_dist['mean'],
                monitor_score_std=score_dist['std'],
                monitor_score_median=score_dist['median'],
                train_rows=len(y_train),
                monitor_rows=len(y_monitor),
                train_positive_ratio=float(train_pos_ratio),
                monitor_positive_ratio=float(monitor_pos_ratio),
                imbalance_strategy=self.imbalance_handler.strategy,
            )
            
            results[model_name] = result
            
            logger.info(f"  CV AUC: {result.cv_auc_mean:.4f} ± {result.cv_auc_std:.4f}")
            logger.info(f"  Monitor AUC: {result.monitor_auc:.4f}")
            logger.info(f"  Monitor F1_reject: {result.monitor_f1_reject:.4f}")
            logger.info(f"  Monitor KS: {result.monitor_ks:.4f}")
            logger.info(f"  Monitor Brier: {result.monitor_brier:.4f}")
        
        return results
    
    def _run_time_based_cv(
        self,
        model_template,
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        cv_splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, float]:
        """執行 Time-based CV"""
        auc_scores, f1_scores, f1_reject_scores = [], [], []
        precision_scores, recall_scores = [], []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Resampling 只在 training fold 內做
            if self.imbalance_handler.strategy in ["smote", "undersample"]:
                X_fold_train, y_fold_train = self.imbalance_handler.resample_training_data(
                    X_fold_train, y_fold_train
                )
            
            model = deepcopy(model_template)

            if hasattr(model, 'scale_pos_weight'):
                new_weight = self.imbalance_handler.calculate_scale_pos_weight(y_fold_train)
                model.set_params(scale_pos_weight=new_weight)

            # If model is XGBoost, enable early stopping with eval_set.
            # early_stopping_rounds is NOT set on the model template (to avoid
            # breaking CalibratedClassifierCV), so we set it here explicitly.
            is_xgb = isinstance(model, xgb.XGBClassifier)

            if is_xgb:
                model.set_params(early_stopping_rounds=30)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False
                )
            else:
                model.fit(X_fold_train, y_fold_train)
            
            # Validation 保留真實分布
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            metrics = self.metrics_calc.calculate_all_metrics(y_fold_val, y_pred_proba)
            
            auc_scores.append(metrics['auc'])
            f1_scores.append(metrics['f1'])
            f1_reject_scores.append(metrics['f1_reject'])
            precision_scores.append(metrics['precision'])
            recall_scores.append(metrics['recall'])
        
        return {
            'auc_mean': float(np.mean(auc_scores)) if auc_scores else 0.0,
            'auc_std': float(np.std(auc_scores)) if auc_scores else 0.0,
            'f1_mean': float(np.mean(f1_scores)) if f1_scores else 0.0,
            'f1_std': float(np.std(f1_scores)) if f1_scores else 0.0,
            'f1_reject_mean': float(np.mean(f1_reject_scores)) if f1_reject_scores else 0.0,
            'f1_reject_std': float(np.std(f1_reject_scores)) if f1_reject_scores else 0.0,
            'precision_mean': float(np.mean(precision_scores)) if precision_scores else 0.0,
            'recall_mean': float(np.mean(recall_scores)) if recall_scores else 0.0,
        }
    
    def select_champion_strategy(self) -> str:
        """
        彙總 Rolling 結果並選出 Champion Strategy
        
        Champion Strategy vs Final Champion Artifact:
        - Champion Strategy: 模型+參數的組合方案
        - Final Champion Artifact: 用該策略在完整資料上訓練的實際模型
        
        Scoring 設計理念（V2）：
        - f1_reject 最高權重（信用風險核心：篩出壞客戶）
        - stability 懲罰加重（生產穩定性基礎）
        - KS 提升（區分力比純 AUC 更有業務意義）
        - CV AUC 降低（基礎校驗，不主導排名）
        
        權重來源: ConfigManager.champion_selection (ChampionSelectionConfig)
        """
        logger.info("\n彙總 Rolling Results...")
        
        # 讀取 champion selection 權重設定
        cs_cfg = self.config.champion_selection if hasattr(self.config, 'champion_selection') else ChampionSelectionConfig()
        
        logger.info(f"Champion Selection 權重:")
        logger.info(f"  w_cv_auc={cs_cfg.w_cv_auc}, w_monitor_auc={cs_cfg.w_monitor_auc}, "
                     f"w_f1_reject={cs_cfg.w_monitor_f1_reject}, w_ks={cs_cfg.w_monitor_ks}, "
                     f"w_stability_penalty={cs_cfg.w_stability_penalty}")
        
        self.strategy_results = {}
        
        for model_name, cycle_results in self.all_cycle_results.items():
            if not cycle_results:
                continue
            
            cv_aucs = [r.cv_auc_mean for r in cycle_results]
            monitor_aucs = [r.monitor_auc for r in cycle_results]
            monitor_f1s = [r.monitor_f1 for r in cycle_results]
            monitor_f1_rejects = [r.monitor_f1_reject for r in cycle_results]
            monitor_ks = [r.monitor_ks for r in cycle_results]
            monitor_brier = [r.monitor_brier for r in cycle_results]
            
            # Stability: 加入 KS std 使穩定性評估更全面
            stability_score = (
                np.std(cv_aucs) + 
                np.std(monitor_aucs) + 
                np.std(monitor_f1_rejects) + 
                np.std(monitor_ks)
            ) / 4
            
            # Overall score (V2 weights)
            overall_score = (
                np.mean(cv_aucs) * cs_cfg.w_cv_auc +
                np.mean(monitor_aucs) * cs_cfg.w_monitor_auc +
                np.mean(monitor_f1_rejects) * cs_cfg.w_monitor_f1_reject +
                np.mean(monitor_ks) * cs_cfg.w_monitor_ks -
                stability_score * cs_cfg.w_stability_penalty
            )
            
            # Build ranking reason
            score_parts = [
                f"cv_auc={np.mean(cv_aucs):.4f}*{cs_cfg.w_cv_auc}",
                f"mon_auc={np.mean(monitor_aucs):.4f}*{cs_cfg.w_monitor_auc}",
                f"f1_rej={np.mean(monitor_f1_rejects):.4f}*{cs_cfg.w_monitor_f1_reject}",
                f"ks={np.mean(monitor_ks):.4f}*{cs_cfg.w_monitor_ks}",
                f"stab_penalty={stability_score:.4f}*{cs_cfg.w_stability_penalty}",
            ]
            ranking_reason = " + ".join(score_parts[:4]) + " - " + score_parts[4]
            
            strategy = StrategyResult(
                model_name=model_name,
                avg_cv_auc=float(np.mean(cv_aucs)),
                std_cv_auc=float(np.std(cv_aucs)),
                avg_monitor_auc=float(np.mean(monitor_aucs)),
                std_monitor_auc=float(np.std(monitor_aucs)),
                avg_monitor_f1=float(np.mean(monitor_f1s)),
                std_monitor_f1=float(np.std(monitor_f1s)),
                avg_monitor_f1_reject=float(np.mean(monitor_f1_rejects)),
                std_monitor_f1_reject=float(np.std(monitor_f1_rejects)),
                avg_monitor_ks=float(np.mean(monitor_ks)),
                avg_monitor_brier=float(np.mean(monitor_brier)),
                stability_score=float(stability_score),
                overall_score=float(overall_score),
                ranking_reason=ranking_reason,
                cycle_results=cycle_results
            )
            
            self.strategy_results[model_name] = strategy
            
            logger.info(f"\n{model_name}:")
            logger.info(f"  Avg CV AUC: {strategy.avg_cv_auc:.4f} ± {strategy.std_cv_auc:.4f}")
            logger.info(f"  Avg Monitor AUC: {strategy.avg_monitor_auc:.4f} ± {strategy.std_monitor_auc:.4f}")
            logger.info(f"  Avg Monitor F1_reject: {strategy.avg_monitor_f1_reject:.4f} ± {strategy.std_monitor_f1_reject:.4f}")
            logger.info(f"  Avg Monitor KS: {strategy.avg_monitor_ks:.4f}")
            logger.info(f"  Stability Score: {strategy.stability_score:.4f}")
            logger.info(f"  Overall Score: {strategy.overall_score:.4f}")
            logger.info(f"  Scoring: {ranking_reason}")
        
        # 選出最佳
        sorted_strategies = sorted(
            self.strategy_results.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        self.champion_strategy = sorted_strategies[0][0]
        
        logger.info("\n" + "=" * 60)
        logger.info("Champion Strategy Selection (V2 Scoring)")
        logger.info("=" * 60)
        logger.info(f"選出 Champion: {self.champion_strategy}")
        
        champion = self.strategy_results[self.champion_strategy]
        logger.info(f"  Overall Score: {champion.overall_score:.4f}")
        logger.info(f"  Avg Monitor AUC: {champion.avg_monitor_auc:.4f}")
        logger.info(f"  Avg Monitor F1_reject: {champion.avg_monitor_f1_reject:.4f}")
        logger.info(f"  Avg Monitor KS: {champion.avg_monitor_ks:.4f}")
        logger.info(f"  Stability: {champion.stability_score:.4f}")
        logger.info(f"  Ranking Reason: {champion.ranking_reason}")
        
        logger.info("\n策略排名:")
        for i, (name, strategy) in enumerate(sorted_strategies):
            marker = " <- Champion" if i == 0 else ""
            logger.info(f"  {i+1}. {name}: Score={strategy.overall_score:.4f} "
                        f"(f1_rej={strategy.avg_monitor_f1_reject:.4f}, "
                        f"ks={strategy.avg_monitor_ks:.4f}, "
                        f"stab={strategy.stability_score:.4f}){marker}")
        
        return self.champion_strategy
    
    # ============================================
    # Conservative Fine-Tuning
    # ============================================
    def run_conservative_tuning(
        self,
        use_calibration: bool = True
    ) -> Dict:
        """
        保守型 Fine-Tuning
        
        在 Phase 1 完成（champion strategy 已選出）後，
        Phase 2 之前執行。
        
        流程：
        1. 對 xgboost / random_forest 各生成少量候選 config
        2. 每個 config 跑完整 rolling window（重用 Phase 1 的 window 定義）
        3. 每個 config × calibration_method 做一次 Phase 2 retraining
        4. 在 holdout 上評估（但不動 Phase 3/4 的正式流程）
        5. 用 tuning_weights 排名，選出最佳 config
        
        注意：
        - 不破壞 Phase 1 的 strategy_results
        - 最後將 best config 設定為新的 champion
        - tuning 結果存入 self.tuning_results
        """
        tuning_cfg = self.config.tuning if hasattr(self.config, 'tuning') else TuningConfig()
        
        if not tuning_cfg.enable_tuning:
            logger.info("Conservative Tuning 已停用，跳過")
            return {}
        
        logger.info("\n" + "=" * 80)
        logger.info("Conservative Fine-Tuning")
        logger.info("=" * 80)
        logger.info(f"XGBoost 候選: {len(tuning_cfg.xgb_candidates)} 組")
        logger.info(f"Random Forest 候選: {len(tuning_cfg.rf_candidates)} 組")
        logger.info(f"Calibration 方法: {tuning_cfg.calibration_methods}")
        
        # 載入資料
        dev_df = self.load_development_data()
        X_dev, y_dev, feature_cols = self._prepare_xy(dev_df)
        self.feature_names = feature_cols
        
        scale_pos_weight = self.imbalance_handler.calculate_scale_pos_weight(y_dev)
        class_weight = self.imbalance_handler.calculate_class_weight(y_dev)
        
        # 載入 holdout 資料（用於 tuning evaluation，不是 Phase 4 正式 holdout）
        spark = self._get_spark()
        oot_df = spark.read.parquet(str(self.oot_path)).toPandas()
        oot_df['進件日'] = pd.to_datetime(oot_df['進件日'])
        oot_df = oot_df.sort_values('進件日')
        cutoff_idx = int(len(oot_df) * 2 / 3)
        tuning_holdout_df = oot_df.iloc[cutoff_idx:].copy()
        X_holdout, y_holdout, _ = self._prepare_xy(tuning_holdout_df, feature_cols)
        
        logger.info(f"Development 資料: {len(y_dev)} 筆")
        logger.info(f"Tuning Holdout 資料: {len(y_holdout)} 筆")
        
        tuning_records = []
        
        # ── XGBoost 候選 ──
        for xgb_cfg in tuning_cfg.xgb_candidates:
            config_id = xgb_cfg.get("config_id", "xgb_unknown")
            logger.info(f"\n--- Tuning: {config_id} ---")
            
            for cal_method in tuning_cfg.calibration_methods:
                label = f"{config_id}_{cal_method}"
                logger.info(f"  Calibration: {cal_method}")
                
                try:
                    record = self._evaluate_tuning_candidate(
                        model_type="xgboost",
                        config_id=config_id,
                        params=xgb_cfg,
                        calibration_method=cal_method,
                        X_dev=X_dev, y_dev=y_dev,
                        X_holdout=X_holdout, y_holdout=y_holdout,
                        dev_df=dev_df,
                        scale_pos_weight=scale_pos_weight,
                        class_weight=class_weight,
                        feature_cols=feature_cols,
                        use_calibration=(cal_method != "none"),
                    )
                    tuning_records.append(record)
                    logger.info(f"    holdout_auc={record['holdout_auc']:.4f}, "
                                f"holdout_f1_reject={record['holdout_f1_reject']:.4f}, "
                                f"holdout_brier={record['holdout_brier']:.4f}")
                except Exception as e:
                    logger.warning(f"    WARNING: {label} 失敗: {e}")
        
        # ── Random Forest 候選 ──
        for rf_cfg in tuning_cfg.rf_candidates:
            config_id = rf_cfg.get("config_id", "rf_unknown")
            logger.info(f"\n--- Tuning: {config_id} ---")
            
            for cal_method in tuning_cfg.calibration_methods:
                label = f"{config_id}_{cal_method}"
                logger.info(f"  Calibration: {cal_method}")
                
                try:
                    record = self._evaluate_tuning_candidate(
                        model_type="random_forest",
                        config_id=config_id,
                        params=rf_cfg,
                        calibration_method=cal_method,
                        X_dev=X_dev, y_dev=y_dev,
                        X_holdout=X_holdout, y_holdout=y_holdout,
                        dev_df=dev_df,
                        scale_pos_weight=scale_pos_weight,
                        class_weight=class_weight,
                        feature_cols=feature_cols,
                        use_calibration=(cal_method != "none"),
                    )
                    tuning_records.append(record)
                    logger.info(f"    holdout_auc={record['holdout_auc']:.4f}, "
                                f"holdout_f1_reject={record['holdout_f1_reject']:.4f}, "
                                f"holdout_brier={record['holdout_brier']:.4f}")
                except Exception as e:
                    logger.warning(f"    WARNING: {label} 失敗: {e}")
        
        if not tuning_records:
            logger.warning("所有 tuning 候選皆失敗，保持原 champion")
            return {}
        
        # ── Scoring & Ranking ──
        tw = tuning_cfg.tuning_weights
        
        for rec in tuning_records:
            # Brier 越低越好 -> 取反 (1 - brier) 作為分數
            score = (
                tw.get("w_holdout_f1_reject", 0.25) * rec["holdout_f1_reject"] +
                tw.get("w_monitor_f1_reject", 0.15) * rec["avg_monitor_f1_reject"] +
                tw.get("w_holdout_brier", 0.15) * (1.0 - rec["holdout_brier"]) +
                tw.get("w_monitor_brier", 0.10) * (1.0 - rec["avg_monitor_brier"]) +
                tw.get("w_holdout_auc", 0.10) * rec["holdout_auc"] +
                tw.get("w_monitor_auc", 0.10) * rec["avg_monitor_auc"] -
                tw.get("w_stability_penalty", 0.15) * rec["stability_score"]
            )
            rec["tuning_score"] = float(score)
        
        tuning_records.sort(key=lambda x: x["tuning_score"], reverse=True)
        
        # 輸出比較
        logger.info("\n" + "=" * 60)
        logger.info("Conservative Tuning 結果排名")
        logger.info("=" * 60)
        for i, rec in enumerate(tuning_records):
            marker = " <- BEST" if i == 0 else ""
            logger.info(
                f"  {i+1}. {rec['model_name']}/{rec['config_id']}/"
                f"{rec['calibration_method']} "
                f"Score={rec['tuning_score']:.4f} "
                f"(h_f1r={rec['holdout_f1_reject']:.4f}, "
                f"h_brier={rec['holdout_brier']:.4f}, "
                f"h_auc={rec['holdout_auc']:.4f}, "
                f"m_f1r={rec['avg_monitor_f1_reject']:.4f}, "
                f"stab={rec['stability_score']:.4f}){marker}"
            )
        
        best = tuning_records[0]
        
        logger.info(f"\nOK: Tuning Champion: {best['model_name']} / {best['config_id']} / {best['calibration_method']}")
        logger.info(f"  Why: tuning_score={best['tuning_score']:.4f}")
        logger.info(f"    holdout_f1_reject={best['holdout_f1_reject']:.4f}")
        logger.info(f"    holdout_brier={best['holdout_brier']:.4f}")
        logger.info(f"    holdout_auc={best['holdout_auc']:.4f}")
        logger.info(f"    avg_monitor_f1_reject={best['avg_monitor_f1_reject']:.4f}")
        logger.info(f"    stability_score={best['stability_score']:.4f}")
        
        # NOTE: 更新 champion_strategy 和 tuning state
        self.champion_strategy = best["model_name"]
        self.tuning_results = tuning_records
        self.tuning_best = best
        
        return {
            "best": best,
            "all_records": tuning_records,
        }
    
    def _evaluate_tuning_candidate(
        self,
        model_type: str,
        config_id: str,
        params: Dict,
        calibration_method: str,
        X_dev: np.ndarray, y_dev: np.ndarray,
        X_holdout: np.ndarray, y_holdout: np.ndarray,
        dev_df: pd.DataFrame,
        scale_pos_weight: float,
        class_weight: Dict,
        feature_cols: List[str],
        use_calibration: bool = True,
    ) -> Dict:
        """
        評估單一 tuning candidate
        
        1. 在 rolling windows 上做 quick CV (用 Phase 1 同樣的 windows)
        2. 在完整 dev 上 retrain + calibrate
        3. 在 holdout 上評估
        """
        # ── 1. 建立模型 ──
        if model_type == "xgboost":
            base_model = xgb.XGBClassifier(
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", 3),
                learning_rate=params.get("learning_rate", 0.03),
                min_child_weight=params.get("min_child_weight", 20),
                subsample=params.get("subsample", 0.7),
                colsample_bytree=params.get("colsample_bytree", 0.7),
                reg_alpha=params.get("reg_alpha", 1.0),
                reg_lambda=params.get("reg_lambda", 5.0),
                gamma=params.get("gamma", 1.0),
                max_delta_step=params.get("max_delta_step", 1),
                scale_pos_weight=scale_pos_weight,
                objective='binary:logistic',
                eval_metric='auc',
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        elif model_type == "random_forest":
            base_model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", 6),
                min_samples_split=params.get("min_samples_split", 50),
                min_samples_leaf=params.get("min_samples_leaf", 20),
                max_features=params.get("max_features", "sqrt"),
                class_weight=class_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # ── 2. Rolling window quick evaluation ──
        monitor_aucs, monitor_f1_rejects, monitor_ks_list = [], [], []
        monitor_briers = []
        
        for window in self.window_definitions:
            try:
                train_df = self.load_development_data(
                    start_date=window.train_start,
                    end_date=window.train_end
                )
                monitor_df = self.load_development_data(
                    start_date=window.monitor_start,
                    end_date=window.monitor_end
                )
                
                if len(train_df) == 0 or len(monitor_df) == 0:
                    continue
                
                X_train, y_train, _ = self._prepare_xy(train_df, feature_cols)
                X_monitor, y_monitor, _ = self._prepare_xy(monitor_df, feature_cols)
                
                model = deepcopy(base_model)
                
                if model_type == "xgboost":
                    model.set_params(early_stopping_rounds=30)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_monitor, y_monitor)],
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
                
                y_pred = model.predict_proba(X_monitor)[:, 1]
                metrics = self.metrics_calc.calculate_all_metrics(y_monitor, y_pred)
                
                monitor_aucs.append(metrics['auc'])
                monitor_f1_rejects.append(metrics['f1_reject'])
                monitor_ks_list.append(metrics['ks'])
                monitor_briers.append(metrics['brier_score'])
            except Exception:
                continue
        
        if not monitor_aucs:
            raise ValueError(f"No valid rolling windows for {config_id}")
        
        stability_score = float((
            np.std(monitor_aucs) + 
            np.std(monitor_f1_rejects) + 
            np.std(monitor_ks_list)
        ) / 3)
        
        # ── 3. Full retrain + calibrate on dev ──
        full_model = deepcopy(base_model)
        
        if use_calibration and calibration_method != "none":
            cal_model = CalibratedClassifierCV(
                estimator=full_model,
                method=calibration_method,
                cv=5,
                n_jobs=-1
            )
            cal_model.fit(X_dev, y_dev)
            final_model = cal_model
        else:
            if model_type == "xgboost":
                # Split dev for early stopping
                from sklearn.model_selection import train_test_split as tts
                X_tr, X_es, y_tr, y_es = tts(
                    X_dev, y_dev, test_size=0.15, random_state=RANDOM_STATE, stratify=y_dev
                )
                full_model.set_params(early_stopping_rounds=30)
                full_model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
            else:
                full_model.fit(X_dev, y_dev)
            final_model = full_model
        
        # ── 4. Holdout evaluation ──
        y_holdout_pred = final_model.predict_proba(X_holdout)[:, 1]
        holdout_metrics = self.metrics_calc.calculate_all_metrics(y_holdout, y_holdout_pred)
        
        return {
            "model_name": model_type,
            "config_id": config_id,
            "calibration_method": calibration_method,
            "params": {k: v for k, v in params.items() if k != "config_id"},
            # Rolling monitor averages
            "avg_monitor_auc": float(np.mean(monitor_aucs)),
            "avg_monitor_f1_reject": float(np.mean(monitor_f1_rejects)),
            "avg_monitor_ks": float(np.mean(monitor_ks_list)),
            "avg_monitor_brier": float(np.mean(monitor_briers)),
            "std_monitor_auc": float(np.std(monitor_aucs)),
            "std_monitor_f1_reject": float(np.std(monitor_f1_rejects)),
            # Holdout
            "holdout_auc": holdout_metrics['auc'],
            "holdout_f1_reject": holdout_metrics['f1_reject'],
            "holdout_ks": holdout_metrics['ks'],
            "holdout_brier": holdout_metrics['brier_score'],
            # Stability
            "stability_score": stability_score,
            # Meta
            "selected_as_champion": False,  # 後續更新
        }

    # ============================================
    # Phase 2: Champion Retraining
    # ============================================
    def run_phase2_champion_retraining(
        self,
        use_calibration: bool = True,
        calibration_method: str = "isotonic"
    ) -> Tuple[Any, Dict]:
        """
        Phase 2: Champion Retraining
        
        使用完整 18 個月 development dataset
        用選出的 Champion Strategy 重訓 -> 產出 Final Champion Artifact
        
        此 Final Champion Artifact 才是後續要用的模型：
        - Phase 3 Policy Validation（threshold / zone policy tuning）
        - Phase 4 Final Blind Holdout（untouched evaluation）
        - Production Batch Scoring（上線後的批次推論）
        """
        if not self.champion_strategy:
            self.select_champion_strategy()
        
        logger.info("\n" + "=" * 80)
        logger.info("Phase 2: Champion Retraining")
        logger.info("=" * 80)
        logger.info(f"Champion Strategy: {self.champion_strategy}")
        
        # 載入完整 development 資料
        dev_df = self.load_development_data()
        X, y, feature_cols = self._prepare_xy(dev_df)
        
        self.feature_names = feature_cols
        positive_ratio = np.mean(y)
        
        logger.info(f"Development 資料: {len(y)} 筆")
        logger.info(f"特徵數: {len(feature_cols)}")
        logger.info(f"正樣本比例: {positive_ratio:.2%}")
        logger.info(f"Imbalance Strategy: {self.imbalance_handler.strategy}")
        
        # 取得模型 — 若有 tuning 結果，使用 tuning best 的 params
        scale_pos_weight = self.imbalance_handler.calculate_scale_pos_weight(y)
        class_weight = self.imbalance_handler.calculate_class_weight(y)
        
        tuning_best = getattr(self, 'tuning_best', None)
        
        if tuning_best is not None:
            # NOTE: 使用 tuning 選出的最佳參數
            logger.info(f"  使用 Tuning Best Config: {tuning_best['config_id']}")
            best_params = tuning_best.get("params", {})
            calibration_method = tuning_best.get("calibration_method", calibration_method)
            use_calibration = (calibration_method != "none")
            
            if self.champion_strategy == "xgboost":
                base_model = xgb.XGBClassifier(
                    n_estimators=best_params.get("n_estimators", 300),
                    max_depth=best_params.get("max_depth", 3),
                    learning_rate=best_params.get("learning_rate", 0.03),
                    min_child_weight=best_params.get("min_child_weight", 20),
                    subsample=best_params.get("subsample", 0.7),
                    colsample_bytree=best_params.get("colsample_bytree", 0.7),
                    reg_alpha=best_params.get("reg_alpha", 1.0),
                    reg_lambda=best_params.get("reg_lambda", 5.0),
                    gamma=best_params.get("gamma", 1.0),
                    max_delta_step=best_params.get("max_delta_step", 1),
                    scale_pos_weight=scale_pos_weight,
                    objective='binary:logistic',
                    eval_metric='auc',
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            elif self.champion_strategy == "random_forest":
                base_model = RandomForestClassifier(
                    n_estimators=best_params.get("n_estimators", 300),
                    max_depth=best_params.get("max_depth", 6),
                    min_samples_split=best_params.get("min_samples_split", 50),
                    min_samples_leaf=best_params.get("min_samples_leaf", 20),
                    max_features=best_params.get("max_features", "sqrt"),
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            else:
                base_model = CandidateModels.get_logistic_regression(class_weight)
            
            logger.info(f"  Calibration Method: {calibration_method}")
        else:
            if self.champion_strategy == "xgboost":
                base_model = CandidateModels.get_xgboost(scale_pos_weight)
            elif self.champion_strategy == "random_forest":
                base_model = CandidateModels.get_random_forest(class_weight)
            else:
                base_model = CandidateModels.get_logistic_regression(class_weight)
        
        # 訓練
        if use_calibration:
            logger.info(f"使用 CalibratedClassifierCV ({calibration_method})")
            self.final_champion_model = CalibratedClassifierCV(
                estimator=base_model,
                method=calibration_method,
                cv=5,
                n_jobs=-1
            )
            self.final_champion_model.fit(X, y)
        else:
            base_model.fit(X, y)
            self.final_champion_model = base_model
        
        logger.info("OK: Final Champion Model 訓練完成")
        
        # In-sample 評估
        y_pred_proba = self.final_champion_model.predict_proba(X)[:, 1]
        in_sample_metrics = self.metrics_calc.calculate_all_metrics(y, y_pred_proba)
        
        logger.info(f"\nIn-Sample (Development) Metrics:")
        logger.info(f"  AUC: {in_sample_metrics['auc']:.4f}")
        logger.info(f"  F1: {in_sample_metrics['f1']:.4f}")
        logger.info(f"  F1_reject: {in_sample_metrics['f1_reject']:.4f}")
        logger.info(f"  KS: {in_sample_metrics['ks']:.4f}")
        logger.info(f"  Brier: {in_sample_metrics['brier_score']:.4f}")
        
        # 更新 diagnostics
        if self.diagnostics is None:
            self.diagnostics = DiagnosticsSummary()
        
        self.diagnostics.train_auc = in_sample_metrics['auc']
        self.diagnostics.train_f1_reject = in_sample_metrics['f1_reject']
        self.diagnostics.train_ks = in_sample_metrics['ks']
        self.diagnostics.train_brier = in_sample_metrics['brier_score']
        
        # 從 rolling / tuning 取得 monitor 平均
        if tuning_best is not None:
            self.diagnostics.avg_monitor_auc = tuning_best.get("avg_monitor_auc", 0.0)
            self.diagnostics.avg_monitor_f1_reject = tuning_best.get("avg_monitor_f1_reject", 0.0)
            self.diagnostics.avg_monitor_ks = tuning_best.get("avg_monitor_ks", 0.0)
            self.diagnostics.avg_monitor_brier = tuning_best.get("avg_monitor_brier", 0.0)
            self.diagnostics.std_monitor_auc = tuning_best.get("std_monitor_auc", 0.0)
        elif self.champion_strategy in self.strategy_results:
            strat = self.strategy_results[self.champion_strategy]
            self.diagnostics.avg_monitor_auc = strat.avg_monitor_auc
            self.diagnostics.avg_monitor_f1_reject = strat.avg_monitor_f1_reject
            self.diagnostics.avg_monitor_ks = strat.avg_monitor_ks
            self.diagnostics.avg_monitor_brier = strat.avg_monitor_brier
            self.diagnostics.std_monitor_auc = strat.std_monitor_auc
        
        return self.final_champion_model, in_sample_metrics
    
    # ============================================
    # Phase 3: Policy Validation
    # ============================================
    def run_phase3_policy_validation(
        self,
        policy_val_df: pd.DataFrame = None,
        lower_thresholds: List[float] = None,
        upper_thresholds: List[float] = None
    ) -> Tuple[pd.DataFrame, List[ThresholdPolicyResult], Dict]:
        """
        Phase 3: Policy Validation
        
        不再調整模型權重！
        只做 Threshold / Zone Policy Tuning
        
        輸出：
        - policy_validation_predictions.csv（也作為 Production Monitoring 的 baseline）
        - threshold_policy_comparison.csv
        """
        if self.final_champion_model is None:
            raise ValueError("請先執行 Phase 2 Champion Retraining")
        
        logger.info("\n" + "=" * 80)
        logger.info("Phase 3: Policy Validation")
        logger.info("=" * 80)
        logger.info("注意：此階段不再調整模型權重，只做 threshold 評估")
        
        # 載入 Policy Validation 資料
        if policy_val_df is None:
            # 從 OOT (Gold Layer legacy storage) 切分：前 2/3 -> policy_validation，後 1/3 -> final_holdout
            spark = self._get_spark()
            oot_df = spark.read.parquet(str(self.oot_path)).toPandas()
            oot_df['進件日'] = pd.to_datetime(oot_df['進件日'])
            oot_df = oot_df.sort_values('進件日')
            
            # 前 2/3 作為 Policy Validation (Phase 3)
            cutoff_idx = int(len(oot_df) * 2 / 3)
            policy_val_df = oot_df.iloc[:cutoff_idx].copy()
            # 後 1/3 作為 Final Blind Holdout (Phase 4)
            self.final_holdout_df = oot_df.iloc[cutoff_idx:].copy()
            
            logger.info(f"OOT (legacy storage) 切分完成:")
            logger.info(f"  Policy Validation (前 2/3): {len(policy_val_df)} 筆")
            logger.info(f"  Final Blind Holdout (後 1/3): {len(self.final_holdout_df)} 筆")
        
        self.policy_validation_df = policy_val_df
        
        X, y, _ = self._prepare_xy(policy_val_df, self.feature_names)
        
        logger.info(f"Policy Validation 資料: {len(y)} 筆")
        logger.info(f"正樣本比例: {np.mean(y):.2%}")
        
        # 使用 Final Champion Artifact 預測
        y_pred_proba = self.final_champion_model.predict_proba(X)[:, 1]
        
        # 計算指標
        pv_metrics = self.metrics_calc.calculate_all_metrics(y, y_pred_proba)
        
        logger.info(f"\nPolicy Validation Metrics:")
        logger.info(f"  AUC: {pv_metrics['auc']:.4f}")
        logger.info(f"  F1_reject: {pv_metrics['f1_reject']:.4f}")
        logger.info(f"  KS: {pv_metrics['ks']:.4f}")
        
        # 更新 diagnostics
        self.diagnostics.policy_val_auc = pv_metrics['auc']
        self.diagnostics.policy_val_f1_reject = pv_metrics['f1_reject']
        self.diagnostics.policy_val_ks = pv_metrics['ks']
        
        # Threshold Grid Evaluation
        logger.info("\n評估 Threshold 組合...")
        threshold_results = evaluate_threshold_grid(
            y, y_pred_proba,
            lower_thresholds=lower_thresholds,
            upper_thresholds=upper_thresholds
        )
        
        logger.info(f"評估了 {len(threshold_results)} 組 threshold 組合")
        
        # ── Business Constraint Filtering + Composite Scoring (V2) ──
        bc_cfg = self.config.business_constraints if hasattr(self.config, 'business_constraints') else BusinessConstraintConfig()
        
        logger.info(f"\nBusiness Constraints:")
        logger.info(f"  max_manual_review_ratio: {bc_cfg.max_manual_review_ratio:.2%}")
        logger.info(f"  min_auto_decision_rate: {bc_cfg.min_auto_decision_rate:.2%}")
        logger.info(f"  min_low_zone_ratio: {bc_cfg.min_low_zone_ratio:.2%}")
        logger.info(f"  target_auto_decision_rate: {bc_cfg.target_auto_decision_rate:.2%}")
        
        scored_results = score_threshold_policy(threshold_results, bc_cfg)
        
        # 統計約束過濾結果
        n_pass = sum(1 for r in scored_results if r.passes_hard_constraints)
        n_fail = sum(1 for r in scored_results if not r.passes_hard_constraints)
        logger.info(f"\n約束過濾結果: {n_pass} 組通過 / {n_fail} 組不通過")
        
        # 選出最佳 threshold
        passing_results = [r for r in scored_results if r.passes_hard_constraints]
        
        if passing_results:
            best = passing_results[0]  # score_threshold_policy 已排序
            
            # NOTE: 保存推薦 threshold 到 class attribute，供 Phase 4 / Monitoring 使用
            self.selected_lower_threshold = best.lower_threshold
            self.selected_upper_threshold = best.upper_threshold
            
            logger.info(f"\nOK: Phase 3 推薦 Threshold（已保存至 trainer state）:")
            logger.info(f"  Lower: {self.selected_lower_threshold}")
            logger.info(f"  Upper: {self.selected_upper_threshold}")
            logger.info(f"  Threshold Score: {best.threshold_score:.4f}")
            logger.info(f"  Auto Decision Rate: {best.auto_decision_rate:.2%}")
            logger.info(f"  Manual Review Load: {best.manual_review_load:.2%}")
            logger.info(f"  High Zone Precision: {best.expected_precision_high:.2%}")
            logger.info(f"  Low Zone Precision: {best.expected_precision_low:.2%}")
            logger.info(f"  High Zone Ratio: {best.high_zone_ratio:.2%}")
            logger.info(f"  Low Zone Ratio: {best.low_zone_ratio:.2%}")
            
            # 顯示 Top-3 候選
            logger.info(f"\n  Top-3 候選:")
            for i, r in enumerate(passing_results[:3]):
                logger.info(f"    {i+1}. lower={r.lower_threshold}, upper={r.upper_threshold} "
                            f"-> score={r.threshold_score:.4f}, "
                            f"auto={r.auto_decision_rate:.2%}, "
                            f"manual={r.manual_review_load:.2%}, "
                            f"high_prec={r.expected_precision_high:.2%}")
        else:
            # Fallback: 放寬約束，從所有結果中選 score 最高的
            logger.warning("WARNING: 無組合通過 hard constraints，放寬至所有結果中選最佳")
            if scored_results:
                best = scored_results[0]
                self.selected_lower_threshold = best.lower_threshold
                self.selected_upper_threshold = best.upper_threshold
                logger.warning(f"  Fallback 選擇: lower={best.lower_threshold}, upper={best.upper_threshold}")
                logger.warning(f"  Violations: {best.constraint_violations}")
            else:
                logger.warning("WARNING: Phase 3 無法找到任何 threshold 組合，將使用預設值")
        
        # 建立 predictions DataFrame
        predictions_df = policy_val_df[['案件編號', '進件日', '授信結果_二元']].copy()
        predictions_df['pred_prob'] = y_pred_proba
        predictions_df['actual_label'] = y
        
        # NOTE: Phase 3 也輸出 zone assignment（使用推薦 threshold 或 fallback）
        pv_lower = self.selected_lower_threshold if self.selected_lower_threshold is not None else 0.5
        pv_upper = self.selected_upper_threshold if self.selected_upper_threshold is not None else 0.85
        predictions_df['pred_zone'] = assign_score_zone(y_pred_proba, pv_lower, pv_upper)
        predictions_df['zone_name'] = predictions_df['pred_zone'].map({
            2: '高通過機率區', 1: '人工審核區', 0: '低通過機率區'
        })
        predictions_df['lower_threshold_used'] = pv_lower
        predictions_df['upper_threshold_used'] = pv_upper
        predictions_df['threshold_source'] = 'phase3_recommended' if self.selected_lower_threshold is not None else 'default'
        
        return predictions_df, threshold_results, pv_metrics
    
    # ============================================
    # Phase 4: Final Blind Holdout
    # ============================================
    def run_phase4_final_holdout(
        self,
        holdout_df: pd.DataFrame = None,
        lower_threshold: float = 0.5,
        upper_threshold: float = 0.85
    ) -> Tuple[pd.DataFrame, Dict, List[ZoneSummary]]:
        """
        Phase 4: Final Blind Holdout Evaluation
        
        完全不再調模型、不再調 threshold
        真正的 untouched final evaluation
        
        Threshold 優先順序：
        1. Phase 3 推薦值 (self.selected_lower/upper_threshold)
        2. 函式參數 (lower_threshold / upper_threshold)
        
        WARNING: 這是有真實 label 的 Final Blind Holdout evaluation
        WARNING: 不是 Production Batch Scoring（production 沒有即時 label，只輸出 predictions）
        """
        if self.final_champion_model is None:
            raise ValueError("請先執行 Phase 2 Champion Retraining")
        
        # NOTE: 優先使用 Phase 3 推薦的 threshold
        if self.selected_lower_threshold is not None:
            lower_threshold = self.selected_lower_threshold
        if self.selected_upper_threshold is not None:
            upper_threshold = self.selected_upper_threshold
        
        logger.info("\n" + "=" * 80)
        logger.info("Phase 4: Final Blind Holdout Evaluation")
        logger.info("=" * 80)
        if self.selected_lower_threshold is not None:
            logger.info(f"NOTE: 使用 Phase 3 推薦 Threshold: lower={lower_threshold}, upper={upper_threshold}")
        else:
            logger.info(f"WARNING: 未找到 Phase 3 推薦 Threshold，使用預設值: lower={lower_threshold}, upper={upper_threshold}")
        logger.info("注意：此階段完全不調模型、不調 threshold")
        logger.info("這是最終的 untouched Final Blind Holdout（有真實 label，可算指標）")
        
        if holdout_df is None:
            holdout_df = self.final_holdout_df
        
        if holdout_df is None or len(holdout_df) == 0:
            # 從 OOT (Gold Layer legacy storage) 取後 1/3 作為 Final Blind Holdout
            spark = self._get_spark()
            oot_df = spark.read.parquet(str(self.oot_path)).toPandas()
            oot_df['進件日'] = pd.to_datetime(oot_df['進件日'])
            oot_df = oot_df.sort_values('進件日')
            
            cutoff_idx = int(len(oot_df) * 2 / 3)
            holdout_df = oot_df.iloc[cutoff_idx:].copy()
            logger.info(f"Final Blind Holdout 從 OOT (legacy storage) 後 1/3 載入: {len(holdout_df)} 筆")
        
        X, y, _ = self._prepare_xy(holdout_df, self.feature_names)
        
        logger.info(f"Final Blind Holdout 資料: {len(y)} 筆")
        logger.info(f"正樣本比例: {np.mean(y):.2%}")
        
        # 使用 Final Champion Artifact 預測（此階段有真實 label，可算指標）
        y_pred_proba = self.final_champion_model.predict_proba(X)[:, 1]
        
        # 計算指標
        holdout_metrics = self.metrics_calc.calculate_all_metrics(y, y_pred_proba)
        
        logger.info(f"\nFinal Holdout Metrics:")
        logger.info(f"  AUC: {holdout_metrics['auc']:.4f}")
        logger.info(f"  F1: {holdout_metrics['f1']:.4f}")
        logger.info(f"  F1_reject: {holdout_metrics['f1_reject']:.4f}")
        logger.info(f"  KS: {holdout_metrics['ks']:.4f}")
        logger.info(f"  Brier Score: {holdout_metrics['brier_score']:.4f}")
        
        logger.info(f"\n  Confusion Matrix:")
        logger.info(f"    TP={holdout_metrics['true_positive']}, FP={holdout_metrics['false_positive']}")
        logger.info(f"    FN={holdout_metrics['false_negative']}, TN={holdout_metrics['true_negative']}")
        
        # 更新 diagnostics
        self.diagnostics.final_holdout_auc = holdout_metrics['auc']
        self.diagnostics.final_holdout_f1_reject = holdout_metrics['f1_reject']
        self.diagnostics.final_holdout_ks = holdout_metrics['ks']
        self.diagnostics.final_holdout_brier = holdout_metrics['brier_score']
        self.diagnostics.compute_gaps()
        
        # 三區間分析
        logger.info(f"\n三區間分析 (lower={lower_threshold}, upper={upper_threshold}):")
        zone_summaries = evaluate_zone_performance(y, y_pred_proba, lower_threshold, upper_threshold)
        
        for zone in zone_summaries:
            logger.info(f"\n  {zone.zone_name}:")
            logger.info(f"    筆數: {zone.count} ({zone.ratio:.2%})")
            logger.info(f"    平均機率: {zone.avg_prob:.4f}")
            logger.info(f"    實際核准率: {zone.actual_approve_rate:.2%}")
            logger.info(f"    實際婉拒率: {zone.actual_reject_rate:.2%}")
        
        # Decile 分析
        decile_summaries = self.metrics_calc.calculate_decile_summary(y, y_pred_proba)
        
        logger.info(f"\nScore Band / Decile 分析:")
        for d in decile_summaries:
            logger.info(f"  Decile {d.decile}: "
                       f"Predicted={d.predicted_avg_prob:.3f}, "
                       f"Actual={d.actual_approve_rate:.3f}, "
                       f"Gap={d.calibration_gap:+.3f}")
        
        # 建立 predictions DataFrame
        predictions_df = holdout_df[['案件編號', '進件日', '授信結果_二元']].copy()
        predictions_df['pred_prob'] = y_pred_proba
        predictions_df['pred_zone'] = assign_score_zone(y_pred_proba, lower_threshold, upper_threshold)
        predictions_df['zone_name'] = predictions_df['pred_zone'].map({
            2: '高通過機率區', 1: '人工審核區', 0: '低通過機率區'
        })
        predictions_df['actual_label'] = y
        predictions_df['lower_threshold_used'] = lower_threshold
        predictions_df['upper_threshold_used'] = upper_threshold
        predictions_df['threshold_source'] = 'phase3_recommended' if self.selected_lower_threshold is not None else 'fallback_default'
        
        return predictions_df, holdout_metrics, zone_summaries
    
    # ============================================
    # Diagnostics
    # ============================================
    def compare_train_monitor_holdout_gap(self) -> DiagnosticsSummary:
        """
        比較 Train / Monitor / Holdout 三階段的指標差距
        
        用於檢測 Overfitting
        """
        if self.diagnostics is None:
            self.diagnostics = DiagnosticsSummary()
        
        self.diagnostics.compute_gaps()
        
        logger.info("\n" + "=" * 60)
        logger.info("Overfitting / Robustness Diagnostics")
        logger.info("=" * 60)
        
        logger.info(f"\nAUC Comparison:")
        logger.info(f"  Train (In-Sample): {self.diagnostics.train_auc:.4f}")
        logger.info(f"  Avg Monitor: {self.diagnostics.avg_monitor_auc:.4f} ± {self.diagnostics.std_monitor_auc:.4f}")
        logger.info(f"  Policy Validation: {self.diagnostics.policy_val_auc:.4f}")
        logger.info(f"  Final Holdout: {self.diagnostics.final_holdout_auc:.4f}")
        
        logger.info(f"\nGap Analysis:")
        logger.info(f"  Train vs Monitor AUC: {self.diagnostics.gap_train_vs_monitor_auc:+.4f}")
        logger.info(f"  Train vs Holdout AUC: {self.diagnostics.gap_train_vs_holdout_auc:+.4f}")
        logger.info(f"  Monitor vs Holdout AUC: {self.diagnostics.gap_monitor_vs_holdout_auc:+.4f}")
        logger.info(f"  Train vs Monitor Brier: {self.diagnostics.gap_train_vs_monitor_brier:+.4f}")
        
        logger.info(f"\nOverfitting Assessment:")
        logger.info(f"  Is Overfitting: {self.diagnostics.is_overfitting}")
        logger.info(f"  Severity: {self.diagnostics.overfitting_severity}")
        
        logger.info(f"\nCalibration & Reject Detection:")
        logger.info(f"  Has Calibration Issue: {self.diagnostics.has_calibration_issue}")
        logger.info(f"    (Train Brier={self.diagnostics.train_brier:.4f}, Monitor Brier={self.diagnostics.avg_monitor_brier:.4f})")
        logger.info(f"  Has Reject Detection Issue: {self.diagnostics.has_reject_detection_issue}")
        logger.info(f"    (Holdout F1_reject={self.diagnostics.final_holdout_f1_reject:.4f})")
        
        return self.diagnostics
    
    # ============================================
    # 輸出
    # ============================================
    def save_all_results(
        self,
        run_id: str = None,
        policy_predictions_df: pd.DataFrame = None,
        threshold_results: List[ThresholdPolicyResult] = None,
        holdout_predictions_df: pd.DataFrame = None,
        holdout_metrics: Dict = None,
        zone_summaries: List[ZoneSummary] = None,
        lower_threshold: float = 0.5,
        upper_threshold: float = 0.85
    ) -> Path:
        """儲存所有結果"""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = self.output_path / f"four_phase_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n儲存結果至: {output_dir}")
        
        # 1. rolling_results.csv
        rolling_records = []
        for model_name, results in self.all_cycle_results.items():
            for r in results:
                rolling_records.append(r.to_dict())
        
        if rolling_records:
            rolling_df = pd.DataFrame(rolling_records)
            rolling_df.to_csv(output_dir / "rolling_results.csv", index=False, encoding='utf-8-sig')
            logger.info("  OK: rolling_results.csv")
        
        # 2. champion_summary.json
        cs_cfg = self.config.champion_selection if hasattr(self.config, 'champion_selection') else ChampionSelectionConfig()
        bc_cfg = self.config.business_constraints if hasattr(self.config, 'business_constraints') else BusinessConstraintConfig()
        
        champion_data = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "champion_strategy": {
                "description": "模型+參數的組合方案（由 Phase 1 Rolling Training 選出）",
                "model_name": self.champion_strategy,
                "imbalance_strategy": self.imbalance_handler.strategy,
            },
            "final_champion_artifact": {
                "description": "用 champion strategy 在完整 development data 上訓練的實際模型檔案",
                "model_file": "final_champion_model.pkl",
                "feature_file": "feature_names.json",
            },
            "threshold_config": {
                "description": "Phase 4 / Production Monitoring 實際使用的 threshold",
                "lower_threshold": lower_threshold,
                "upper_threshold": upper_threshold,
                "source": "phase3_recommended" if self.selected_lower_threshold is not None else "default",
            },
            "champion_selection_config": cs_cfg.to_dict(),
            "business_constraints_config": bc_cfg.to_dict(),
            "strategy_results": {
                name: {k: v for k, v in strategy.to_dict().items() if k != 'cycle_results'}
                for name, strategy in self.strategy_results.items()
            }
        }
        
        # NOTE: 如果有 tuning 結果，加入 tuning metadata
        tuning_best = getattr(self, 'tuning_best', None)
        if tuning_best is not None:
            champion_data["tuning_config"] = {
                "config_id": tuning_best.get("config_id", ""),
                "calibration_method": tuning_best.get("calibration_method", ""),
                "params": tuning_best.get("params", {}),
                "tuning_score": tuning_best.get("tuning_score", 0.0),
                "why_selected": (
                    f"Tuning score={tuning_best.get('tuning_score', 0):.4f}: "
                    f"holdout_f1_reject={tuning_best.get('holdout_f1_reject', 0):.4f}, "
                    f"holdout_brier={tuning_best.get('holdout_brier', 0):.4f}, "
                    f"holdout_auc={tuning_best.get('holdout_auc', 0):.4f}, "
                    f"stability={tuning_best.get('stability_score', 0):.4f}"
                ),
            }
            champion_data["rejection_detection_score"] = tuning_best.get("holdout_f1_reject", 0.0)
            champion_data["calibration_score"] = 1.0 - tuning_best.get("holdout_brier", 0.0)
        
        with open(output_dir / "champion_summary.json", 'w', encoding='utf-8') as f:
            json.dump(champion_data, f, ensure_ascii=False, indent=2)
        logger.info("  OK: champion_summary.json")
        
        # 2b. tuning_comparison.csv
        tuning_results = getattr(self, 'tuning_results', None)
        if tuning_results:
            # Mark the best as selected
            if tuning_best:
                for rec in tuning_results:
                    rec["selected_as_champion"] = (
                        rec.get("config_id") == tuning_best.get("config_id") and
                        rec.get("calibration_method") == tuning_best.get("calibration_method") and
                        rec.get("model_name") == tuning_best.get("model_name")
                    )
            
            tuning_df = pd.DataFrame(tuning_results)
            # Reorder columns for readability
            priority_cols = [
                "model_name", "config_id", "calibration_method",
                "tuning_score",
                "avg_monitor_auc", "avg_monitor_f1_reject",
                "avg_monitor_ks", "avg_monitor_brier",
                "holdout_auc", "holdout_f1_reject",
                "holdout_ks", "holdout_brier",
                "stability_score", "selected_as_champion",
            ]
            existing_priority = [c for c in priority_cols if c in tuning_df.columns]
            other_cols = [c for c in tuning_df.columns if c not in existing_priority]
            tuning_df = tuning_df[existing_priority + other_cols]
            tuning_df.to_csv(
                output_dir / "tuning_comparison.csv",
                index=False, encoding='utf-8-sig'
            )
            logger.info("  OK: tuning_comparison.csv")
        
        # 3. policy_validation_predictions.csv
        if policy_predictions_df is not None:
            policy_predictions_df.to_csv(
                output_dir / "policy_validation_predictions.csv",
                index=False, encoding='utf-8-sig'
            )
            logger.info("  OK: policy_validation_predictions.csv")
        
        # 4. threshold_policy_comparison.csv
        if threshold_results is not None:
            threshold_df = pd.DataFrame([r.to_dict() for r in threshold_results])
            threshold_df.to_csv(
                output_dir / "threshold_policy_comparison.csv",
                index=False, encoding='utf-8-sig'
            )
            logger.info("  OK: threshold_policy_comparison.csv")
        
        # 5. final_holdout_predictions.csv
        if holdout_predictions_df is not None:
            holdout_predictions_df.to_csv(
                output_dir / "final_holdout_predictions.csv",
                index=False, encoding='utf-8-sig'
            )
            logger.info("  OK: final_holdout_predictions.csv")
        
        # 6. final_holdout_metrics.json
        if holdout_metrics is not None:
            holdout_metrics_with_meta = {
                **holdout_metrics,
                "threshold_config": {
                    "lower_threshold": lower_threshold,
                    "upper_threshold": upper_threshold,
                    "source": "phase3_recommended" if self.selected_lower_threshold is not None else "default",
                },
            }
            with open(output_dir / "final_holdout_metrics.json", 'w', encoding='utf-8') as f:
                json.dump(holdout_metrics_with_meta, f, ensure_ascii=False, indent=2)
            logger.info("  OK: final_holdout_metrics.json")
        
        # 7. zone_summary.csv
        if zone_summaries is not None:
            zone_df = pd.DataFrame([z.to_dict() for z in zone_summaries])
            zone_df.to_csv(output_dir / "zone_summary.csv", index=False, encoding='utf-8-sig')
            logger.info("  OK: zone_summary.csv")
        
        # 8. diagnostics_summary.json
        if self.diagnostics is not None:
            with open(output_dir / "diagnostics_summary.json", 'w', encoding='utf-8') as f:
                json.dump(self.diagnostics.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info("  OK: diagnostics_summary.json")
        
        # 9. 儲存模型
        if self.final_champion_model is not None:
            model_path = output_dir / "final_champion_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.final_champion_model, f)
            logger.info("  OK: final_champion_model.pkl")
        
        # 10. feature_names.json
        with open(output_dir / "feature_names.json", 'w', encoding='utf-8') as f:
            json.dump(self.feature_names, f, ensure_ascii=False, indent=2)
        logger.info("  OK: feature_names.json")
        
        # 11. zone_policy_summary.json
        if threshold_results is not None:
            # 使用 score_threshold_policy 排序後的結果
            bc_cfg_save = self.config.business_constraints if hasattr(self.config, 'business_constraints') else BusinessConstraintConfig()
            scored_for_save = score_threshold_policy(threshold_results, bc_cfg_save)
            passing_for_save = [r for r in scored_for_save if r.passes_hard_constraints]
            
            if passing_for_save:
                best = passing_for_save[0]
            elif scored_for_save:
                best = scored_for_save[0]
            else:
                best = None
            
            if best is not None:
                zone_policy = {
                    "selected_lower_threshold": lower_threshold,
                    "selected_upper_threshold": upper_threshold,
                    "threshold_source": "phase3_recommended" if self.selected_lower_threshold is not None else "default",
                    "recommended_lower_threshold": best.lower_threshold,
                    "recommended_upper_threshold": best.upper_threshold,
                    "threshold_score": best.threshold_score,
                    "passes_hard_constraints": best.passes_hard_constraints,
                    "auto_decision_rate": best.auto_decision_rate,
                    "manual_review_load": best.manual_review_load,
                    "high_zone_ratio": best.high_zone_ratio,
                    "review_zone_ratio": best.review_zone_ratio,
                    "low_zone_ratio": best.low_zone_ratio,
                    "high_zone_precision": best.expected_precision_high,
                    "low_zone_precision": best.expected_precision_low,
                    "business_constraints": bc_cfg_save.to_dict(),
                }
                
                with open(output_dir / "zone_policy_summary.json", 'w', encoding='utf-8') as f:
                    json.dump(zone_policy, f, ensure_ascii=False, indent=2)
                logger.info("  OK: zone_policy_summary.json")
        
        return output_dir
    
    # ============================================
    # 完整執行
    # ============================================
    def run_full_pipeline(
        self,
        model_names: List[str] = None,
        use_calibration: bool = True,
        lower_threshold: float = 0.5,
        upper_threshold: float = 0.85,
        lower_thresholds: List[float] = None,
        upper_thresholds: List[float] = None
    ) -> Dict:
        """
        執行完整四階段 Pipeline
        
        Phase 1: Model Development (Rolling Training)
        Phase 2: Champion Retraining
        Phase 3: Policy Validation
        Phase 4: Final Blind Holdout
        """
        try:
            # Phase 1: Rolling Training
            self.run_phase1_rolling_training(model_names)
            
            # Select Champion (Phase 1 結果)
            self.select_champion_strategy()
            
            # NOTE: Conservative Fine-Tuning (Phase 1 -> Phase 2 之間)
            tuning_cfg = self.config.tuning if hasattr(self.config, 'tuning') else TuningConfig()
            if tuning_cfg.enable_tuning:
                tuning_result = self.run_conservative_tuning(use_calibration=use_calibration)
            else:
                tuning_result = {}
            
            # Phase 2: Champion Retraining (使用 tuning best config)
            self.run_phase2_champion_retraining(use_calibration=use_calibration)
            
            # Phase 3: Policy Validation
            policy_pred_df, threshold_results, pv_metrics = self.run_phase3_policy_validation(
                lower_thresholds=lower_thresholds,
                upper_thresholds=upper_thresholds
            )
            
            # NOTE: Phase 3 之後，決定最終使用的 threshold
            # 優先使用 Phase 3 推薦值；若無，fallback 到 CLI / default
            effective_lower = self.selected_lower_threshold or lower_threshold
            effective_upper = self.selected_upper_threshold or upper_threshold
            
            logger.info(f"\nNOTE: 最終使用 Threshold: lower={effective_lower}, upper={effective_upper}")
            if self.selected_lower_threshold is not None:
                logger.info(f"  來源: Phase 3 Policy Validation 推薦")
            else:
                logger.info(f"  來源: CLI / 預設值 (Phase 3 未產生推薦)")
            
            # Phase 4: Final Holdout — 使用 effective threshold
            holdout_pred_df, holdout_metrics, zone_summaries = self.run_phase4_final_holdout(
                lower_threshold=effective_lower,
                upper_threshold=effective_upper
            )
            
            # Diagnostics
            self.compare_train_monitor_holdout_gap()
            
            # 儲存結果 — 使用 effective threshold
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.save_all_results(
                run_id=run_id,
                policy_predictions_df=policy_pred_df,
                threshold_results=threshold_results,
                holdout_predictions_df=holdout_pred_df,
                holdout_metrics=holdout_metrics,
                zone_summaries=zone_summaries,
                lower_threshold=effective_lower,
                upper_threshold=effective_upper
            )
            
            logger.info("\n" + "=" * 80)
            logger.info(" 四階段 Pipeline 完成！")
            logger.info("=" * 80)
            
            return {
                "output_dir": str(output_dir),
                "champion_strategy": self.champion_strategy,
                "holdout_metrics": holdout_metrics,
                "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
                "selected_lower_threshold": effective_lower,
                "selected_upper_threshold": effective_upper,
            }
            
        finally:
            self._stop_spark()


# ============================================
# Entry Point
# ============================================
def run_four_phase_pipeline(
    project_root: Path = None,
    model_names: List[str] = None,
    use_calibration: bool = True,
    imbalance_strategy: str = "scale_weight",
    lower_threshold: float = 0.5,
    upper_threshold: float = 0.85
) -> Dict:
    """
    執行 Four Phase Training Pipeline
    
    Args:
        project_root: 專案根目錄
        model_names: 要訓練的模型列表
        use_calibration: 是否使用 probability calibration
        imbalance_strategy: 不平衡處理策略
        lower_threshold: 三區間低門檻（fallback，若 Phase 3 有推薦值則優先使用）
        upper_threshold: 三區間高門檻（fallback，若 Phase 3 有推薦值則優先使用）
        
    Returns:
        Pipeline 結果字典（含 selected_lower/upper_threshold）
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent
    
    # 嘗試載入 pipeline_config.yaml
    config_path = Path(project_root) / "config" / "pipeline_config.yaml"
    if config_path.exists():
        config = ConfigManager(config_path)
        logger.info(f"載入設定檔: {config_path}")
    else:
        config = default_config
        logger.info("使用預設設定 (default_config)")
    
    trainer = FourPhaseTrainer(
        project_root=project_root,
        imbalance_strategy=imbalance_strategy,
        config=config
    )
    
    return trainer.run_full_pipeline(
        model_names=model_names,
        use_calibration=use_calibration,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Four Phase Training for Credit Scoring")
    parser.add_argument("--project-root", type=str, default=".", help="專案根目錄")
    parser.add_argument("--models", nargs="+", default=None, help="要訓練的模型")
    parser.add_argument("--no-calibration", action="store_true", help="不使用 calibration")
    parser.add_argument("--imbalance", type=str, default="scale_weight", 
                        choices=["scale_weight", "class_weight", "smote", "undersample", "none"],
                        help="Imbalance handling 策略")
    parser.add_argument("--lower-threshold", type=float, default=0.5, help="三區間低門檻 (fallback)")
    parser.add_argument("--upper-threshold", type=float, default=0.85, help="三區間高門檻 (fallback)")
    
    args = parser.parse_args()
    
    run_four_phase_pipeline(
        project_root=Path(args.project_root),
        model_names=args.models,
        use_calibration=not args.no_calibration,
        imbalance_strategy=args.imbalance,
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold
    )
