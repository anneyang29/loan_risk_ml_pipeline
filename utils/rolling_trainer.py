"""
Rolling Training Module (LEGACY)
=================================
⚠️ 此模組為 legacy / backward compatibility 用途。
⚠️ 正式主訓練架構已改用 four_phase_trainer.py (FourPhaseTrainer)。

保留原因：
1. 向後相容：部分 notebook / 分析腳本可能仍引用此模組
2. Baseline 比較：可作為單純 rolling training 的 baseline

如果你正在開發新功能，請使用 four_phase_trainer.py。

原始功能說明：
實作真正的 Rolling Training Framework for Credit Scoring

核心設計理念：
1. Development Period (前18個月) 內採 Rolling Window Training
   - 每個 cycle: 4 個月 training + 2 個月 monitoring
   - Training window 內做 time-based CV（不是 random K-fold）
   
2. Champion Strategy Selection
   - 綜合所有 rolling cycles 的結果
   - 考慮 average metrics + stability (std)
   - 不是選單一 cycle 的最佳模型

3. Final Champion Model
   - 選出 champion strategy 後，用完整 18 個月 development 重訓
   - 此模型才用於 final OOT inference 和 production

4. 三區間 Threshold
   - upper_threshold / lower_threshold
   - 自動審核區 / 人工審核區 / 自動拒絕區

5. Retraining 支援
   - Metric trigger (AUC/F1 低於門檻)
   - Time trigger (固定週期)
   - 動態資料窗（當前日期往前推 18 個月）

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

from .config import ConfigManager, default_config, CONFIG_VERSION
from .model_registry import ModelRegistry

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 忽略部分警告
warnings.filterwarnings('ignore', category=UserWarning)

# Random seed
RANDOM_STATE = 2022


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
    """單一 Rolling Cycle 的結果"""
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
    
    # Monitor window metrics
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
    
    # Stability score (lower is better)
    stability_score: float = 0.0
    
    # Overall score (for ranking)
    overall_score: float = 0.0
    
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


# ============================================
# Imbalance Handling
# ============================================
class ImbalanceHandler:
    """
    處理 Label Imbalance 的策略
    
    支援：
    1. scale_pos_weight (XGBoost)
    2. class_weight (sklearn models)
    3. SMOTE (僅在 training folds 內)
    4. Under/Over sampling
    """
    
    def __init__(
        self,
        strategy: str = "scale_weight",  # "scale_weight", "smote", "undersample"
        random_state: int = RANDOM_STATE
    ):
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
    
    def resample_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        對訓練資料進行 resampling
        注意：只能在 training folds 內使用！
        """
        if self.strategy == "smote":
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=self.random_state)
                X_res, y_res = smote.fit_resample(X, y)
                logger.info(f"SMOTE: {len(y)} -> {len(y_res)} samples")
                return X_res, y_res
            except ImportError:
                logger.warning("imblearn not installed, using scale_weight instead")
                return X, y
                
        elif self.strategy == "undersample":
            # 對多數類進行下採樣
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
            # scale_weight 策略不需要 resample
            return X, y


# ============================================
# Time-Based Cross Validation
# ============================================
class TimeBasedCV:
    """
    Time-Based Cross Validation (Walk-Forward / Expanding Window)
    
    確保訓練時只用「過去」預測「未來」，避免 data leakage
    
    設計：
    - 在 4 個月 training window 內
    - 採用 expanding window validation
    - 例如：第1-2月訓練 -> 驗證第3月
    -       第1-3月訓練 -> 驗證第4月
    """
    
    def __init__(
        self,
        n_splits: int = 2,
        min_train_periods: int = 2  # 最少需要幾個月的訓練資料
    ):
        self.n_splits = n_splits
        self.min_train_periods = min_train_periods
    
    def split(
        self,
        df: pd.DataFrame,
        date_column: str = "進件日"
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        產生 time-based CV splits
        
        Args:
            df: 訓練資料（必須包含日期欄位）
            date_column: 日期欄位名稱
            
        Returns:
            List of (train_indices, val_indices)
        """
        # 確保日期排序
        df = df.sort_values(date_column).reset_index(drop=True)
        
        # 取得月份
        df['_year_month'] = pd.to_datetime(df[date_column]).dt.to_period('M')
        unique_months = df['_year_month'].unique()
        unique_months = sorted(unique_months)
        
        n_months = len(unique_months)
        splits = []
        
        # Expanding window: 至少要有 min_train_periods 個月來訓練
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
        
        # 清理臨時欄位
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
    
    支援的模型：
    1. Logistic Regression (baseline)
    2. Random Forest
    3. XGBoost (優先)
    """
    
    @staticmethod
    def get_logistic_regression(class_weight: Dict = None) -> LogisticRegression:
        """Logistic Regression baseline"""
        return LogisticRegression(
            penalty='l2',
            C=1.0,
            class_weight=class_weight or 'balanced',
            solver='lbfgs',
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
    @staticmethod
    def get_random_forest(class_weight: Dict = None) -> RandomForestClassifier:
        """Random Forest"""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight=class_weight or 'balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
    @staticmethod
    def get_xgboost(scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
        """XGBoost (優先選擇)"""
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
    @staticmethod
    def get_all_candidates(
        scale_pos_weight: float = 1.0,
        class_weight: Dict = None
    ) -> Dict[str, Any]:
        """取得所有候選模型"""
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


# ============================================
# Score Zone Assignment
# ============================================
def assign_score_zone(
    prob: np.ndarray,
    lower_threshold: float = 0.4,
    upper_threshold: float = 0.7
) -> np.ndarray:
    """
    分配三區間
    
    Args:
        prob: 預測機率 (核准機率，值越高越可能核准)
        lower_threshold: 低門檻
        upper_threshold: 高門檻
        
    Returns:
        zone: 區間標籤
        - 2: 高通過機率區 (prob >= upper_threshold) -> 自動核准
        - 1: 人工審核區 (lower_threshold <= prob < upper_threshold)
        - 0: 低通過機率區 (prob < lower_threshold) -> 自動拒絕
    """
    zone = np.zeros(len(prob), dtype=int)
    zone[prob >= upper_threshold] = 2
    zone[(prob >= lower_threshold) & (prob < upper_threshold)] = 1
    zone[prob < lower_threshold] = 0
    
    return zone


def evaluate_zone_performance(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    lower_threshold: float = 0.4,
    upper_threshold: float = 0.7
) -> List[ZoneSummary]:
    """
    評估三區間的表現
    
    Args:
        y_true: 實際標籤 (1=核准, 0=婉拒)
        y_pred_proba: 預測核准機率
        lower_threshold: 低門檻
        upper_threshold: 高門檻
        
    Returns:
        三個 ZoneSummary
    """
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
            summary = ZoneSummary(
                zone_name=zone_names[zone_id],
                count=0,
                ratio=0.0
            )
        else:
            y_zone = y_true[mask]
            prob_zone = y_pred_proba[mask]
            
            # 計算該區間內的實際核准/婉拒率
            actual_approve_rate = np.mean(y_zone) if len(y_zone) > 0 else 0
            actual_reject_rate = 1 - actual_approve_rate
            
            # 計算 precision/recall（在該區間內）
            # 對於高通過機率區，我們預測全部核准
            # 對於低通過機率區，我們預測全部拒絕
            if zone_id == 2:  # 高通過機率區 - 預測核准
                precision = actual_approve_rate  # TP / (TP + FP)
                recall = None  # 需要全域計算
            elif zone_id == 0:  # 低通過機率區 - 預測拒絕
                precision = actual_reject_rate  # TN / (TN + FN)
                recall = None
            else:  # 人工審核區
                precision = None
                recall = None
            
            summary = ZoneSummary(
                zone_name=zone_names[zone_id],
                count=int(count),
                ratio=count / total,
                avg_prob=float(np.mean(prob_zone)),
                min_prob=float(np.min(prob_zone)),
                max_prob=float(np.max(prob_zone)),
                actual_approve_rate=float(actual_approve_rate),
                actual_reject_rate=float(actual_reject_rate),
                precision=precision
            )
        
        results.append(summary)
    
    return results


# ============================================
# Rolling Trainer (主要類別)
# ============================================
class RollingTrainer:
    """
    Rolling Training Framework
    
    核心流程：
    1. 載入 rolling window definition
    2. 對每個 cycle 執行:
       - 載入 training window 資料
       - 執行 time-based CV 訓練多個候選模型
       - 載入 monitor window 資料
       - 評估 monitor 表現
    3. 彙總所有 cycles 的結果
    4. 選出 champion strategy
    5. 用完整 development 資料重訓 final champion model
    6. 對 OOT 做 inference 並評估
    7. 輸出三區間分析
    """
    
    def __init__(
        self,
        project_root: Path,
        config: ConfigManager = None,
        random_state: int = RANDOM_STATE
    ):
        self.project_root = Path(project_root)
        self.config = config or default_config
        self.random_state = random_state
        
        # Paths
        self.gold_path = self.project_root / "datamart" / "gold"
        self.development_path = self.gold_path / "development"
        self.oot_path = self.gold_path / "oot"
        self.rolling_def_path = self.gold_path / "rolling_window_definition.csv"
        self.output_path = self.project_root / "model_bank"
        
        # Handlers
        self.imbalance_handler = ImbalanceHandler(strategy="scale_weight")
        self.time_cv = TimeBasedCV(n_splits=2)
        self.metrics_calc = MetricsCalculator()
        
        # Results storage
        self.window_definitions: List[WindowDefinition] = []
        self.all_cycle_results: Dict[str, List[CycleResult]] = {}  # model_name -> results
        self.strategy_results: Dict[str, StrategyResult] = {}
        self.champion_strategy: Optional[str] = None
        self.final_champion_model = None
        self.feature_names: List[str] = []
        
        # Spark session
        self.spark: Optional[SparkSession] = None
        
    def _get_spark(self) -> SparkSession:
        """取得或建立 Spark Session"""
        if self.spark is None:
            self.spark = SparkSession.builder \
                .appName("RollingTrainer") \
                .getOrCreate()
        return self.spark
    
    def _stop_spark(self):
        """停止 Spark Session"""
        if self.spark is not None:
            self.spark.stop()
            self.spark = None
    
    # ============================================
    # 資料載入
    # ============================================
    def load_rolling_window_definition(self) -> List[WindowDefinition]:
        """
        載入 rolling window 定義
        
        rolling 的目的：
        1. 模擬未來部署後的短期表現
        2. 進行 model selection 與 stability testing
        3. 每個 cycle 不是直接上線模型，而是用來選出最佳策略
        """
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
    
    def load_development_data(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        載入 development 資料
        
        Args:
            start_date: 開始日期（可選，用於過濾）
            end_date: 結束日期（可選，用於過濾）
        """
        logger.info(f"載入 Development 資料...")
        spark = self._get_spark()
        
        df_spark = spark.read.parquet(str(self.development_path))
        
        # 日期過濾
        if start_date:
            df_spark = df_spark.filter(F.col("進件日") >= start_date)
        if end_date:
            df_spark = df_spark.filter(F.col("進件日") <= end_date)
        
        df = df_spark.toPandas()
        logger.info(f"Development 資料: {len(df)} 筆")
        
        return df
    
    def load_oot_data(self) -> pd.DataFrame:
        """載入 OOT 資料"""
        logger.info(f"載入 OOT 資料...")
        spark = self._get_spark()
        
        df = spark.read.parquet(str(self.oot_path)).toPandas()
        logger.info(f"OOT 資料: {len(df)} 筆")
        
        return df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """取得特徵欄位"""
        # 排除 ID、日期、標籤等非特徵欄位
        exclude_cols = [
            '案件編號', '進件日', '進件年月', '授信結果', '授信結果_二元',
            # 原始類別欄位（已編碼）
            '性別', '教育程度', '婚姻狀況', '月所得', '職業說明', '居住地', '廠牌車型', '動產設定',
            # 其他
            'bronze_load_timestamp', 'silver_process_timestamp', 'gold_process_timestamp',
            # Partitioning
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
        
        # 確保欄位存在
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[feature_cols].values.astype(np.float32)
        y = df['授信結果_二元'].values.astype(int)
        
        # 處理 NaN
        X = np.nan_to_num(X, nan=0.0)
        
        return X, y, feature_cols
    
    # ============================================
    # 單一 Cycle 訓練
    # ============================================
    def train_candidate_models_for_window(
        self,
        window: WindowDefinition,
        model_names: List[str] = None
    ) -> Dict[str, CycleResult]:
        """
        對單一 training window 訓練所有候選模型
        
        Args:
            window: Window 定義
            model_names: 要訓練的模型名稱列表
            
        Returns:
            Dict[model_name, CycleResult]
        """
        logger.info("=" * 60)
        logger.info(f"Window {window.window_id}: 訓練候選模型")
        logger.info(f"  Training: {window.train_start} ~ {window.train_end}")
        logger.info(f"  Monitor: {window.monitor_start} ~ {window.monitor_end}")
        logger.info("=" * 60)
        
        # 載入訓練資料
        train_df = self.load_development_data(
            start_date=window.train_start,
            end_date=window.train_end
        )
        
        if len(train_df) == 0:
            logger.warning(f"Window {window.window_id}: 無訓練資料！")
            return {}
        
        # 載入監控資料
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
        
        logger.info(f"訓練資料: {len(y_train)} 筆, 正樣本比例: {np.mean(y_train):.2%}")
        logger.info(f"監控資料: {len(y_monitor)} 筆, 正樣本比例: {np.mean(y_monitor):.2%}")
        
        # 計算 imbalance weights
        scale_pos_weight = self.imbalance_handler.calculate_scale_pos_weight(y_train)
        class_weight = self.imbalance_handler.calculate_class_weight(y_train)
        
        # 取得候選模型
        if model_names is None:
            model_names = ["logistic_regression", "random_forest", "xgboost"]
        
        candidates = CandidateModels.get_all_candidates(scale_pos_weight, class_weight)
        
        # Time-based CV splits
        cv_splits = self.time_cv.split(train_df.reset_index(drop=True))
        
        results = {}
        
        for model_name in model_names:
            if model_name not in candidates:
                continue
                
            logger.info(f"\n--- 訓練 {model_name} ---")
            
            model_template = candidates[model_name]
            
            # Time-based CV
            cv_metrics = self._run_time_based_cv(
                model_template,
                X_train,
                y_train,
                train_df.reset_index(drop=True),
                cv_splits
            )
            
            # 用完整 training window 訓練最終模型
            model = deepcopy(model_template)
            model.fit(X_train, y_train)
            
            # 在 monitor window 上評估
            monitor_pred_proba = model.predict_proba(X_monitor)[:, 1]
            monitor_metrics = self.metrics_calc.calculate_all_metrics(
                y_monitor, monitor_pred_proba
            )
            score_dist = self.metrics_calc.calculate_score_distribution(monitor_pred_proba)
            
            # 建立結果
            result = CycleResult(
                window_id=window.window_id,
                model_name=model_name,
                # CV metrics
                cv_auc_mean=cv_metrics['auc_mean'],
                cv_auc_std=cv_metrics['auc_std'],
                cv_f1_mean=cv_metrics['f1_mean'],
                cv_f1_std=cv_metrics['f1_std'],
                cv_f1_reject_mean=cv_metrics['f1_reject_mean'],
                cv_f1_reject_std=cv_metrics['f1_reject_std'],
                cv_precision_mean=cv_metrics['precision_mean'],
                cv_recall_mean=cv_metrics['recall_mean'],
                # Monitor metrics
                monitor_auc=monitor_metrics['auc'],
                monitor_f1=monitor_metrics['f1'],
                monitor_f1_reject=monitor_metrics['f1_reject'],
                monitor_precision=monitor_metrics['precision'],
                monitor_recall=monitor_metrics['recall'],
                monitor_ks=monitor_metrics['ks'],
                monitor_brier=monitor_metrics['brier_score'],
                # Score distribution
                monitor_score_mean=score_dist['mean'],
                monitor_score_std=score_dist['std'],
                monitor_score_median=score_dist['median'],
                # Data counts
                train_rows=len(y_train),
                monitor_rows=len(y_monitor),
                train_positive_ratio=float(np.mean(y_train)),
                monitor_positive_ratio=float(np.mean(y_monitor)),
            )
            
            results[model_name] = result
            
            logger.info(f"  CV AUC: {result.cv_auc_mean:.4f} ± {result.cv_auc_std:.4f}")
            logger.info(f"  Monitor AUC: {result.monitor_auc:.4f}")
            logger.info(f"  Monitor F1_reject: {result.monitor_f1_reject:.4f}")
        
        return results
    
    def _run_time_based_cv(
        self,
        model_template,
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        cv_splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, float]:
        """
        執行 Time-based CV
        
        Args:
            model_template: 模型模板
            X: 特徵矩陣
            y: 標籤
            df: 原始 DataFrame（用於取得日期）
            cv_splits: CV splits
            
        Returns:
            CV metrics 摘要
        """
        auc_scores = []
        f1_scores = []
        f1_reject_scores = []
        precision_scores = []
        recall_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # 若使用 resampling，只在訓練 fold 內做
            if self.imbalance_handler.strategy in ["smote", "undersample"]:
                X_fold_train, y_fold_train = self.imbalance_handler.resample_training_data(
                    X_fold_train, y_fold_train
                )
            
            # 訓練
            model = deepcopy(model_template)
            
            # 重新計算 scale_pos_weight（如果是 XGBoost）
            if hasattr(model, 'scale_pos_weight'):
                new_weight = self.imbalance_handler.calculate_scale_pos_weight(y_fold_train)
                model.set_params(scale_pos_weight=new_weight)
            
            model.fit(X_fold_train, y_fold_train)
            
            # 預測（對 monitor/validation 只做 transform，不 resample）
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            
            # 計算指標
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
    
    # ============================================
    # Rolling Training (所有 Cycles)
    # ============================================
    def run_rolling_training(
        self,
        model_names: List[str] = None
    ) -> Dict[str, List[CycleResult]]:
        """
        執行所有 rolling cycles 的訓練
        
        Args:
            model_names: 要訓練的模型列表
            
        Returns:
            Dict[model_name, List[CycleResult]]
        """
        logger.info("=" * 80)
        logger.info("開始 Rolling Training")
        logger.info("=" * 80)
        
        # 載入 window 定義
        if not self.window_definitions:
            self.load_rolling_window_definition()
        
        # 初始化結果儲存
        if model_names is None:
            model_names = ["logistic_regression", "random_forest", "xgboost"]
        
        self.all_cycle_results = {name: [] for name in model_names}
        
        # 執行每個 cycle
        for window in self.window_definitions:
            cycle_results = self.train_candidate_models_for_window(window, model_names)
            
            for model_name, result in cycle_results.items():
                self.all_cycle_results[model_name].append(result)
        
        logger.info("\n" + "=" * 80)
        logger.info("Rolling Training 完成")
        logger.info("=" * 80)
        
        return self.all_cycle_results
    
    # ============================================
    # Champion Strategy Selection
    # ============================================
    def aggregate_rolling_results(self) -> Dict[str, StrategyResult]:
        """
        彙總所有 rolling cycles 的結果
        
        Champion Strategy vs Final Champion Artifact 的差別：
        - Champion Strategy: 指一種模型+參數的組合，由所有 cycles 的平均表現決定
        - Final Champion Artifact: 用 champion strategy 在完整 development 資料上重訓的實際模型檔案
        """
        logger.info("\n彙總 Rolling Results...")
        
        self.strategy_results = {}
        
        for model_name, cycle_results in self.all_cycle_results.items():
            if not cycle_results:
                continue
            
            # 計算各指標的平均與標準差
            cv_aucs = [r.cv_auc_mean for r in cycle_results]
            monitor_aucs = [r.monitor_auc for r in cycle_results]
            monitor_f1s = [r.monitor_f1 for r in cycle_results]
            monitor_f1_rejects = [r.monitor_f1_reject for r in cycle_results]
            cv_f1_rejects = [r.cv_f1_reject_mean for r in cycle_results]
            
            # 計算穩定性分數（標準差越低越穩定）
            stability_score = (
                np.std(cv_aucs) + 
                np.std(monitor_aucs) + 
                np.std(monitor_f1_rejects)
            ) / 3
            
            # 計算整體分數
            # 考慮: avg_cv_auc, avg_monitor_auc, avg_monitor_f1_reject, -stability
            overall_score = (
                np.mean(cv_aucs) * 0.25 +
                np.mean(monitor_aucs) * 0.35 +
                np.mean(monitor_f1_rejects) * 0.30 -
                stability_score * 0.10
            )
            
            strategy = StrategyResult(
                model_name=model_name,
                avg_cv_auc=float(np.mean(cv_aucs)),
                std_cv_auc=float(np.std(cv_aucs)),
                avg_cv_f1_reject=float(np.mean(cv_f1_rejects)),
                std_cv_f1_reject=float(np.std(cv_f1_rejects)),
                avg_monitor_auc=float(np.mean(monitor_aucs)),
                std_monitor_auc=float(np.std(monitor_aucs)),
                avg_monitor_f1=float(np.mean(monitor_f1s)),
                std_monitor_f1=float(np.std(monitor_f1s)),
                avg_monitor_f1_reject=float(np.mean(monitor_f1_rejects)),
                std_monitor_f1_reject=float(np.std(monitor_f1_rejects)),
                stability_score=float(stability_score),
                overall_score=float(overall_score),
                cycle_results=cycle_results
            )
            
            self.strategy_results[model_name] = strategy
            
            logger.info(f"\n{model_name}:")
            logger.info(f"  Avg CV AUC: {strategy.avg_cv_auc:.4f} ± {strategy.std_cv_auc:.4f}")
            logger.info(f"  Avg Monitor AUC: {strategy.avg_monitor_auc:.4f} ± {strategy.std_monitor_auc:.4f}")
            logger.info(f"  Avg Monitor F1_reject: {strategy.avg_monitor_f1_reject:.4f} ± {strategy.std_monitor_f1_reject:.4f}")
            logger.info(f"  Stability Score: {strategy.stability_score:.4f}")
            logger.info(f"  Overall Score: {strategy.overall_score:.4f}")
        
        return self.strategy_results
    
    def select_champion_strategy(self) -> str:
        """
        選出最佳 champion strategy
        
        選擇標準：
        1. 綜合 average cv_auc, monitor_auc, monitor_f1_reject
        2. 考慮穩定性（標準差）
        3. 不是只看單一 cycle
        """
        if not self.strategy_results:
            self.aggregate_rolling_results()
        
        # 依 overall_score 排序
        sorted_strategies = sorted(
            self.strategy_results.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        self.champion_strategy = sorted_strategies[0][0]
        
        logger.info("\n" + "=" * 60)
        logger.info("Champion Strategy Selection")
        logger.info("=" * 60)
        logger.info(f"選出 Champion: {self.champion_strategy}")
        
        champion = self.strategy_results[self.champion_strategy]
        logger.info(f"  Overall Score: {champion.overall_score:.4f}")
        logger.info(f"  Avg Monitor AUC: {champion.avg_monitor_auc:.4f}")
        logger.info(f"  Avg Monitor F1_reject: {champion.avg_monitor_f1_reject:.4f}")
        
        # 輸出排名
        logger.info("\n策略排名:")
        for i, (name, strategy) in enumerate(sorted_strategies):
            logger.info(f"  {i+1}. {name}: Score={strategy.overall_score:.4f}")
        
        return self.champion_strategy
    
    # ============================================
    # Final Champion Model Training
    # ============================================
    def retrain_final_champion(
        self,
        use_calibration: bool = True,
        calibration_method: str = "isotonic"
    ):
        """
        用完整 development 資料重訓 final champion model
        
        這是實際會被部署的模型
        """
        if not self.champion_strategy:
            self.select_champion_strategy()
        
        logger.info("\n" + "=" * 60)
        logger.info("重訓 Final Champion Model")
        logger.info("=" * 60)
        logger.info(f"Champion Strategy: {self.champion_strategy}")
        
        # 載入完整 development 資料
        dev_df = self.load_development_data()
        X, y, feature_cols = self._prepare_xy(dev_df)
        
        self.feature_names = feature_cols
        
        logger.info(f"Development 資料: {len(y)} 筆")
        logger.info(f"特徵數: {len(feature_cols)}")
        
        # 取得對應的模型
        scale_pos_weight = self.imbalance_handler.calculate_scale_pos_weight(y)
        class_weight = self.imbalance_handler.calculate_class_weight(y)
        
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
        
        logger.info("✓ Final Champion Model 訓練完成")
        
        # 評估 in-sample
        y_pred_proba = self.final_champion_model.predict_proba(X)[:, 1]
        metrics = self.metrics_calc.calculate_all_metrics(y, y_pred_proba)
        
        logger.info(f"\nIn-Sample Metrics:")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        logger.info(f"  F1_reject: {metrics['f1_reject']:.4f}")
        logger.info(f"  KS: {metrics['ks']:.4f}")
        
        return self.final_champion_model
    
    # ============================================
    # OOT Inference
    # ============================================
    def infer_on_oot(
        self,
        lower_threshold: float = 0.4,
        upper_threshold: float = 0.7
    ) -> Tuple[pd.DataFrame, Dict, List[ZoneSummary]]:
        """
        對 OOT 做 inference 並評估
        
        OOT 不參與任何 model selection，只用於最終評估
        
        Args:
            lower_threshold: 低通過門檻
            upper_threshold: 高通過門檻
            
        Returns:
            (predictions_df, metrics_dict, zone_summaries)
        """
        if self.final_champion_model is None:
            raise ValueError("請先執行 retrain_final_champion()")
        
        logger.info("\n" + "=" * 60)
        logger.info("OOT Inference")
        logger.info("=" * 60)
        
        # 載入 OOT 資料
        oot_df = self.load_oot_data()
        X_oot, y_oot, _ = self._prepare_xy(oot_df, self.feature_names)
        
        logger.info(f"OOT 資料: {len(y_oot)} 筆")
        logger.info(f"正樣本比例: {np.mean(y_oot):.2%}")
        
        # Inference
        y_pred_proba = self.final_champion_model.predict_proba(X_oot)[:, 1]
        
        # 計算 metrics
        oot_metrics = self.metrics_calc.calculate_all_metrics(y_oot, y_pred_proba)
        
        logger.info(f"\nOOT Metrics:")
        logger.info(f"  AUC: {oot_metrics['auc']:.4f}")
        logger.info(f"  F1: {oot_metrics['f1']:.4f}")
        logger.info(f"  F1_reject: {oot_metrics['f1_reject']:.4f}")
        logger.info(f"  KS: {oot_metrics['ks']:.4f}")
        logger.info(f"  Brier Score: {oot_metrics['brier_score']:.4f}")
        
        # Confusion Matrix
        logger.info(f"\n  Confusion Matrix:")
        logger.info(f"    TP={oot_metrics['true_positive']}, FP={oot_metrics['false_positive']}")
        logger.info(f"    FN={oot_metrics['false_negative']}, TN={oot_metrics['true_negative']}")
        
        # 三區間分析
        logger.info(f"\n三區間分析 (lower={lower_threshold}, upper={upper_threshold}):")
        zone_summaries = evaluate_zone_performance(
            y_oot, y_pred_proba, lower_threshold, upper_threshold
        )
        
        for zone in zone_summaries:
            logger.info(f"\n  {zone.zone_name}:")
            logger.info(f"    筆數: {zone.count} ({zone.ratio:.2%})")
            logger.info(f"    平均機率: {zone.avg_prob:.4f}")
            logger.info(f"    實際核准率: {zone.actual_approve_rate:.2%}")
            logger.info(f"    實際婉拒率: {zone.actual_reject_rate:.2%}")
        
        # 建立 predictions DataFrame
        predictions_df = oot_df[['案件編號', '進件日', '授信結果_二元']].copy()
        predictions_df['pred_prob'] = y_pred_proba
        predictions_df['pred_zone'] = assign_score_zone(y_pred_proba, lower_threshold, upper_threshold)
        predictions_df['zone_name'] = predictions_df['pred_zone'].map({
            2: '高通過機率區',
            1: '人工審核區',
            0: '低通過機率區'
        })
        
        return predictions_df, oot_metrics, zone_summaries
    
    # ============================================
    # 輸出
    # ============================================
    def save_results(
        self,
        run_id: str = None,
        predictions_df: pd.DataFrame = None,
        oot_metrics: Dict = None,
        zone_summaries: List[ZoneSummary] = None,
        lower_threshold: float = 0.4,
        upper_threshold: float = 0.7
    ):
        """儲存所有結果"""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = self.output_path / f"rolling_training_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n儲存結果至: {output_dir}")
        
        # 1. rolling_results.csv
        rolling_records = []
        for model_name, results in self.all_cycle_results.items():
            for r in results:
                rolling_records.append(r.to_dict())
        
        rolling_df = pd.DataFrame(rolling_records)
        rolling_df.to_csv(output_dir / "rolling_results.csv", index=False, encoding='utf-8-sig')
        logger.info("  ✓ rolling_results.csv")
        
        # 2. champion_summary.json
        champion_data = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "champion_strategy": self.champion_strategy,
            "threshold_config": {
                "lower_threshold": lower_threshold,
                "upper_threshold": upper_threshold
            },
            "strategy_results": {
                name: strategy.to_dict() 
                for name, strategy in self.strategy_results.items()
            }
        }
        
        with open(output_dir / "champion_summary.json", 'w', encoding='utf-8') as f:
            json.dump(champion_data, f, ensure_ascii=False, indent=2)
        logger.info("  ✓ champion_summary.json")
        
        # 3. oot_predictions.csv
        if predictions_df is not None:
            predictions_df.to_csv(
                output_dir / "oot_predictions.csv", 
                index=False, 
                encoding='utf-8-sig'
            )
            logger.info("  ✓ oot_predictions.csv")
        
        # 4. oot_metrics.json
        if oot_metrics is not None:
            with open(output_dir / "oot_metrics.json", 'w', encoding='utf-8') as f:
                json.dump(oot_metrics, f, ensure_ascii=False, indent=2)
            logger.info("  ✓ oot_metrics.json")
        
        # 5. zone_summary.csv
        if zone_summaries is not None:
            zone_df = pd.DataFrame([z.to_dict() for z in zone_summaries])
            zone_df.to_csv(
                output_dir / "zone_summary.csv", 
                index=False, 
                encoding='utf-8-sig'
            )
            logger.info("  ✓ zone_summary.csv")
        
        # 6. 儲存模型
        if self.final_champion_model is not None:
            model_path = output_dir / "final_champion_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.final_champion_model, f)
            logger.info("  ✓ final_champion_model.pkl")
        
        # 7. 儲存 feature names
        with open(output_dir / "feature_names.json", 'w', encoding='utf-8') as f:
            json.dump(self.feature_names, f, ensure_ascii=False, indent=2)
        logger.info("  ✓ feature_names.json")
        
        return output_dir
    
    # ============================================
    # 完整執行
    # ============================================
    def run_full_pipeline(
        self,
        model_names: List[str] = None,
        use_calibration: bool = True,
        lower_threshold: float = 0.4,
        upper_threshold: float = 0.7
    ) -> Path:
        """
        執行完整 pipeline
        
        1. Rolling Training (所有 cycles)
        2. 彙總結果
        3. 選出 Champion Strategy
        4. 重訓 Final Champion Model
        5. OOT Inference
        6. 儲存結果
        """
        try:
            # 1. Rolling Training
            self.run_rolling_training(model_names)
            
            # 2. 彙總結果
            self.aggregate_rolling_results()
            
            # 3. 選出 Champion
            self.select_champion_strategy()
            
            # 4. 重訓 Final Champion
            self.retrain_final_champion(use_calibration=use_calibration)
            
            # 5. OOT Inference
            predictions_df, oot_metrics, zone_summaries = self.infer_on_oot(
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold
            )
            
            # 6. 儲存結果
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.save_results(
                run_id=run_id,
                predictions_df=predictions_df,
                oot_metrics=oot_metrics,
                zone_summaries=zone_summaries,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold
            )
            
            logger.info("\n" + "=" * 80)
            logger.info("Rolling Training Pipeline 完成！")
            logger.info("=" * 80)
            
            return output_dir
            
        finally:
            self._stop_spark()


# ============================================
# Retraining Support
# ============================================
def generate_retraining_window(
    current_date: Union[str, date],
    development_months: int = 18,
    oot_months: int = 6
) -> Dict[str, str]:
    """
    動態產生 retraining 資料窗
    
    為什麼要用「當下日期往前推」而不是固定 historical range？
    - 確保模型總是用最新的 18 個月資料訓練
    - 避免模型陳舊（staleness）
    - 每次 retraining 都能學到最新的 pattern
    
    Args:
        current_date: 當前日期（review/inference date）
        development_months: Development period 長度（月）
        oot_months: OOT period 長度（月）
        
    Returns:
        Dict with keys: dev_start, dev_end, oot_start, oot_end
    """
    if isinstance(current_date, str):
        current_date = pd.to_datetime(current_date).date()
    
    # OOT: current_date 往前推 oot_months
    oot_end = current_date
    oot_start = current_date - relativedelta(months=oot_months)
    
    # Development: oot_start 往前推 development_months
    dev_end = oot_start - relativedelta(days=1)
    dev_start = dev_end - relativedelta(months=development_months) + relativedelta(days=1)
    
    return {
        "dev_start": dev_start.strftime("%Y-%m-%d"),
        "dev_end": dev_end.strftime("%Y-%m-%d"),
        "oot_start": oot_start.strftime("%Y-%m-%d"),
        "oot_end": oot_end.strftime("%Y-%m-%d"),
    }


# ============================================
# Entry Point
# ============================================
def run_rolling_training_pipeline(
    project_root: Path = None,
    model_names: List[str] = None,
    use_calibration: bool = True,
    lower_threshold: float = 0.4,
    upper_threshold: float = 0.7
) -> Path:
    """
    執行 Rolling Training Pipeline 的入口點
    
    Args:
        project_root: 專案根目錄
        model_names: 要訓練的模型列表
        use_calibration: 是否使用 probability calibration
        lower_threshold: 三區間低門檻
        upper_threshold: 三區間高門檻
        
    Returns:
        輸出目錄路徑
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent
    
    trainer = RollingTrainer(project_root=project_root)
    
    return trainer.run_full_pipeline(
        model_names=model_names,
        use_calibration=use_calibration,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rolling Training for Credit Scoring")
    parser.add_argument("--project-root", type=str, default=".", help="專案根目錄")
    parser.add_argument("--models", nargs="+", default=None, help="要訓練的模型")
    parser.add_argument("--no-calibration", action="store_true", help="不使用 calibration")
    parser.add_argument("--lower-threshold", type=float, default=0.4, help="三區間低門檻")
    parser.add_argument("--upper-threshold", type=float, default=0.7, help="三區間高門檻")
    
    args = parser.parse_args()
    
    run_rolling_training_pipeline(
        project_root=Path(args.project_root),
        model_names=args.models,
        use_calibration=not args.no_calibration,
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold
    )
