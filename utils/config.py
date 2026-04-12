"""
Pipeline Configuration
======================
集中管理所有參數、映射表、閾值設定
支援版本控制與可追溯性

Version: 1.0.0
Last Updated: 2024-04-07
Owner: Data Science Team
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import yaml
import logging

logger = logging.getLogger(__name__)


# ============================================
# Version Information
# ============================================
CONFIG_VERSION = "1.0.0"
CONFIG_EFFECTIVE_DATE = "2024-04-07"
CONFIG_OWNER = "Data Science Team"


# ============================================
# Schema Contract: Required Columns
# ============================================
@dataclass
class SchemaContract:
    """資料契約定義"""
    version: str = "1.0.0"
    
    # Bronze 必要欄位
    bronze_required_columns: List[str] = field(default_factory=lambda: [
        "案件編號",
        "進件日",
        "授信結果",
        "年齡",
        "性別",
        "教育程度",
        "婚姻狀況",
        "月所得",
        "職業說明",
        "居住地",
        "廠牌車型",
        "原申辦金額",
        "申辦期數",
        "車齡",
        "動產設定",
        "內部往來次數",
        "近半年同業查詢次數",
        "所留市內電話數",
    ])
    
    # 資料型別定義
    column_types: Dict[str, str] = field(default_factory=lambda: {
        "案件編號": "string",
        "進件日": "date",
        "授信結果": "string",
        "年齡": "integer",
        "性別": "string",
        "教育程度": "string",
        "婚姻狀況": "string",
        "月所得": "string",
        "職業說明": "string",
        "居住地": "string",
        "廠牌車型": "string",
        "原申辦金額": "double",
        "申辦期數": "integer",
        "車齡": "double",
        "動產設定": "string",
        "內部往來次數": "integer",
        "近半年同業查詢次數": "integer",
        "所留市內電話數": "integer",
    })
    
    # 允許的類別值
    allowed_values: Dict[str, List[str]] = field(default_factory=lambda: {
        "性別": ["男", "女", None, "Missing"],
        "婚姻狀況": ["已婚", "未婚", "離婚", "同居", None, "Missing"],
        "教育程度": [
            "國中", "高中", "專科", "大學", "碩士", "碩士以上", "博士以上", "其他", 
            None, "Missing"
        ],
        "授信結果": [
            "APP(核准)", "WTCD(婉拒)", 
            # 其他可能的狀態（需要監控）
            "PEND(審核中)", "CANCEL(取消)",
        ],
        "動產設定": ["Y", "N", "0", "1", None],
    })
    
    # 數值欄位範圍
    numeric_ranges: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "年齡": {"min": 18, "max": 100},
        "申辦期數": {"min": 1, "max": 120},
        "原申辦金額": {"min": 0, "max": 10000000},
        "車齡": {"min": -1, "max": 100},  # -1 表示缺失
        "內部往來次數": {"min": -1, "max": 1000},
        "近半年同業查詢次數": {"min": -1, "max": 100},
        "所留市內電話數": {"min": 0, "max": 10},
    })


# ============================================
# Data Quality Thresholds
# ============================================
@dataclass
class DataQualityThresholds:
    """資料品質閾值"""
    version: str = "1.0.0"
    
    # NULL 比例閾值
    null_ratio_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "案件編號": 0.0,      # 不允許 NULL
        "進件日": 0.0,        # 不允許 NULL
        "授信結果": 0.0,      # 不允許 NULL
        "年齡": 0.05,         # 最多 5% NULL
        "性別": 0.10,         # 最多 10% NULL
        "教育程度": 0.50,     # 最多 50% NULL
        "月所得": 0.30,       # 最多 30% NULL
        "車齡": 0.50,         # 最多 50% NULL（車貸案件可能無車齡）
        "default": 0.30,      # 預設閾值
    })
    
    # 資料筆數閾值
    row_count_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "min_total_rows": 1000,        # 最少總筆數
        "min_monthly_rows": 50,        # 每月最少筆數
        "max_duplicate_ratio": 0.01,   # 最大重複比例
    })
    
    # 類別值異常閾值
    category_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "max_unseen_ratio": 0.05,          # 最大未見類別比例
        "max_category_drift_psi": 0.25,    # PSI 警戒值
        "max_single_category_ratio": 0.95, # 單一類別最大比例
    })
    
    # Label 分布閾值
    label_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "min_positive_ratio": 0.05,    # 最低核准率
        "max_positive_ratio": 0.95,    # 最高核准率
        "max_label_drift": 0.10,       # 月間最大 label 變化
    })


# ============================================
# Feature Encoding Mappings
# ============================================
@dataclass
class FeatureEncodingConfig:
    """特徵編碼設定"""
    version: str = "1.0.0"
    effective_date: str = "2024-04-07"
    
    # 教育程度序位映射
    education_ordinal_mapping: Dict[str, int] = field(default_factory=lambda: {
        "Missing": 0,
        "其他": 1,
        "國中": 2,
        "高中": 3,
        "專科": 4,
        "大學": 5,
        "碩士": 6,
        "碩士以上": 6,
        "博士以上": 7,
    })
    
    # 月所得序位映射（細分區間）
    income_ordinal_mapping_detailed: Dict[str, int] = field(default_factory=lambda: {
        "Missing": 0,
        "~20,000": 1,
        "20,000~24,999": 2,
        "25,000~29,999": 3,
        "30,000~34,999": 4,
        "35,000~39,999": 5,
        "40,000~44,999": 6,
        "45,000~49,999": 7,
        "50,000~54,999": 8,
        "55,000~59,999": 9,
        "60,000~64,999": 10,
        "65,000~69,999": 11,
        "70,000~74,999": 12,
        "75,000~79,999": 13,
        "80,000~": 14,
    })
    
    # 月所得序位映射（粗分區間）
    income_ordinal_mapping_coarse: Dict[str, int] = field(default_factory=lambda: {
        "20,000~29,999": 2,
        "30,000~39,999": 4,
        "40,000~49,999": 6,
        "50,000~59,999": 8,
        "60,000~69,999": 10,
        "70,000~79,999": 12,
    })
    
    # 月所得中位數估計（用於計算合理的負債收入比）
    income_midpoint_mapping: Dict[str, float] = field(default_factory=lambda: {
        "Missing": 30000.0,       # 預設中位數
        "~20,000": 15000.0,
        "20,000~24,999": 22500.0,
        "25,000~29,999": 27500.0,
        "30,000~34,999": 32500.0,
        "35,000~39,999": 37500.0,
        "40,000~44,999": 42500.0,
        "45,000~49,999": 47500.0,
        "50,000~54,999": 52500.0,
        "55,000~59,999": 57500.0,
        "60,000~64,999": 62500.0,
        "65,000~69,999": 67500.0,
        "70,000~74,999": 72500.0,
        "75,000~79,999": 77500.0,
        "80,000~": 100000.0,
        # 粗分區間
        "20,000~29,999": 25000.0,
        "30,000~39,999": 35000.0,
        "40,000~49,999": 45000.0,
        "50,000~59,999": 55000.0,
        "60,000~69,999": 65000.0,
        "70,000~79,999": 75000.0,
    })
    
    # 年齡組映射
    age_group_mapping: Dict[str, Dict] = field(default_factory=lambda: {
        "~20": {"min": 0, "max": 20, "ordinal": 0},
        "21-30": {"min": 21, "max": 30, "ordinal": 1},
        "31-40": {"min": 31, "max": 40, "ordinal": 2},
        "41-50": {"min": 41, "max": 50, "ordinal": 3},
        "51-60": {"min": 51, "max": 60, "ordinal": 4},
        "61~": {"min": 61, "max": 999, "ordinal": 5},
    })
    
    # 目標變數映射
    target_mapping: Dict[str, int] = field(default_factory=lambda: {
        "APP(核准)": 1,
        "WTCD(婉拒)": 0,
    })
    
    # 教育程度推斷規則（職業 → 教育程度）
    education_inference_rules: Dict[str, str] = field(default_factory=lambda: {
        "學生(大專生)": "大學",
        "學生(高中職生)": "高中",
        "學生(國中生)": "國中",
    })
    
    # 特殊值定義
    special_values: Dict[str, Dict] = field(default_factory=lambda: {
        "車齡": {
            "missing_indicator": -1,
            "anomaly_threshold": 100,
        },
        "內部往來次數": {
            "missing_indicator": -1,
        },
        "近半年同業查詢次數": {
            "missing_indicator": -1,
        },
    })


# ============================================
# Feature Definition Config
# ============================================
@dataclass
class FeatureDefinitionConfig:
    """特徵定義設定"""
    version: str = "1.0.0"
    
    # 數值特徵（需要 MinMaxScaler）
    numeric_features_to_scale: List[str] = field(default_factory=lambda: [
        "年齡",
        "申辦期數",
        "原申辦金額_log",
        "車齡_log",
        "內部往來次數_log",
        "近半年同業查詢次數_log",
        "所留市內電話數_log",
    ])
    
    # 序位特徵（已編碼，不需要 scale）
    ordinal_features: List[str] = field(default_factory=lambda: [
        "教育程度_序位",
        "月所得_序位",
        "年齡組_序位",
    ])
    
    # 二元特徵（0/1，不需要 scale）
    binary_features: List[str] = field(default_factory=lambda: [
        "性別_二元",
        "婚姻狀況_二元",
        "動產設定",
        "車齡_是否缺失",
        "車齡_異常旗標",
        "教育程度_是否缺失",
        "月所得_是否缺失",
        "內部往來次數_是否特殊值",
        "近半年同業查詢次數_是否缺失",
    ])
    
    # 高基數類別（用 Frequency Encoding）
    high_cardinality_features: List[str] = field(default_factory=lambda: [
        "居住地",
        "職業說明",
        "廠牌車型",
    ])
    
    # 交互特徵定義
    cross_features: Dict[str, Dict] = field(default_factory=lambda: {
        "負債月所得比": {
            "formula": "原申辦金額 / 月所得_估計值",
            "description": "申辦金額與估計月所得的比值，用於評估負債負擔",
            "numerator": "原申辦金額",
            "denominator": "月所得_估計值",
            "fallback": "原申辦金額",  # 當分母為 0 時
        },
        "年齡_婚姻交互": {
            "formula": "年齡 * 婚姻狀況_二元",
            "description": "年齡與婚姻狀態的交互，捕捉已婚年齡效應",
        },
        "車齡_金額交互": {
            "formula": "車齡_清理後 * 原申辦金額_log",
            "description": "車齡與申辦金額的交互",
        },
        "教育_所得交互": {
            "formula": "教育程度_序位 * 月所得_序位",
            "description": "教育程度與所得的交互",
        },
    })
    
    # 目標變數
    target_column: str = "授信結果_二元"
    
    # Key 欄位
    key_columns: List[str] = field(default_factory=lambda: [
        "案件編號", "進件日", "進件年月"
    ])


# ============================================
# Time Period Config — LEGACY（僅供 Gold Layer 向後相容）
# ============================================
@dataclass
class TimePeriodConfig:
    """
    時間區間設定 — LEGACY
    
    ⚠️ 此 Config 僅供 Gold Layer (data_processing_gold_table) 使用，
       用來定義 development / oot 的原始切分點與 Rolling Window 參數。
    ⚠️ 四階段訓練架構的主設定請使用 PhaseConfigV2。
    """
    version: str = "1.0.0"
    
    # 資料區間
    data_start: date = date(2024, 4, 1)
    data_end: date = date(2026, 3, 31)
    
    # OOT 切分點
    oot_start: date = date(2025, 10, 1)
    
    # Rolling Window 設定
    train_months: int = 4
    monitor_months: int = 2
    step_months: int = 2
    
    def to_dict(self) -> Dict:
        return {
            "data_start": str(self.data_start),
            "data_end": str(self.data_end),
            "oot_start": str(self.oot_start),
            "train_months": self.train_months,
            "monitor_months": self.monitor_months,
            "step_months": self.step_months,
        }


# ============================================
# Phase Config — ACTIVE（四階段主訓練設定）
# ============================================
@dataclass
class PhaseConfigV2:
    """
    四階段架構的時間設定 — ACTIVE（Source of Truth）
    
    ✅ 這是四階段訓練流程的主設定，由 FourPhaseTrainer 與 main.py 使用。
    
    Phase 1: Development (Rolling Training) - 18個月
    Phase 2: Champion Retraining (用全部 development 重訓)
    Phase 3: Policy Validation (Threshold Tuning) - 4個月
    Phase 4: Final Blind Holdout - 2個月
    """
    version: str = "2.0.0"
    
    # 總資料區間
    data_start: date = date(2024, 4, 1)
    data_end: date = date(2026, 3, 31)
    
    # 各階段長度（月）
    development_months: int = 18      # Phase 1: Rolling Training
    policy_months: int = 4            # Phase 3: Policy Validation
    holdout_months: int = 2           # Phase 4: Final Blind Holdout
    
    # Rolling Training 設定
    rolling_train_months: int = 4     # 每個 cycle 訓練長度
    rolling_monitor_months: int = 2   # 每個 cycle 監控長度
    rolling_step_months: int = 2      # 滑動步長
    
    # Imbalance 策略
    imbalance_strategies: List[str] = field(default_factory=lambda: [
        "scale_weight",                # Scale pos weight
        "class_weight",                # Class weight balanced
        "smote",                       # SMOTE oversampling
        "undersample",                 # Random undersampling
        "none",                        # 不處理
    ])
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "data_start": str(self.data_start),
            "data_end": str(self.data_end),
            "development_months": self.development_months,
            "policy_months": self.policy_months,
            "holdout_months": self.holdout_months,
            "rolling": {
                "train_months": self.rolling_train_months,
                "monitor_months": self.rolling_monitor_months,
                "step_months": self.rolling_step_months,
            },
            "imbalance_strategies": self.imbalance_strategies,
        }


# ============================================
# Threshold Grid Config
# ============================================
@dataclass
class ThresholdGridConfig:
    """
    Threshold Grid 設定
    
    用於 Phase 3: Policy Validation
    搜尋最佳 lower/upper threshold 組合
    """
    version: str = "1.0.0"
    
    # 預設 grid（針對高 imbalance 場景，分數集中在高位）
    lower_thresholds: List[float] = field(default_factory=lambda: [
        0.30, 0.40, 0.50, 0.60, 0.70, 0.80
    ])
    
    upper_thresholds: List[float] = field(default_factory=lambda: [
        0.70, 0.80, 0.85, 0.90, 0.95
    ])
    
    # 約束條件
    min_threshold_gap: float = 0.15   # lower 與 upper 最小間距
    
    # 指標權重（用於加權選擇最佳組合）
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "auc": 0.3,
        "f1_reject": 0.3,
        "zone_high_precision": 0.2,
        "zone_low_recall": 0.2,
    })
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "lower_thresholds": self.lower_thresholds,
            "upper_thresholds": self.upper_thresholds,
            "min_threshold_gap": self.min_threshold_gap,
            "metric_weights": self.metric_weights,
        }
    
    def get_threshold_grid(self) -> List[Dict[str, float]]:
        """產生有效的 threshold 組合"""
        grid = []
        for lower in self.lower_thresholds:
            for upper in self.upper_thresholds:
                if upper - lower >= self.min_threshold_gap:
                    grid.append({"lower": lower, "upper": upper})
        return grid


# ============================================
# Business Constraint Config (Phase 3 Threshold Policy)
# ============================================
@dataclass
class BusinessConstraintConfig:
    """
    業務約束條件 — 用於 Phase 3 Threshold Policy Selection
    
    控制自動選出的 threshold 組合必須滿足的業務規則，
    避免選出 manual review 過大或 low zone 過小的極端組合。
    
    Hard Constraints（不滿足直接排除）:
    - max_manual_review_ratio: 人工審核比例上限
    - min_auto_decision_rate: 自動決策率下限
    - min_low_zone_ratio: 低通過區最小佔比（確保模型有篩選能力）
    
    Soft Targets（納入 scoring，但不硬性排除）:
    - min_high_zone_precision: 高區域核准精確度目標
    - min_low_zone_reject_precision: 低區域婉拒精確度目標
    - target_auto_decision_rate: 自動決策率理想值（用於 scoring 加分）
    
    Scoring Weights（threshold 組合評分權重）:
    - w_precision: 精確度權重
    - w_auto_rate: 自動決策率權重
    - w_zone_balance: 區間平衡性權重
    """
    version: str = "1.0.0"
    
    # ── Hard Constraints ──
    max_manual_review_ratio: float = 0.15    # 人工審核不得超過 15%
    min_auto_decision_rate: float = 0.85     # 自動決策率至少 85%
    min_low_zone_ratio: float = 0.02         # low zone 至少 2%（確保有篩選力）
    
    # ── Soft Targets ──
    min_high_zone_precision: float = 0.95    # 高區核准精確度目標
    min_low_zone_reject_precision: float = 0.70  # 低區婉拒精確度目標
    target_auto_decision_rate: float = 0.90  # 理想自動決策率
    
    # ── Scoring Weights ──
    w_precision: float = 0.40               # 精確度在 threshold score 中的權重
    w_auto_rate: float = 0.35               # 自動決策率在 threshold score 中的權重
    w_zone_balance: float = 0.25            # 區間平衡性在 threshold score 中的權重
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "hard_constraints": {
                "max_manual_review_ratio": self.max_manual_review_ratio,
                "min_auto_decision_rate": self.min_auto_decision_rate,
                "min_low_zone_ratio": self.min_low_zone_ratio,
            },
            "soft_targets": {
                "min_high_zone_precision": self.min_high_zone_precision,
                "min_low_zone_reject_precision": self.min_low_zone_reject_precision,
                "target_auto_decision_rate": self.target_auto_decision_rate,
            },
            "scoring_weights": {
                "w_precision": self.w_precision,
                "w_auto_rate": self.w_auto_rate,
                "w_zone_balance": self.w_zone_balance,
            },
        }


# ============================================
# Champion Selection Config
# ============================================
@dataclass
class ChampionSelectionConfig:
    """
    Champion Strategy 選擇權重設定
    
    用於 Phase 1 的 select_champion_strategy()
    控制各指標在 overall_score 中的權重
    
    設計理念：
    - 信用風險模型重視「拒絕壞客戶」的能力 → f1_reject 最重要
    - 跨 cycle 穩定性是生產可靠性的基礎 → stability 懲罰加重
    - KS 反映區分好壞客戶的能力 → 比純 AUC 更有業務意義
    - CV AUC 作為基礎校驗，不應主導排名
    """
    version: str = "1.0.0"
    
    # ── 指標權重 ──
    w_cv_auc: float = 0.10              # CV AUC（基礎校驗，降低權重）
    w_monitor_auc: float = 0.20         # Monitor AUC（OOT 表現）
    w_monitor_f1_reject: float = 0.35   # Monitor F1_reject（核心指標，最高權重）
    w_monitor_ks: float = 0.15          # Monitor KS（區分力）
    w_stability_penalty: float = 0.20   # 穩定性懲罰（加重）
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "weights": {
                "w_cv_auc": self.w_cv_auc,
                "w_monitor_auc": self.w_monitor_auc,
                "w_monitor_f1_reject": self.w_monitor_f1_reject,
                "w_monitor_ks": self.w_monitor_ks,
                "w_stability_penalty": self.w_stability_penalty,
            },
        }


# ============================================
# Monitoring Config
# ============================================
@dataclass
class MonitoringConfigV2:
    """
    Production Monitoring 設定 (V2)
    
    包含 Retraining Trigger 閾值
    """
    version: str = "2.0.0"
    
    # Metric Triggers (低於閾值觸發 retraining)
    min_auc: float = 0.85               # ← 提高（原 0.75），AUC 低於 0.85 即需檢視
    min_f1_reject: float = 0.30         # ← 提高（原 0.20），reject 辨識力門檻
    max_score_psi: float = 0.25
    
    # Time Trigger
    retrain_interval_months: int = 6  # 每 6 個月強制 review
    
    # Warning Thresholds (不觸發 retraining，但要警示)
    warning_auc: float = 0.88           # ← 提高（原 0.80）
    warning_score_psi: float = 0.10
    warning_zone_shift: float = 0.10    # ← 降低（原 0.15），更敏感偵測 zone 變化
    
    # Default Zone Thresholds
    default_lower_threshold: float = 0.5   # ← 提高（原 0.4），搭配更嚴格的三區分配
    default_upper_threshold: float = 0.85  # ← 提高（原 0.7），避免 96%+ 自動核准
    
    # PSI 計算設定
    psi_n_bins: int = 10
    
    # Calibration 檢查
    calibration_tolerance: float = 0.05  # 預測與實際差異容忍度
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "metric_triggers": {
                "min_auc": self.min_auc,
                "min_f1_reject": self.min_f1_reject,
                "max_score_psi": self.max_score_psi,
            },
            "time_trigger": {
                "retrain_interval_months": self.retrain_interval_months,
            },
            "warning_thresholds": {
                "warning_auc": self.warning_auc,
                "warning_score_psi": self.warning_score_psi,
                "warning_zone_shift": self.warning_zone_shift,
            },
            "default_zone_thresholds": {
                "lower": self.default_lower_threshold,
                "upper": self.default_upper_threshold,
            },
        }


# ============================================
# Diagnostics Config
# ============================================
@dataclass
class DiagnosticsConfig:
    """
    Overfitting / Robustness Diagnostics 設定
    """
    version: str = "1.0.0"
    
    # Overfitting Detection
    max_train_test_gap: float = 0.05   # Train-Test AUC 最大差距
    max_train_oot_gap: float = 0.08    # Train-OOT AUC 最大差距
    
    # Stability Detection
    max_cycle_std: float = 0.03        # Cycle 間 AUC 標準差上限
    min_cycles_for_stable: int = 5     # 判斷穩定需要的最少 cycle 數
    
    # Calibration Check
    max_calibration_error: float = 0.05  # Brier Score 容忍上限
    
    # Score Distribution Shift
    max_score_distribution_shift: float = 0.15
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "overfitting": {
                "max_train_test_gap": self.max_train_test_gap,
                "max_train_oot_gap": self.max_train_oot_gap,
            },
            "stability": {
                "max_cycle_std": self.max_cycle_std,
                "min_cycles_for_stable": self.min_cycles_for_stable,
            },
            "calibration": {
                "max_calibration_error": self.max_calibration_error,
            },
        }


# ============================================
# Conservative Tuning Config
# ============================================
@dataclass
class TuningConfig:
    """
    保守型 Fine-Tuning 設定
    
    不做大範圍 grid search，只在已知 champion 的基礎上
    做小範圍候選組合搜尋，重點是：
    - 提升 F1_reject
    - 改善 calibration (Brier score)
    - 降低 overfitting / 提升 robustness
    
    同時保留 challenger (random_forest) 做比較
    """
    version: str = "1.0.0"
    
    # 是否啟用 tuning
    enable_tuning: bool = True
    
    # XGBoost 候選參數（保守方向：降 depth, 增 regularization）
    xgb_candidates: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "config_id": "xgb_baseline",
            "max_depth": 3, "min_child_weight": 20,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 1.0, "reg_lambda": 5.0,
            "learning_rate": 0.03, "n_estimators": 300,
            "gamma": 1.0, "max_delta_step": 1,
        },
        {
            "config_id": "xgb_conservative",
            "max_depth": 3, "min_child_weight": 30,
            "subsample": 0.7, "colsample_bytree": 0.6,
            "reg_alpha": 2.0, "reg_lambda": 8.0,
            "learning_rate": 0.02, "n_estimators": 400,
            "gamma": 1.5, "max_delta_step": 2,
        },
        {
            "config_id": "xgb_depth4",
            "max_depth": 4, "min_child_weight": 15,
            "subsample": 0.8, "colsample_bytree": 0.7,
            "reg_alpha": 1.0, "reg_lambda": 5.0,
            "learning_rate": 0.03, "n_estimators": 300,
            "gamma": 0.5, "max_delta_step": 1,
        },
        {
            "config_id": "xgb_shallow_strong_reg",
            "max_depth": 2, "min_child_weight": 25,
            "subsample": 0.7, "colsample_bytree": 0.6,
            "reg_alpha": 3.0, "reg_lambda": 10.0,
            "learning_rate": 0.03, "n_estimators": 500,
            "gamma": 2.0, "max_delta_step": 2,
        },
    ])
    
    # Random Forest 候選參數（challenger）
    rf_candidates: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "config_id": "rf_baseline",
            "n_estimators": 300, "max_depth": 6,
            "min_samples_split": 50, "min_samples_leaf": 20,
            "max_features": "sqrt",
        },
        {
            "config_id": "rf_conservative",
            "n_estimators": 500, "max_depth": 5,
            "min_samples_split": 80, "min_samples_leaf": 30,
            "max_features": "sqrt",
        },
    ])
    
    # Calibration 候選方法
    calibration_methods: List[str] = field(default_factory=lambda: [
        "isotonic", "sigmoid", "none"
    ])
    
    # Tuning 選模權重（重視 reject detection + calibration + stability）
    tuning_weights: Dict[str, float] = field(default_factory=lambda: {
        "w_holdout_f1_reject": 0.25,
        "w_monitor_f1_reject": 0.15,
        "w_holdout_brier": 0.15,        # 越低越好（取反）
        "w_monitor_brier": 0.10,        # 越低越好（取反）
        "w_holdout_auc": 0.10,
        "w_monitor_auc": 0.10,
        "w_stability_penalty": 0.15,
    })
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "enable_tuning": self.enable_tuning,
            "xgb_candidates": self.xgb_candidates,
            "rf_candidates": self.rf_candidates,
            "calibration_methods": self.calibration_methods,
            "tuning_weights": self.tuning_weights,
        }


# ============================================
# Model Training Config
# ============================================
@dataclass
class ModelTrainingConfig:
    """
    模型訓練設定
    """
    version: str = "2.0.0"
    
    # 模型類型
    model_type: str = "xgboost"
    
    # XGBoost 預設參數
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 150,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    })
    
    # Time-Based CV 設定
    cv_n_splits: int = 5
    cv_gap: int = 0  # train 與 test 之間的間隔月數
    
    # Early Stopping
    early_stopping_rounds: int = 20
    
    # Random Seed
    random_seed: int = 42
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "model_type": self.model_type,
            "xgboost_params": self.xgboost_params,
            "cv": {
                "n_splits": self.cv_n_splits,
                "gap": self.cv_gap,
            },
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_seed": self.random_seed,
        }


# ============================================
# Error Taxonomy
# ============================================
class ErrorSeverity:
    """錯誤嚴重程度"""
    FATAL = "FATAL"           # 必須停止 pipeline
    ERROR = "ERROR"           # 需要人工介入
    WARNING = "WARNING"       # 需要監控
    INFO = "INFO"             # 僅記錄


@dataclass
class ErrorDefinition:
    """錯誤定義"""
    code: str
    severity: str
    message_template: str
    action: str


ERROR_CATALOG = {
    # Schema 錯誤
    "SCH001": ErrorDefinition(
        code="SCH001",
        severity=ErrorSeverity.FATAL,
        message_template="缺少必要欄位: {columns}",
        action="停止 pipeline，通知資料來源修正"
    ),
    "SCH002": ErrorDefinition(
        code="SCH002",
        severity=ErrorSeverity.ERROR,
        message_template="欄位型別不符: {column} 預期 {expected} 實際 {actual}",
        action="嘗試型別轉換，失敗則停止"
    ),
    "SCH003": ErrorDefinition(
        code="SCH003",
        severity=ErrorSeverity.WARNING,
        message_template="發現未預期的新欄位: {columns}",
        action="記錄並繼續處理"
    ),
    
    # 資料品質錯誤
    "DQ001": ErrorDefinition(
        code="DQ001",
        severity=ErrorSeverity.ERROR,
        message_template="NULL 比例超過閾值: {column} = {ratio:.2%} > {threshold:.2%}",
        action="檢查資料來源，考慮是否繼續"
    ),
    "DQ002": ErrorDefinition(
        code="DQ002",
        severity=ErrorSeverity.ERROR,
        message_template="資料筆數異常: {count} < {min_count}",
        action="確認資料是否完整"
    ),
    "DQ003": ErrorDefinition(
        code="DQ003",
        severity=ErrorSeverity.WARNING,
        message_template="發現未預期的類別值: {column} = {values}",
        action="記錄並歸入 Other bucket"
    ),
    
    # 類別漂移錯誤
    "DR001": ErrorDefinition(
        code="DR001",
        severity=ErrorSeverity.WARNING,
        message_template="類別分布漂移: {column} PSI = {psi:.4f}",
        action="記錄並監控模型效能"
    ),
    "DR002": ErrorDefinition(
        code="DR002",
        severity=ErrorSeverity.WARNING,
        message_template="未見類別比例過高: {column} = {ratio:.2%}",
        action="考慮重新訓練 encoding"
    ),
    
    # Label 錯誤
    "LB001": ErrorDefinition(
        code="LB001",
        severity=ErrorSeverity.ERROR,
        message_template="Label 分布異常: 核准率 = {ratio:.2%}",
        action="確認業務流程是否變更"
    ),
}


# ============================================
# Pipeline Metadata
# ============================================
@dataclass
class PipelineRunMetadata:
    """Pipeline 執行元資料"""
    run_id: str
    run_timestamp: str
    config_version: str
    code_version: str = "unknown"  # 應從 git 取得
    input_path: str = ""
    output_path: str = ""
    parameters: Dict = field(default_factory=dict)
    
    # 執行統計
    row_counts: Dict[str, int] = field(default_factory=dict)
    null_summaries: Dict[str, Dict] = field(default_factory=dict)
    category_summaries: Dict[str, Dict] = field(default_factory=dict)
    validation_results: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "config_version": self.config_version,
            "code_version": self.code_version,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "parameters": self.parameters,
            "row_counts": self.row_counts,
            "null_summaries": self.null_summaries,
            "category_summaries": self.category_summaries,
            "validation_results": self.validation_results,
            "warnings": self.warnings,
            "errors": self.errors,
        }
    
    def save(self, path: Path):
        """儲存元資料"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Pipeline metadata saved to: {path}")


# ============================================
# Config Manager
# ============================================
class ConfigManager:
    """
    設定管理器
    
    包含兩組時間設定：
    - time_period  (TimePeriodConfig)  — LEGACY：Gold Layer 資料切分用
    - phase_config (PhaseConfigV2)     — ACTIVE：四階段訓練主設定
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.schema_contract = SchemaContract()
        self.data_quality = DataQualityThresholds()
        self.feature_encoding = FeatureEncodingConfig()
        self.feature_definition = FeatureDefinitionConfig()
        
        # LEGACY — 僅供 Gold Layer 向後相容
        self.time_period = TimePeriodConfig()
        
        # ACTIVE — 四階段訓練主設定
        self.phase_config = PhaseConfigV2()
        self.threshold_grid = ThresholdGridConfig()
        self.business_constraints = BusinessConstraintConfig()
        self.champion_selection = ChampionSelectionConfig()
        self.tuning = TuningConfig()
        self.monitoring = MonitoringConfigV2()
        self.diagnostics = DiagnosticsConfig()
        self.model_training = ModelTrainingConfig()
        
        if config_path and config_path.exists():
            self.load_from_file(config_path)
    
    def load_from_file(self, path: Path):
        """從 YAML/JSON 檔案載入設定"""
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # 更新設定（這裡可以擴展更多欄位）
        if 'time_period' in config:
            tp = config['time_period']
            self.time_period = TimePeriodConfig(
                data_start=date.fromisoformat(tp.get('data_start', '2024-04-01')),
                oot_start=date.fromisoformat(tp.get('oot_start', '2025-10-01')),
                train_months=tp.get('train_months', 4),
                monitor_months=tp.get('monitor_months', 2),
                step_months=tp.get('step_months', 2),
            )
        
        # V2 Config Loading
        if 'phase_config' in config:
            pc = config['phase_config']
            self.phase_config = PhaseConfigV2(
                data_start=date.fromisoformat(pc.get('data_start', '2024-04-01')),
                data_end=date.fromisoformat(pc.get('data_end', '2026-03-31')),
                development_months=pc.get('development_months', 18),
                policy_months=pc.get('policy_months', 4),
                holdout_months=pc.get('holdout_months', 2),
            )
        
        if 'threshold_grid' in config:
            tg = config['threshold_grid']
            self.threshold_grid = ThresholdGridConfig(
                lower_thresholds=tg.get('lower_thresholds', [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]),
                upper_thresholds=tg.get('upper_thresholds', [0.70, 0.80, 0.85, 0.90, 0.95]),
                min_threshold_gap=tg.get('min_threshold_gap', 0.15),
            )
        
        if 'business_constraints' in config:
            bc = config['business_constraints']
            hard = bc.get('hard_constraints', {})
            soft = bc.get('soft_targets', {})
            sw = bc.get('scoring_weights', {})
            self.business_constraints = BusinessConstraintConfig(
                max_manual_review_ratio=hard.get('max_manual_review_ratio', 0.15),
                min_auto_decision_rate=hard.get('min_auto_decision_rate', 0.85),
                min_low_zone_ratio=hard.get('min_low_zone_ratio', 0.02),
                min_high_zone_precision=soft.get('min_high_zone_precision', 0.95),
                min_low_zone_reject_precision=soft.get('min_low_zone_reject_precision', 0.70),
                target_auto_decision_rate=soft.get('target_auto_decision_rate', 0.90),
                w_precision=sw.get('w_precision', 0.40),
                w_auto_rate=sw.get('w_auto_rate', 0.35),
                w_zone_balance=sw.get('w_zone_balance', 0.25),
            )
        
        if 'champion_selection' in config:
            cs = config['champion_selection']
            weights = cs.get('weights', {})
            self.champion_selection = ChampionSelectionConfig(
                w_cv_auc=weights.get('w_cv_auc', 0.10),
                w_monitor_auc=weights.get('w_monitor_auc', 0.20),
                w_monitor_f1_reject=weights.get('w_monitor_f1_reject', 0.35),
                w_monitor_ks=weights.get('w_monitor_ks', 0.15),
                w_stability_penalty=weights.get('w_stability_penalty', 0.20),
            )
        
        if 'tuning' in config:
            tc = config['tuning']
            self.tuning = TuningConfig(
                enable_tuning=tc.get('enable_tuning', True),
                xgb_candidates=tc.get('xgb_candidates', TuningConfig().xgb_candidates),
                rf_candidates=tc.get('rf_candidates', TuningConfig().rf_candidates),
                calibration_methods=tc.get('calibration_methods', ["isotonic", "sigmoid", "none"]),
                tuning_weights=tc.get('tuning_weights', TuningConfig().tuning_weights),
            )
        
        if 'monitoring' in config:
            mc = config['monitoring']
            self.monitoring = MonitoringConfigV2(
                min_auc=mc.get('min_auc', 0.85),
                min_f1_reject=mc.get('min_f1_reject', 0.30),
                max_score_psi=mc.get('max_score_psi', 0.25),
                retrain_interval_months=mc.get('retrain_interval_months', 6),
            )
        
        if 'diagnostics' in config:
            dc = config['diagnostics']
            self.diagnostics = DiagnosticsConfig(
                max_train_test_gap=dc.get('max_train_test_gap', 0.05),
                max_train_oot_gap=dc.get('max_train_oot_gap', 0.08),
                max_cycle_std=dc.get('max_cycle_std', 0.03),
            )
        
        logger.info(f"Config loaded from: {path}")
    
    def save_to_file(self, path: Path):
        """儲存設定至檔案"""
        config = {
            "version": CONFIG_VERSION,
            "effective_date": CONFIG_EFFECTIVE_DATE,
            "time_period": self.time_period.to_dict(),
            "feature_definition": {
                "numeric_features_to_scale": self.feature_definition.numeric_features_to_scale,
                "ordinal_features": self.feature_definition.ordinal_features,
                "binary_features": self.feature_definition.binary_features,
                "high_cardinality_features": self.feature_definition.high_cardinality_features,
                "target_column": self.feature_definition.target_column,
                "key_columns": self.feature_definition.key_columns,
            },
            # V2 Config
            "phase_config": self.phase_config.to_dict(),
            "threshold_grid": self.threshold_grid.to_dict(),
            "business_constraints": self.business_constraints.to_dict(),
            "champion_selection": self.champion_selection.to_dict(),
            "tuning": self.tuning.to_dict(),
            "monitoring": self.monitoring.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "model_training": self.model_training.to_dict(),
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            if path.suffix == '.yaml' or path.suffix == '.yml':
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            else:
                json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Config saved to: {path}")
    
    def get_all_feature_names(self) -> List[str]:
        """取得所有特徵名稱"""
        features = []
        features.extend(self.feature_definition.numeric_features_to_scale)
        features.extend(self.feature_definition.ordinal_features)
        features.extend(self.feature_definition.binary_features)
        features.extend([
            f"{col}_頻率" 
            for col in self.feature_definition.high_cardinality_features
        ])
        return features
    
    def get_v2_config_summary(self) -> Dict:
        """取得 V2 設定摘要"""
        return {
            "phase_config": self.phase_config.to_dict(),
            "threshold_grid": self.threshold_grid.to_dict(),
            "business_constraints": self.business_constraints.to_dict(),
            "champion_selection": self.champion_selection.to_dict(),
            "tuning": self.tuning.to_dict(),
            "monitoring": self.monitoring.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "model_training": self.model_training.to_dict(),
        }


# 建立預設 config 實例
default_config = ConfigManager()
