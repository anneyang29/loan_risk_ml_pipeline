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
        "婚姻狀況": ["已婚", "未婚", "離婚", None, "Missing"],
        "教育程度": [
            "國中", "高中", "專科", "大學", "碩士以上", "其他", 
            None, "Missing"
        ],
        "授信結果": [
            "APP(核准)", "WTCD(婉拒)", 
            # 其他可能的狀態（需要監控）
            "PEND(審核中)", "CANCEL(取消)",
        ],
        "動產設定": ["Y", "N", None],
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
        "碩士以上": 6,
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
# Time Period Config
# ============================================
@dataclass
class TimePeriodConfig:
    """時間區間設定"""
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
    """設定管理器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.schema_contract = SchemaContract()
        self.data_quality = DataQualityThresholds()
        self.feature_encoding = FeatureEncodingConfig()
        self.feature_definition = FeatureDefinitionConfig()
        self.time_period = TimePeriodConfig()
        
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


# 建立預設 config 實例
default_config = ConfigManager()
