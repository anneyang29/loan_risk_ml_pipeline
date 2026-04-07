"""
Schema Validation Module
========================
資料契約驗證、資料品質檢查

提供：
- Required columns check
- Schema type check
- Allowed category values check
- Null ratio threshold check
- Row count sanity check
- Unexpected category drift alert
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType, IntegerType, DoubleType, DateType, 
    LongType, FloatType, TimestampType
)

from .config import (
    ConfigManager, default_config,
    ErrorSeverity, ERROR_CATALOG, ErrorDefinition
)

logger = logging.getLogger(__name__)


# ============================================
# Validation Result Classes
# ============================================
@dataclass
class ValidationResult:
    """驗證結果"""
    is_valid: bool
    error_code: Optional[str] = None
    severity: str = ErrorSeverity.INFO
    message: str = ""
    details: Dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """驗證報告"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    overall_status: str = "PASS"  # PASS, WARNING, FAIL
    fatal_errors: List[ValidationResult] = field(default_factory=list)
    errors: List[ValidationResult] = field(default_factory=list)
    warnings: List[ValidationResult] = field(default_factory=list)
    info: List[ValidationResult] = field(default_factory=list)
    
    # 統計摘要
    row_count: int = 0
    column_count: int = 0
    null_summary: Dict[str, float] = field(default_factory=dict)
    category_summary: Dict[str, Dict] = field(default_factory=dict)
    
    def add_result(self, result: ValidationResult):
        """新增驗證結果"""
        if result.severity == ErrorSeverity.FATAL:
            self.fatal_errors.append(result)
            self.overall_status = "FAIL"
        elif result.severity == ErrorSeverity.ERROR:
            self.errors.append(result)
            if self.overall_status != "FAIL":
                self.overall_status = "FAIL"
        elif result.severity == ErrorSeverity.WARNING:
            self.warnings.append(result)
            if self.overall_status == "PASS":
                self.overall_status = "WARNING"
        else:
            self.info.append(result)
    
    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "summary": {
                "row_count": self.row_count,
                "column_count": self.column_count,
                "fatal_count": len(self.fatal_errors),
                "error_count": len(self.errors),
                "warning_count": len(self.warnings),
            },
            "fatal_errors": [
                {"code": r.error_code, "message": r.message, "details": r.details}
                for r in self.fatal_errors
            ],
            "errors": [
                {"code": r.error_code, "message": r.message, "details": r.details}
                for r in self.errors
            ],
            "warnings": [
                {"code": r.error_code, "message": r.message, "details": r.details}
                for r in self.warnings
            ],
            "null_summary": self.null_summary,
            "category_summary": self.category_summary,
        }
    
    def log_summary(self):
        """輸出摘要日誌"""
        logger.info("=" * 60)
        logger.info(f"Validation Report - Status: {self.overall_status}")
        logger.info("=" * 60)
        logger.info(f"Row Count: {self.row_count:,}")
        logger.info(f"Column Count: {self.column_count}")
        logger.info(f"Fatal Errors: {len(self.fatal_errors)}")
        logger.info(f"Errors: {len(self.errors)}")
        logger.info(f"Warnings: {len(self.warnings)}")
        
        if self.fatal_errors:
            logger.error("FATAL ERRORS:")
            for r in self.fatal_errors:
                logger.error(f"  [{r.error_code}] {r.message}")
        
        if self.errors:
            logger.error("ERRORS:")
            for r in self.errors:
                logger.error(f"  [{r.error_code}] {r.message}")
        
        if self.warnings:
            logger.warning("WARNINGS:")
            for r in self.warnings:
                logger.warning(f"  [{r.error_code}] {r.message}")


# ============================================
# Schema Validator
# ============================================
class SchemaValidator:
    """Schema 驗證器"""
    
    def __init__(self, config: ConfigManager = None):
        self.config = config or default_config
    
    def validate_required_columns(
        self, 
        df: DataFrame, 
        required_columns: List[str] = None
    ) -> ValidationResult:
        """
        驗證必要欄位
        
        Args:
            df: DataFrame
            required_columns: 必要欄位列表（若為 None 則使用設定）
        """
        if required_columns is None:
            required_columns = self.config.schema_contract.bronze_required_columns
        
        actual_columns = set(df.columns)
        missing_columns = [c for c in required_columns if c not in actual_columns]
        
        if missing_columns:
            error_def = ERROR_CATALOG["SCH001"]
            return ValidationResult(
                is_valid=False,
                error_code="SCH001",
                severity=error_def.severity,
                message=error_def.message_template.format(columns=missing_columns),
                details={"missing_columns": missing_columns}
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ErrorSeverity.INFO,
            message=f"所有 {len(required_columns)} 個必要欄位都存在"
        )
    
    def validate_unexpected_columns(
        self, 
        df: DataFrame,
        expected_columns: List[str] = None
    ) -> ValidationResult:
        """檢查是否有未預期的新欄位"""
        if expected_columns is None:
            expected_columns = self.config.schema_contract.bronze_required_columns
        
        actual_columns = set(df.columns)
        expected_set = set(expected_columns)
        unexpected = actual_columns - expected_set
        
        # 排除系統欄位
        system_columns = {"bronze_load_timestamp", "silver_process_timestamp"}
        unexpected = unexpected - system_columns
        
        if unexpected:
            error_def = ERROR_CATALOG["SCH003"]
            return ValidationResult(
                is_valid=True,  # 不是致命錯誤
                error_code="SCH003",
                severity=error_def.severity,
                message=error_def.message_template.format(columns=list(unexpected)),
                details={"unexpected_columns": list(unexpected)}
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ErrorSeverity.INFO,
            message="無未預期的欄位"
        )
    
    def validate_column_types(
        self, 
        df: DataFrame,
        type_mapping: Dict[str, str] = None
    ) -> List[ValidationResult]:
        """驗證欄位型別"""
        if type_mapping is None:
            type_mapping = self.config.schema_contract.column_types
        
        results = []
        schema_dict = {f.name: f.dataType for f in df.schema.fields}
        
        type_class_mapping = {
            "string": (StringType,),
            "integer": (IntegerType, LongType),
            "double": (DoubleType, FloatType),
            "date": (DateType, TimestampType),
        }
        
        for col, expected_type in type_mapping.items():
            if col not in schema_dict:
                continue
            
            actual_type = schema_dict[col]
            expected_classes = type_class_mapping.get(expected_type, (StringType,))
            
            if not isinstance(actual_type, expected_classes):
                error_def = ERROR_CATALOG["SCH002"]
                results.append(ValidationResult(
                    is_valid=False,
                    error_code="SCH002",
                    severity=error_def.severity,
                    message=error_def.message_template.format(
                        column=col,
                        expected=expected_type,
                        actual=str(actual_type)
                    ),
                    details={
                        "column": col,
                        "expected_type": expected_type,
                        "actual_type": str(actual_type)
                    }
                ))
        
        if not results:
            results.append(ValidationResult(
                is_valid=True,
                severity=ErrorSeverity.INFO,
                message="所有欄位型別符合預期"
            ))
        
        return results


# ============================================
# Data Quality Validator
# ============================================
class DataQualityValidator:
    """資料品質驗證器"""
    
    def __init__(self, config: ConfigManager = None):
        self.config = config or default_config
    
    def calculate_null_ratios(self, df: DataFrame) -> Dict[str, float]:
        """計算所有欄位的 NULL 比例"""
        total = df.count()
        if total == 0:
            return {}
        
        null_ratios = {}
        
        # 使用單次 aggregation 計算所有欄位的 null count
        agg_exprs = [
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}_null")
            for c in df.columns
        ]
        
        null_counts = df.agg(*agg_exprs).collect()[0]
        
        for c in df.columns:
            null_count = null_counts[f"{c}_null"]
            null_ratios[c] = null_count / total if null_count else 0.0
        
        return null_ratios
    
    def validate_null_ratios(
        self, 
        df: DataFrame,
        thresholds: Dict[str, float] = None
    ) -> List[ValidationResult]:
        """驗證 NULL 比例"""
        if thresholds is None:
            thresholds = self.config.data_quality.null_ratio_thresholds
        
        default_threshold = thresholds.get("default", 0.30)
        null_ratios = self.calculate_null_ratios(df)
        results = []
        
        for col, ratio in null_ratios.items():
            threshold = thresholds.get(col, default_threshold)
            
            if ratio > threshold:
                error_def = ERROR_CATALOG["DQ001"]
                results.append(ValidationResult(
                    is_valid=False,
                    error_code="DQ001",
                    severity=error_def.severity,
                    message=error_def.message_template.format(
                        column=col, ratio=ratio, threshold=threshold
                    ),
                    details={
                        "column": col,
                        "null_ratio": ratio,
                        "threshold": threshold
                    }
                ))
        
        if not any(not r.is_valid for r in results):
            results.append(ValidationResult(
                is_valid=True,
                severity=ErrorSeverity.INFO,
                message="所有欄位 NULL 比例在閾值內"
            ))
        
        return results
    
    def validate_row_count(
        self, 
        df: DataFrame,
        min_rows: int = None
    ) -> ValidationResult:
        """驗證資料筆數"""
        if min_rows is None:
            min_rows = self.config.data_quality.row_count_thresholds.get("min_total_rows", 1000)
        
        count = df.count()
        
        if count < min_rows:
            error_def = ERROR_CATALOG["DQ002"]
            return ValidationResult(
                is_valid=False,
                error_code="DQ002",
                severity=error_def.severity,
                message=error_def.message_template.format(count=count, min_count=min_rows),
                details={"actual_count": count, "min_count": min_rows}
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ErrorSeverity.INFO,
            message=f"資料筆數 {count:,} >= {min_rows:,}"
        )
    
    def validate_category_values(
        self, 
        df: DataFrame,
        column: str,
        allowed_values: List[str] = None
    ) -> ValidationResult:
        """驗證類別值"""
        if allowed_values is None:
            allowed_values = self.config.schema_contract.allowed_values.get(column, [])
        
        if not allowed_values:
            return ValidationResult(
                is_valid=True,
                severity=ErrorSeverity.INFO,
                message=f"{column} 無設定允許值檢查"
            )
        
        # 取得實際的 distinct values
        actual_values = [
            row[column] for row in df.select(column).distinct().collect()
        ]
        
        allowed_set = set(allowed_values)
        unexpected_values = [v for v in actual_values if v not in allowed_set]
        
        if unexpected_values:
            error_def = ERROR_CATALOG["DQ003"]
            return ValidationResult(
                is_valid=True,  # 不是致命錯誤，但需要處理
                error_code="DQ003",
                severity=error_def.severity,
                message=error_def.message_template.format(
                    column=column, values=unexpected_values
                ),
                details={
                    "column": column,
                    "unexpected_values": unexpected_values,
                    "allowed_values": allowed_values
                }
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ErrorSeverity.INFO,
            message=f"{column} 所有類別值都在允許範圍內"
        )
    
    def validate_numeric_range(
        self, 
        df: DataFrame,
        column: str,
        min_val: float = None,
        max_val: float = None
    ) -> ValidationResult:
        """驗證數值範圍"""
        ranges = self.config.schema_contract.numeric_ranges.get(column, {})
        
        if min_val is None:
            min_val = ranges.get("min")
        if max_val is None:
            max_val = ranges.get("max")
        
        if min_val is None and max_val is None:
            return ValidationResult(
                is_valid=True,
                severity=ErrorSeverity.INFO,
                message=f"{column} 無設定範圍檢查"
            )
        
        # 計算實際範圍
        stats = df.select(
            F.min(column).alias("min"),
            F.max(column).alias("max")
        ).collect()[0]
        
        actual_min = stats["min"]
        actual_max = stats["max"]
        
        violations = []
        if min_val is not None and actual_min is not None and actual_min < min_val:
            violations.append(f"min={actual_min} < {min_val}")
        if max_val is not None and actual_max is not None and actual_max > max_val:
            violations.append(f"max={actual_max} > {max_val}")
        
        if violations:
            return ValidationResult(
                is_valid=True,  # 警告而非錯誤
                error_code="DQ004",
                severity=ErrorSeverity.WARNING,
                message=f"{column} 數值範圍異常: {', '.join(violations)}",
                details={
                    "column": column,
                    "actual_min": actual_min,
                    "actual_max": actual_max,
                    "expected_min": min_val,
                    "expected_max": max_val
                }
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ErrorSeverity.INFO,
            message=f"{column} 數值範圍正常"
        )


# ============================================
# Label Validator
# ============================================
class LabelValidator:
    """目標變數驗證器"""
    
    def __init__(self, config: ConfigManager = None):
        self.config = config or default_config
    
    def validate_label_distribution(
        self, 
        df: DataFrame,
        label_column: str = "授信結果",
        target_mapping: Dict[str, int] = None
    ) -> ValidationResult:
        """驗證 Label 分布"""
        if target_mapping is None:
            target_mapping = self.config.feature_encoding.target_mapping
        
        thresholds = self.config.data_quality.label_thresholds
        
        # 計算各類別數量
        dist = df.groupBy(label_column).count().collect()
        dist_dict = {row[label_column]: row["count"] for row in dist}
        
        total = sum(dist_dict.values())
        if total == 0:
            return ValidationResult(
                is_valid=False,
                error_code="LB002",
                severity=ErrorSeverity.FATAL,
                message="無任何 Label 資料",
                details={}
            )
        
        # 計算核准/拒絕比例
        positive_key = [k for k, v in target_mapping.items() if v == 1]
        positive_count = sum(dist_dict.get(k, 0) for k in positive_key)
        positive_ratio = positive_count / total
        
        # 檢查其他狀態（非核准/拒絕）
        known_labels = set(target_mapping.keys())
        other_labels = {k: v for k, v in dist_dict.items() if k not in known_labels and k is not None}
        other_ratio = sum(other_labels.values()) / total if other_labels else 0
        
        details = {
            "distribution": dist_dict,
            "positive_ratio": positive_ratio,
            "other_labels": other_labels,
            "other_ratio": other_ratio,
        }
        
        # 檢查核准率是否在合理範圍
        min_ratio = thresholds.get("min_positive_ratio", 0.05)
        max_ratio = thresholds.get("max_positive_ratio", 0.95)
        
        if positive_ratio < min_ratio or positive_ratio > max_ratio:
            error_def = ERROR_CATALOG["LB001"]
            return ValidationResult(
                is_valid=False,
                error_code="LB001",
                severity=error_def.severity,
                message=error_def.message_template.format(ratio=positive_ratio),
                details=details
            )
        
        # 警告：其他狀態比例
        if other_ratio > 0.05:
            return ValidationResult(
                is_valid=True,
                error_code="LB003",
                severity=ErrorSeverity.WARNING,
                message=f"非核准/拒絕狀態比例 = {other_ratio:.2%}，標籤: {list(other_labels.keys())}",
                details=details
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ErrorSeverity.INFO,
            message=f"Label 分布正常: 核准率 = {positive_ratio:.2%}",
            details=details
        )


# ============================================
# Comprehensive Validator
# ============================================
class DataValidator:
    """綜合資料驗證器"""
    
    def __init__(self, config: ConfigManager = None):
        self.config = config or default_config
        self.schema_validator = SchemaValidator(config)
        self.quality_validator = DataQualityValidator(config)
        self.label_validator = LabelValidator(config)
    
    def validate_bronze(self, df: DataFrame) -> ValidationReport:
        """Bronze Layer 完整驗證"""
        report = ValidationReport()
        
        # 基本資訊
        report.row_count = df.count()
        report.column_count = len(df.columns)
        
        # 1. Required columns
        result = self.schema_validator.validate_required_columns(df)
        report.add_result(result)
        
        # 如果缺少必要欄位，直接返回
        if result.severity == ErrorSeverity.FATAL:
            report.log_summary()
            return report
        
        # 2. Unexpected columns
        result = self.schema_validator.validate_unexpected_columns(df)
        report.add_result(result)
        
        # 3. Row count
        result = self.quality_validator.validate_row_count(df)
        report.add_result(result)
        
        # 4. NULL ratios
        report.null_summary = self.quality_validator.calculate_null_ratios(df)
        results = self.quality_validator.validate_null_ratios(df)
        for result in results:
            report.add_result(result)
        
        # 5. Category values
        for col in ["性別", "婚姻狀況", "教育程度", "授信結果", "動產設定"]:
            if col in df.columns:
                result = self.quality_validator.validate_category_values(df, col)
                report.add_result(result)
        
        # 6. Label distribution
        if "授信結果" in df.columns:
            result = self.label_validator.validate_label_distribution(df)
            report.add_result(result)
        
        report.log_summary()
        return report
    
    def validate_silver(self, df: DataFrame) -> ValidationReport:
        """Silver Layer 完整驗證"""
        report = ValidationReport()
        
        report.row_count = df.count()
        report.column_count = len(df.columns)
        
        # 1. Row count
        result = self.quality_validator.validate_row_count(df)
        report.add_result(result)
        
        # 2. NULL ratios for derived features
        report.null_summary = self.quality_validator.calculate_null_ratios(df)
        
        # 3. Check encoded columns exist
        expected_encoded = [
            "授信結果_二元",
            "性別_二元",
            "婚姻狀況_二元",
            "教育程度_序位",
            "月所得_序位",
            "年齡組_序位",
        ]
        missing_encoded = [c for c in expected_encoded if c not in df.columns]
        
        if missing_encoded:
            report.add_result(ValidationResult(
                is_valid=False,
                error_code="SLV001",
                severity=ErrorSeverity.ERROR,
                message=f"缺少編碼後欄位: {missing_encoded}",
                details={"missing_columns": missing_encoded}
            ))
        
        # 4. Validate label binary column
        if "授信結果_二元" in df.columns:
            invalid_labels = df.filter(
                ~F.col("授信結果_二元").isin([0, 1]) & 
                F.col("授信結果_二元").isNotNull()
            ).count()
            
            if invalid_labels > 0:
                report.add_result(ValidationResult(
                    is_valid=False,
                    error_code="SLV002",
                    severity=ErrorSeverity.ERROR,
                    message=f"授信結果_二元 含有非 0/1 值: {invalid_labels} 筆",
                    details={"invalid_count": invalid_labels}
                ))
        
        report.log_summary()
        return report
    
    def validate_gold(
        self, 
        df_dev: DataFrame, 
        df_oot: DataFrame
    ) -> ValidationReport:
        """Gold Layer 完整驗證"""
        report = ValidationReport()
        
        report.row_count = df_dev.count() + df_oot.count()
        
        # 1. Dev/OOT 基本檢查
        dev_count = df_dev.count()
        oot_count = df_oot.count()
        
        if dev_count == 0:
            report.add_result(ValidationResult(
                is_valid=False,
                error_code="GLD001",
                severity=ErrorSeverity.FATAL,
                message="Development 資料集為空",
                details={}
            ))
        
        if oot_count == 0:
            report.add_result(ValidationResult(
                is_valid=False,
                error_code="GLD002",
                severity=ErrorSeverity.WARNING,
                message="OOT 資料集為空",
                details={}
            ))
        
        # 2. NULL check in final features
        all_features = (
            self.config.feature_definition.ordinal_features +
            self.config.feature_definition.binary_features
        )
        
        for col in all_features:
            if col in df_dev.columns:
                null_count = df_dev.filter(F.col(col).isNull()).count()
                if null_count > 0:
                    report.add_result(ValidationResult(
                        is_valid=False,
                        error_code="GLD003",
                        severity=ErrorSeverity.ERROR,
                        message=f"[Dev] {col} 仍有 {null_count} 個 NULL",
                        details={"column": col, "null_count": null_count}
                    ))
        
        report.log_summary()
        return report


# ============================================
# Helper Functions
# ============================================
def validate_dataframe(
    df: DataFrame,
    layer: str = "bronze",
    config: ConfigManager = None
) -> Tuple[bool, ValidationReport]:
    """
    便捷函數：驗證 DataFrame
    
    Args:
        df: DataFrame
        layer: "bronze", "silver", "gold"
        config: 設定管理器
    
    Returns:
        (is_valid, report)
    """
    validator = DataValidator(config)
    
    if layer == "bronze":
        report = validator.validate_bronze(df)
    elif layer == "silver":
        report = validator.validate_silver(df)
    else:
        raise ValueError(f"Unknown layer: {layer}")
    
    is_valid = report.overall_status != "FAIL"
    return is_valid, report


def check_schema_compatibility(
    df: DataFrame,
    required_columns: List[str],
    raise_on_failure: bool = True
) -> bool:
    """
    快速檢查 schema 相容性
    
    Args:
        df: DataFrame
        required_columns: 必要欄位
        raise_on_failure: 失敗時是否拋出異常
    
    Returns:
        是否相容
    """
    missing = [c for c in required_columns if c not in df.columns]
    
    if missing:
        message = f"缺少必要欄位: {missing}"
        logger.error(message)
        
        if raise_on_failure:
            raise ValueError(message)
        return False
    
    return True
