"""
Utils Package
=============
Bronze / Silver / Gold 資料處理函數
企業級資料治理模組
模型訓練模組
"""

# Core pipelines
from .data_processing_bronze_table import run_bronze_pipeline
from .data_processing_silver_table import run_silver_pipeline
from .data_processing_gold_table import run_gold_pipeline

# Configuration
from .config import (
    ConfigManager,
    default_config,
    SchemaContract,
    DataQualityThresholds,
    FeatureEncodingConfig,
    FeatureDefinitionConfig,
    TimePeriodConfig,
    CONFIG_VERSION,
)

# Schema Validation
from .schema_validation import (
    DataValidator,
    SchemaValidator,
    DataQualityValidator,
    LabelValidator,
    ValidationReport,
    ValidationResult,
    validate_dataframe,
    check_schema_compatibility,
)

# Transformation Artifacts
from .transformation_artifacts import (
    ArtifactManager,
    TransformationPackage,
    ScalerArtifact,
    FrequencyEncodingArtifact,
    OrdinalEncodingArtifact,
    save_transformation_artifacts,
    load_transformation_artifacts,
)

# Monitoring
from .monitoring import (
    DriftMonitor,
    DriftReport,
    DriftResult,
    AuditLogger,
    AuditRecord,
    check_unseen_categories,
    compute_feature_statistics,
    calculate_psi,
    ProductionMonitor,
    ProductionMonitorConfig,
    MonitoringResult,
    check_retrain_trigger,
)

# Model Registry
from .model_registry import (
    ModelRecord,
    ModelRegistry,
    RollingTrainedModelRecord,
    ExtendedModelRegistry,
    load_best_model_name,
)

# Rolling Training
from .rolling_trainer import (
    RollingTrainer,
    WindowDefinition,
    CycleResult,
    StrategyResult,
    ZoneSummary,
    TimeBasedCV,
    run_rolling_training,
    aggregate_rolling_results,
    select_champion_strategy,
    retrain_final_champion,
    infer_on_oot,
    assign_score_zone,
    evaluate_zone_performance,
    generate_retraining_window,
)

__all__ = [
    # Pipelines
    "run_bronze_pipeline",
    "run_silver_pipeline",
    "run_gold_pipeline",
    # Config
    "ConfigManager",
    "default_config",
    "SchemaContract",
    "DataQualityThresholds",
    "FeatureEncodingConfig",
    "FeatureDefinitionConfig",
    "TimePeriodConfig",
    "CONFIG_VERSION",
    # Validation
    "DataValidator",
    "SchemaValidator",
    "DataQualityValidator",
    "LabelValidator",
    "ValidationReport",
    "ValidationResult",
    "validate_dataframe",
    "check_schema_compatibility",
    # Artifacts
    "ArtifactManager",
    "TransformationPackage",
    "ScalerArtifact",
    "FrequencyEncodingArtifact",
    "OrdinalEncodingArtifact",
    "save_transformation_artifacts",
    "load_transformation_artifacts",
    # Monitoring
    "DriftMonitor",
    "DriftReport",
    "DriftResult",
    "AuditLogger",
    "AuditRecord",
    "check_unseen_categories",
    "compute_feature_statistics",
    "calculate_psi",
    "ProductionMonitor",
    "ProductionMonitorConfig",
    "MonitoringResult",
    "check_retrain_trigger",
    # Model Registry
    "ModelRecord",
    "ModelRegistry",
    "RollingTrainedModelRecord",
    "ExtendedModelRegistry",
    "load_best_model_name",
    # Rolling Training
    "RollingTrainer",
    "WindowDefinition",
    "CycleResult",
    "StrategyResult",
    "ZoneSummary",
    "TimeBasedCV",
    "run_rolling_training",
    "aggregate_rolling_results",
    "select_champion_strategy",
    "retrain_final_champion",
    "infer_on_oot",
    "assign_score_zone",
    "evaluate_zone_performance",
    "generate_retraining_window",
]
