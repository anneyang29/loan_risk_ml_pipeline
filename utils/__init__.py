"""
Utils Package
=============
信用風險 ML Pipeline 模組

═══════════════════════════════════════════
Primary Architecture（主流程模組）
═══════════════════════════════════════════
- data_processing_bronze_table: Bronze Layer（原始資料載入）
- data_processing_silver_table: Silver Layer（清理與特徵工程）
- data_processing_gold_table: Gold Layer（特徵準備與切分）
- four_phase_trainer: 四階段訓練架構（唯一主訓練模組）
- production_monitor: Production Monitoring / Retraining Trigger
- config: 集中式設定管理

═══════════════════════════════════════════
Supporting Modules（輔助模組）
═══════════════════════════════════════════
- monitoring: 基礎 drift / audit / PSI 工具
- model_registry: 模型版本管理
- schema_validation: 資料契約驗證
- transformation_artifacts: Transformation 一致性管理

═══════════════════════════════════════════
Legacy（保留供向後相容，主流程不依賴）
═══════════════════════════════════════════
- rolling_trainer: 舊版 Rolling Trainer，已由 four_phase_trainer 取代
"""

# ──────────────────────────────────────────
# Primary: Data Pipelines (function-based)
# ──────────────────────────────────────────
from .data_processing_bronze_table import run_bronze_pipeline
from .data_processing_silver_table import run_silver_pipeline
from .data_processing_gold_table import run_gold_pipeline

# ──────────────────────────────────────────
# Primary: Four-Phase Trainer（唯一主訓練架構）
# ──────────────────────────────────────────
from .four_phase_trainer import (
    FourPhaseTrainer,
    PhaseConfig,
    ImbalanceHandler,
    MetricsCalculator,
    ThresholdPolicyResult,
    DiagnosticsSummary,
    DecileSummary,
    split_development_policy_holdout,
    evaluate_threshold_grid,
    run_four_phase_pipeline,
)

# ──────────────────────────────────────────
# Primary: Production Monitor
# ──────────────────────────────────────────
from .production_monitor import (
    ProductionMonitoringConfig,
    ProductionMonitoringResult,
    ProductionBatchResult,
    ProductionMonitor as AdvancedProductionMonitor,
    score_production_batch,
    check_retrain_trigger as check_retrain_trigger_advanced,
    generate_retraining_data_window,
    calculate_numeric_psi,
)

# ──────────────────────────────────────────
# Primary: Configuration
# ──────────────────────────────────────────
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

# ──────────────────────────────────────────
# Supporting: Schema Validation
# ──────────────────────────────────────────
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

# ──────────────────────────────────────────
# Supporting: Transformation Artifacts
# ──────────────────────────────────────────
from .transformation_artifacts import (
    ArtifactManager,
    TransformationPackage,
    ScalerArtifact,
    FrequencyEncodingArtifact,
    OrdinalEncodingArtifact,
    save_transformation_artifacts,
    load_transformation_artifacts,
)

# ──────────────────────────────────────────
# Supporting: Monitoring (V1 - 基礎 drift / audit / PSI)
# ──────────────────────────────────────────
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

# ──────────────────────────────────────────
# Supporting: Model Registry
# ──────────────────────────────────────────
from .model_registry import (
    ModelRecord,
    ModelRegistry,
    RollingTrainedModelRecord,
    ExtendedModelRegistry,
    load_best_model_name,
)

# ──────────────────────────────────────────
# Supporting: Baseline Manager
# ──────────────────────────────────────────
from .baseline_manager import (
    BaselineManager,
    BaselineRecord,
)

# ──────────────────────────────────────────
# Legacy: Rolling Trainer（已由 FourPhaseTrainer 取代）
# ⚠️ 不建議新開發直接 import 這些模組
# ⚠️ 僅保留供 notebook / 分析腳本 / CI 向後相容
# ⚠️ 主流程請使用上方 Primary 區塊的模組
# ──────────────────────────────────────────
from .rolling_trainer import (
    RollingTrainer,
    WindowDefinition,
    CycleResult,
    StrategyResult,
    ZoneSummary,
    TimeBasedCV,
    assign_score_zone,
    evaluate_zone_performance,
    generate_retraining_window,
    run_rolling_training_pipeline,
)


__all__ = [
    # ── Primary: Data Pipelines ──
    "run_bronze_pipeline",
    "run_silver_pipeline",
    "run_gold_pipeline",
    # ── Primary: Four-Phase Trainer ──
    "FourPhaseTrainer",
    "PhaseConfig",
    "ImbalanceHandler",
    "MetricsCalculator",
    "ThresholdPolicyResult",
    "DiagnosticsSummary",
    "DecileSummary",
    "split_development_policy_holdout",
    "evaluate_threshold_grid",
    "run_four_phase_pipeline",
    # ── Primary: Production Monitor ──
    "ProductionMonitoringConfig",
    "ProductionMonitoringResult",
    "ProductionBatchResult",
    "AdvancedProductionMonitor",
    "score_production_batch",
    "check_retrain_trigger_advanced",
    "generate_retraining_data_window",
    "calculate_numeric_psi",
    # ── Primary: Config ──
    "ConfigManager",
    "default_config",
    "SchemaContract",
    "DataQualityThresholds",
    "FeatureEncodingConfig",
    "FeatureDefinitionConfig",
    "TimePeriodConfig",
    "CONFIG_VERSION",
    # ── Supporting: Validation ──
    "DataValidator",
    "SchemaValidator",
    "DataQualityValidator",
    "LabelValidator",
    "ValidationReport",
    "ValidationResult",
    "validate_dataframe",
    "check_schema_compatibility",
    # ── Supporting: Artifacts ──
    "ArtifactManager",
    "TransformationPackage",
    "ScalerArtifact",
    "FrequencyEncodingArtifact",
    "OrdinalEncodingArtifact",
    "save_transformation_artifacts",
    "load_transformation_artifacts",
    # ── Supporting: Model Registry ──
    "ModelRecord",
    "ModelRegistry",
    # ── Supporting: Baseline Manager ──
    "BaselineManager",
    "BaselineRecord",
    # ── Legacy (向後相容，不建議新開發使用) ──
    # 僅暴露最小 API 供舊 notebook / CI 腳本使用
    # 新功能請使用 FourPhaseTrainer / AdvancedProductionMonitor
    "RollingTrainer",
    "run_rolling_training_pipeline",
]
