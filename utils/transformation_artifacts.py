"""
Transformation Artifacts Module
===============================
持久化 transformation 參數，確保 train/inference parity

提供：
- Scaler 參數儲存/載入
- Encoding mapping 儲存/載入
- Frequency table 儲存/載入
- 完整 transformation package
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import ConfigManager, default_config, CONFIG_VERSION

logger = logging.getLogger(__name__)


# ============================================
# Artifact Data Classes
# ============================================
@dataclass
class ScalerArtifact:
    """MinMax Scaler 參數"""
    version: str = CONFIG_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def add_feature(self, column: str, min_val: float, max_val: float):
        """新增特徵的 min/max"""
        self.feature_stats[column] = {
            "min": min_val,
            "max": max_val,
            "range": max_val - min_val if max_val != min_val else 1.0
        }
    
    def get_params(self, column: str) -> Optional[Dict]:
        """取得特徵的 scaling 參數"""
        return self.feature_stats.get(column)
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "artifact_type": "minmax_scaler",
            "feature_stats": self.feature_stats
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ScalerArtifact":
        artifact = cls()
        artifact.version = data.get("version", CONFIG_VERSION)
        artifact.created_at = data.get("created_at", "")
        artifact.feature_stats = data.get("feature_stats", {})
        return artifact
    
    def save(self, path: Path):
        """儲存至 JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Scaler artifact saved to: {path}")
    
    @classmethod
    def load(cls, path: Path) -> "ScalerArtifact":
        """從 JSON 載入"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Scaler artifact loaded from: {path}")
        return cls.from_dict(data)


@dataclass
class FrequencyEncodingArtifact:
    """Frequency Encoding 映射表"""
    version: str = CONFIG_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    training_row_count: int = 0
    frequency_maps: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 未見類別策略
    unseen_strategy: str = "zero"  # "zero", "mean", "min"
    unseen_values: Dict[str, float] = field(default_factory=dict)
    
    def add_frequency_map(self, column: str, freq_map: Dict[str, float]):
        """新增類別頻率映射"""
        self.frequency_maps[column] = freq_map
        
        # 計算 unseen 預設值
        if self.unseen_strategy == "mean":
            self.unseen_values[column] = sum(freq_map.values()) / len(freq_map)
        elif self.unseen_strategy == "min":
            self.unseen_values[column] = min(freq_map.values())
        else:
            self.unseen_values[column] = 0.0
    
    def get_frequency(self, column: str, value: Any) -> float:
        """取得類別的頻率"""
        freq_map = self.frequency_maps.get(column, {})
        if value in freq_map:
            return freq_map[value]
        return self.unseen_values.get(column, 0.0)
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "artifact_type": "frequency_encoding",
            "training_row_count": self.training_row_count,
            "unseen_strategy": self.unseen_strategy,
            "unseen_values": self.unseen_values,
            "frequency_maps": self.frequency_maps
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FrequencyEncodingArtifact":
        artifact = cls()
        artifact.version = data.get("version", CONFIG_VERSION)
        artifact.created_at = data.get("created_at", "")
        artifact.training_row_count = data.get("training_row_count", 0)
        artifact.unseen_strategy = data.get("unseen_strategy", "zero")
        artifact.unseen_values = data.get("unseen_values", {})
        artifact.frequency_maps = data.get("frequency_maps", {})
        return artifact
    
    def save(self, path: Path):
        """儲存至 JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Frequency encoding artifact saved to: {path}")
    
    def save_as_csv(self, output_dir: Path):
        """儲存為 CSV（方便審查）"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for column, freq_map in self.frequency_maps.items():
            df = pd.DataFrame([
                {"value": k, "frequency": v}
                for k, v in freq_map.items()
            ])
            df = df.sort_values("frequency", ascending=False)
            
            csv_path = output_dir / f"freq_encoding_{column}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Saved frequency map for {column} to: {csv_path}")
    
    @classmethod
    def load(cls, path: Path) -> "FrequencyEncodingArtifact":
        """從 JSON 載入"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Frequency encoding artifact loaded from: {path}")
        return cls.from_dict(data)


@dataclass
class OrdinalEncodingArtifact:
    """序位編碼映射表"""
    version: str = CONFIG_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ordinal_maps: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # 未見類別預設值
    default_values: Dict[str, int] = field(default_factory=dict)
    
    def add_ordinal_map(self, column: str, mapping: Dict[str, int], default: int = 0):
        """新增序位映射"""
        self.ordinal_maps[column] = mapping
        self.default_values[column] = default
    
    def get_ordinal(self, column: str, value: Any) -> int:
        """取得類別的序位值"""
        ordinal_map = self.ordinal_maps.get(column, {})
        return ordinal_map.get(value, self.default_values.get(column, 0))
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "artifact_type": "ordinal_encoding",
            "ordinal_maps": self.ordinal_maps,
            "default_values": self.default_values
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "OrdinalEncodingArtifact":
        artifact = cls()
        artifact.version = data.get("version", CONFIG_VERSION)
        artifact.created_at = data.get("created_at", "")
        artifact.ordinal_maps = data.get("ordinal_maps", {})
        artifact.default_values = data.get("default_values", {})
        return artifact
    
    def save(self, path: Path):
        """儲存至 JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Ordinal encoding artifact saved to: {path}")
    
    @classmethod
    def load(cls, path: Path) -> "OrdinalEncodingArtifact":
        """從 JSON 載入"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Ordinal encoding artifact loaded from: {path}")
        return cls.from_dict(data)


# ============================================
# Transformation Package
# ============================================
@dataclass
class TransformationPackage:
    """完整 Transformation 套件"""
    version: str = CONFIG_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    run_id: str = ""
    
    # 各類 artifact
    scaler: Optional[ScalerArtifact] = None
    frequency_encoding: Optional[FrequencyEncodingArtifact] = None
    ordinal_encoding: Optional[OrdinalEncodingArtifact] = None
    
    # 特徵列表
    feature_list: List[str] = field(default_factory=list)
    target_column: str = ""
    key_columns: List[str] = field(default_factory=list)
    
    # 元資料
    training_period: Dict[str, str] = field(default_factory=dict)
    training_row_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "package_type": "transformation_package",
            "scaler": self.scaler.to_dict() if self.scaler else None,
            "frequency_encoding": self.frequency_encoding.to_dict() if self.frequency_encoding else None,
            "ordinal_encoding": self.ordinal_encoding.to_dict() if self.ordinal_encoding else None,
            "feature_list": self.feature_list,
            "target_column": self.target_column,
            "key_columns": self.key_columns,
            "training_period": self.training_period,
            "training_row_count": self.training_row_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TransformationPackage":
        package = cls()
        package.version = data.get("version", CONFIG_VERSION)
        package.created_at = data.get("created_at", "")
        package.run_id = data.get("run_id", "")
        
        if data.get("scaler"):
            package.scaler = ScalerArtifact.from_dict(data["scaler"])
        if data.get("frequency_encoding"):
            package.frequency_encoding = FrequencyEncodingArtifact.from_dict(
                data["frequency_encoding"]
            )
        if data.get("ordinal_encoding"):
            package.ordinal_encoding = OrdinalEncodingArtifact.from_dict(
                data["ordinal_encoding"]
            )
        
        package.feature_list = data.get("feature_list", [])
        package.target_column = data.get("target_column", "")
        package.key_columns = data.get("key_columns", [])
        package.training_period = data.get("training_period", {})
        package.training_row_count = data.get("training_row_count", 0)
        
        return package
    
    def save(self, output_dir: Path):
        """儲存完整套件"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 主 manifest
        manifest_path = output_dir / "transformation_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Transformation manifest saved to: {manifest_path}")
        
        # 2. 個別 artifact
        if self.scaler:
            self.scaler.save(output_dir / "scaler_params.json")
        
        if self.frequency_encoding:
            self.frequency_encoding.save(output_dir / "frequency_encoding.json")
            self.frequency_encoding.save_as_csv(output_dir / "frequency_maps")
        
        if self.ordinal_encoding:
            self.ordinal_encoding.save(output_dir / "ordinal_encoding.json")
        
        # 3. Feature list
        feature_df = pd.DataFrame({
            "feature_name": self.feature_list,
            "feature_index": range(len(self.feature_list))
        })
        feature_df.to_csv(output_dir / "feature_list.csv", index=False)
        
        logger.info(f"Transformation package saved to: {output_dir}")
    
    @classmethod
    def load(cls, input_dir: Path) -> "TransformationPackage":
        """載入完整套件"""
        manifest_path = input_dir / "transformation_manifest.json"
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        package = cls.from_dict(data)
        logger.info(f"Transformation package loaded from: {input_dir}")
        return package


# ============================================
# Artifact Manager
# ============================================
class ArtifactManager:
    """Transformation Artifact 管理器"""
    
    def __init__(self, config: ConfigManager = None):
        self.config = config or default_config
    
    def compute_scaler_artifact(
        self, 
        df: DataFrame, 
        columns: List[str]
    ) -> ScalerArtifact:
        """
        從 Development 資料計算 MinMax Scaler 參數
        """
        artifact = ScalerArtifact()
        
        # 一次計算所有欄位的 min/max
        agg_exprs = []
        for col in columns:
            if col in df.columns:
                agg_exprs.extend([
                    F.min(col).alias(f"{col}_min"),
                    F.max(col).alias(f"{col}_max")
                ])
        
        if not agg_exprs:
            return artifact
        
        stats = df.agg(*agg_exprs).collect()[0]
        
        for col in columns:
            if col in df.columns:
                min_val = stats[f"{col}_min"]
                max_val = stats[f"{col}_max"]
                
                # 處理 None
                min_val = min_val if min_val is not None else 0.0
                max_val = max_val if max_val is not None else 1.0
                
                artifact.add_feature(col, min_val, max_val)
                logger.info(f"Scaler [{col}]: min={min_val:.4f}, max={max_val:.4f}")
        
        return artifact
    
    def compute_frequency_artifact(
        self, 
        df: DataFrame, 
        columns: List[str],
        unseen_strategy: str = "zero"
    ) -> FrequencyEncodingArtifact:
        """
        從 Development 資料計算 Frequency Encoding 映射
        """
        artifact = FrequencyEncodingArtifact()
        artifact.training_row_count = df.count()
        artifact.unseen_strategy = unseen_strategy
        
        total = artifact.training_row_count
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # 計算頻率
            freq_df = df.groupBy(col).count()
            freq_data = freq_df.collect()
            
            freq_map = {}
            for row in freq_data:
                value = row[col]
                if value is not None:
                    freq_map[value] = row["count"] / total
            
            artifact.add_frequency_map(col, freq_map)
            logger.info(f"Frequency [{col}]: {len(freq_map)} unique values")
        
        return artifact
    
    def compute_ordinal_artifact(
        self, 
        config: ConfigManager = None
    ) -> OrdinalEncodingArtifact:
        """
        從 Config 建立 Ordinal Encoding 映射
        """
        if config is None:
            config = self.config
        
        artifact = OrdinalEncodingArtifact()
        
        # 教育程度
        artifact.add_ordinal_map(
            "教育程度",
            config.feature_encoding.education_ordinal_mapping,
            default=0
        )
        
        # 月所得（合併細分與粗分）
        income_map = {}
        income_map.update(config.feature_encoding.income_ordinal_mapping_detailed)
        income_map.update(config.feature_encoding.income_ordinal_mapping_coarse)
        artifact.add_ordinal_map("月所得", income_map, default=0)
        
        # 年齡組
        age_group_map = {
            group: info["ordinal"]
            for group, info in config.feature_encoding.age_group_mapping.items()
        }
        artifact.add_ordinal_map("年齡組", age_group_map, default=0)
        
        return artifact
    
    def apply_scaler(
        self,
        df: DataFrame,
        artifact: ScalerArtifact,
        columns: List[str] = None
    ) -> DataFrame:
        """
        套用 Scaler 到 DataFrame
        """
        if columns is None:
            columns = list(artifact.feature_stats.keys())
        
        for col in columns:
            params = artifact.get_params(col)
            if params is None or col not in df.columns:
                continue
            
            min_val = params["min"]
            range_val = params["range"]
            
            scaled_col = f"{col}_scaled"
            df = df.withColumn(
                scaled_col,
                (F.col(col) - F.lit(min_val)) / F.lit(range_val)
            )
        
        return df
    
    def apply_frequency_encoding(
        self,
        df: DataFrame,
        artifact: FrequencyEncodingArtifact,
        columns: List[str] = None
    ) -> DataFrame:
        """
        套用 Frequency Encoding 到 DataFrame
        """
        if columns is None:
            columns = list(artifact.frequency_maps.keys())
        
        for col in columns:
            freq_map = artifact.frequency_maps.get(col, {})
            if not freq_map or col not in df.columns:
                continue
            
            freq_col = f"{col}_頻率"
            default_val = artifact.unseen_values.get(col, 0.0)
            
            # 建立 mapping expression
            mapping_expr = F.lit(default_val)
            for value, freq in freq_map.items():
                mapping_expr = F.when(
                    F.col(col) == F.lit(value), F.lit(freq)
                ).otherwise(mapping_expr)
            
            df = df.withColumn(freq_col, mapping_expr)
        
        return df
    
    def build_transformation_package(
        self,
        df_dev: DataFrame,
        config: ConfigManager = None,
        run_id: str = ""
    ) -> TransformationPackage:
        """
        建立完整 Transformation Package
        """
        if config is None:
            config = self.config
        
        package = TransformationPackage()
        package.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        package.training_row_count = df_dev.count()
        
        # 1. Scaler
        scale_cols = config.feature_definition.numeric_features_to_scale
        package.scaler = self.compute_scaler_artifact(df_dev, scale_cols)
        
        # 2. Frequency Encoding
        freq_cols = config.feature_definition.high_cardinality_features
        package.frequency_encoding = self.compute_frequency_artifact(df_dev, freq_cols)
        
        # 3. Ordinal Encoding
        package.ordinal_encoding = self.compute_ordinal_artifact(config)
        
        # 4. Feature list
        package.feature_list = config.get_all_feature_names()
        package.target_column = config.feature_definition.target_column
        package.key_columns = config.feature_definition.key_columns
        
        # 5. Training period
        if "進件日" in df_dev.columns:
            date_range = df_dev.agg(
                F.min("進件日").alias("min_date"),
                F.max("進件日").alias("max_date")
            ).collect()[0]
            
            package.training_period = {
                "start": str(date_range["min_date"]),
                "end": str(date_range["max_date"])
            }
        
        logger.info(f"Transformation package built: {package.run_id}")
        return package


# ============================================
# Helper Functions
# ============================================
def save_transformation_artifacts(
    package: TransformationPackage,
    output_dir: Path
) -> Dict[str, Path]:
    """
    儲存所有 transformation artifacts
    
    Returns:
        輸出路徑字典
    """
    package.save(output_dir)
    
    return {
        "manifest": output_dir / "transformation_manifest.json",
        "scaler": output_dir / "scaler_params.json",
        "frequency": output_dir / "frequency_encoding.json",
        "ordinal": output_dir / "ordinal_encoding.json",
        "features": output_dir / "feature_list.csv",
    }


def load_transformation_artifacts(
    input_dir: Path
) -> TransformationPackage:
    """
    載入 transformation artifacts
    """
    return TransformationPackage.load(input_dir)
