"""
Model Registry
==============
追蹤不同時間點訓練的模型，記錄最佳模型

功能：
1. 記錄每次訓練的模型 metadata
2. 比較不同模型的表現
3. 標記並追蹤「最佳模型」
4. 支援模型版本回溯

Version: 1.0.0
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ModelRecord:
    """單一模型的紀錄"""
    model_version: str  # e.g., "credit_model_2026_04_08"
    run_id: str
    created_at: str
    
    # 訓練資訊
    training_date: str  # snapshot date used for training
    train_start_date: str
    train_end_date: str
    oot_start_date: Optional[str] = None
    oot_end_date: Optional[str] = None
    
    # 資料統計
    train_rows: int = 0
    test_rows: int = 0
    oot_rows: int = 0
    positive_ratio: float = 0.0
    
    # 模型表現（關鍵指標）
    auc_train: float = 0.0
    auc_test: float = 0.0
    auc_oot: Optional[float] = None
    gini_train: float = 0.0
    gini_test: float = 0.0
    gini_oot: Optional[float] = None
    
    # 額外指標
    ks_test: Optional[float] = None
    f1_reject_test: Optional[float] = None
    brier_score_test: Optional[float] = None
    
    # 模型設定
    model_type: str = "xgboost_calibrated"
    calibration_method: str = "isotonic"
    hyperparameters: Dict = field(default_factory=dict)
    
    # 檔案路徑
    model_path: str = ""
    artifact_path: str = ""
    
    # 狀態
    is_best: bool = False
    is_production: bool = False
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelRegistry:
    """
    Model Registry - 追蹤所有訓練過的模型
    
    檔案結構：
        model_bank/
        ├── registry.json              # 模型登錄表
        ├── best_model.txt             # 最佳模型名稱
        ├── credit_model_2026_04_08/
        │   ├── model.pkl
        │   ├── model_artifact.json
        │   └── ...
        └── credit_model_2026_04_15/
            └── ...
    """
    
    def __init__(self, model_bank_path: Path):
        self.model_bank_path = Path(model_bank_path)
        self.model_bank_path.mkdir(parents=True, exist_ok=True)
        
        self.registry_path = self.model_bank_path / "registry.json"
        self.best_model_path = self.model_bank_path / "best_model.txt"
        
        # 載入現有的 registry
        self.records: Dict[str, ModelRecord] = {}
        self._load_registry()
    
    def _load_registry(self):
        """載入 registry"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for version, record_data in data.get("models", {}).items():
                self.records[version] = ModelRecord.from_dict(record_data)
            
            logger.info(f"載入 {len(self.records)} 個模型紀錄")
        else:
            logger.info("Registry 不存在，建立新的")
    
    def _save_registry(self):
        """儲存 registry"""
        data = {
            "last_updated": datetime.now().isoformat(),
            "total_models": len(self.records),
            "best_model": self.get_best_model_version(),
            "production_model": self.get_production_model_version(),
            "models": {v: r.to_dict() for v, r in self.records.items()}
        }
        
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Registry 已儲存: {self.registry_path}")
    
    def register_model(
        self,
        model_version: str,
        run_id: str,
        training_date: str,
        metrics: Dict[str, float],
        data_stats: Dict[str, Any],
        hyperparameters: Dict = None,
        model_path: str = "",
        artifact_path: str = "",
        date_config: Dict = None,
        **kwargs
    ) -> ModelRecord:
        """
        註冊新模型
        
        Args:
            model_version: 模型版本名稱 (e.g., "credit_model_2026_04_08")
            run_id: 執行 ID
            training_date: 訓練日期
            metrics: 模型指標 dict
            data_stats: 資料統計 dict
            hyperparameters: 超參數
            model_path: 模型檔案路徑
            artifact_path: artifact 路徑
            date_config: 日期設定
        """
        record = ModelRecord(
            model_version=model_version,
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            training_date=training_date,
            train_start_date=date_config.get("train_start_date", "") if date_config else "",
            train_end_date=date_config.get("train_end_date", "") if date_config else "",
            oot_start_date=date_config.get("oot_start_date") if date_config else None,
            oot_end_date=date_config.get("oot_end_date") if date_config else None,
            
            # 資料統計
            train_rows=data_stats.get("train_rows", 0),
            test_rows=data_stats.get("test_rows", 0),
            oot_rows=data_stats.get("oot_rows", 0),
            positive_ratio=data_stats.get("positive_ratio", 0.0),
            
            # 模型表現
            auc_train=metrics.get("auc_train", metrics.get("auc_roc", 0.0)),
            auc_test=metrics.get("auc_test", 0.0),
            auc_oot=metrics.get("auc_oot"),
            gini_train=metrics.get("gini_train", 2 * metrics.get("auc_train", 0.0) - 1),
            gini_test=metrics.get("gini_test", 2 * metrics.get("auc_test", 0.0) - 1),
            gini_oot=metrics.get("gini_oot"),
            
            # 額外指標
            ks_test=metrics.get("ks_statistic"),
            f1_reject_test=metrics.get("f1_reject"),
            brier_score_test=metrics.get("brier_score"),
            
            # 模型設定
            model_type=kwargs.get("model_type", "xgboost_calibrated"),
            calibration_method=kwargs.get("calibration_method", "isotonic"),
            hyperparameters=hyperparameters or {},
            
            # 路徑
            model_path=model_path,
            artifact_path=artifact_path,
        )
        
        self.records[model_version] = record
        self._save_registry()
        
        logger.info(f"OK: 模型已註冊: {model_version}")
        logger.info(f"  AUC Test: {record.auc_test:.4f}, GINI Test: {record.gini_test:.4f}")
        
        return record
    
    def get_best_model_version(self) -> Optional[str]:
        """取得最佳模型版本"""
        for version, record in self.records.items():
            if record.is_best:
                return version
        return None
    
    def get_production_model_version(self) -> Optional[str]:
        """取得 production 模型版本"""
        for version, record in self.records.items():
            if record.is_production:
                return version
        return None
    
    def set_best_model(self, model_version: str, reason: str = ""):
        """
        設定最佳模型
        
        Args:
            model_version: 模型版本
            reason: 選擇原因
        """
        if model_version not in self.records:
            raise ValueError(f"模型不存在: {model_version}")
        
        # 清除舊的 best 標記
        for record in self.records.values():
            record.is_best = False
        
        # 設定新的 best
        self.records[model_version].is_best = True
        if reason:
            self.records[model_version].notes = f"Best model: {reason}"
        
        # 寫入 best_model.txt
        with open(self.best_model_path, 'w') as f:
            f.write(model_version)
        
        self._save_registry()
        logger.info(f"OK: 最佳模型已設定: {model_version}")
    
    def set_production_model(self, model_version: str):
        """設定 production 模型"""
        if model_version not in self.records:
            raise ValueError(f"模型不存在: {model_version}")
        
        # 清除舊的 production 標記
        for record in self.records.values():
            record.is_production = False
        
        self.records[model_version].is_production = True
        self._save_registry()
        logger.info(f"OK: Production 模型已設定: {model_version}")
    
    def auto_select_best(self, metric: str = "auc_test") -> str:
        """
        自動選擇最佳模型（根據指定指標）
        
        Args:
            metric: 用來比較的指標 (auc_test, gini_test, auc_oot, etc.)
            
        Returns:
            最佳模型版本名稱
        """
        if not self.records:
            raise ValueError("沒有任何模型紀錄")
        
        # 找出指標最高的模型
        best_version = None
        best_value = -float('inf')
        
        for version, record in self.records.items():
            value = getattr(record, metric, None)
            if value is not None and value > best_value:
                best_value = value
                best_version = version
        
        if best_version:
            self.set_best_model(best_version, f"Auto-selected by {metric}={best_value:.4f}")
        
        return best_version
    
    def compare_models(self, versions: List[str] = None) -> str:
        """
        比較模型表現
        
        Args:
            versions: 要比較的版本列表，None 表示全部
            
        Returns:
            比較結果的表格字串
        """
        if versions is None:
            versions = list(self.records.keys())
        
        # 表頭
        lines = [
            "=" * 100,
            "Model Comparison",
            "=" * 100,
            f"{'Version':<30} {'AUC_Test':>10} {'GINI_Test':>10} {'AUC_OOT':>10} {'KS':>8} {'Best':>6} {'Prod':>6}",
            "-" * 100,
        ]
        
        for version in sorted(versions, reverse=True):
            if version not in self.records:
                continue
            
            r = self.records[version]
            auc_oot_str = f"{r.auc_oot:.4f}" if r.auc_oot else "N/A"
            ks_str = f"{r.ks_test:.4f}" if r.ks_test else "N/A"
            best_str = "OK:" if r.is_best else ""
            prod_str = "OK:" if r.is_production else ""
            
            lines.append(
                f"{version:<30} {r.auc_test:>10.4f} {r.gini_test:>10.4f} {auc_oot_str:>10} {ks_str:>8} {best_str:>6} {prod_str:>6}"
            )
        
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def get_model_record(self, model_version: str) -> Optional[ModelRecord]:
        """取得模型紀錄"""
        return self.records.get(model_version)
    
    def list_models(self) -> List[str]:
        """列出所有模型版本"""
        return sorted(self.records.keys(), reverse=True)
    
    def get_latest_model(self) -> Optional[str]:
        """取得最新的模型版本"""
        if not self.records:
            return None
        return sorted(self.records.keys(), reverse=True)[0]


# ============================================
# Extended Model Record for Rolling Training
# ============================================
@dataclass
class RollingTrainedModelRecord(ModelRecord):
    """
    Rolling Training 產出的模型紀錄
    
    擴展 ModelRecord 以支援：
    - Rolling training 彙總結果
    - Champion strategy 資訊
    - OOT 指標
    - Threshold 設定
    """
    # Rolling training 資訊
    champion_strategy: str = ""
    rolling_cycles_count: int = 0
    
    # 彙總的 rolling metrics
    avg_cv_auc: float = 0.0
    std_cv_auc: float = 0.0
    avg_monitor_auc: float = 0.0
    std_monitor_auc: float = 0.0
    avg_monitor_f1_reject: float = 0.0
    std_monitor_f1_reject: float = 0.0
    overall_score: float = 0.0
    
    # OOT metrics
    oot_auc: Optional[float] = None
    oot_f1: Optional[float] = None
    oot_f1_reject: Optional[float] = None
    oot_ks: Optional[float] = None
    oot_brier: Optional[float] = None
    
    # Threshold config
    lower_threshold: float = 0.4
    upper_threshold: float = 0.7
    
    # Zone summary (from OOT)
    zone_high_count: int = 0
    zone_high_approve_rate: float = 0.0
    zone_manual_count: int = 0
    zone_manual_approve_rate: float = 0.0
    zone_low_count: int = 0
    zone_low_approve_rate: float = 0.0
    
    # Rolling results path
    rolling_results_path: str = ""


class ExtendedModelRegistry(ModelRegistry):
    """
    擴展的 Model Registry，支援 Rolling Training
    """
    
    def register_rolling_model(
        self,
        model_version: str,
        run_id: str,
        training_date: str,
        champion_strategy: str,
        rolling_summary: Dict,
        oot_metrics: Dict = None,
        zone_summary: List[Dict] = None,
        threshold_config: Dict = None,
        model_path: str = "",
        rolling_results_path: str = "",
        **kwargs
    ) -> RollingTrainedModelRecord:
        """
        註冊 Rolling Training 產出的模型
        """
        # 從 rolling_summary 取得彙總指標
        strategy_data = rolling_summary.get(champion_strategy, {})
        
        record = RollingTrainedModelRecord(
            model_version=model_version,
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            training_date=training_date,
            train_start_date=kwargs.get('train_start_date', ''),
            train_end_date=kwargs.get('train_end_date', ''),
            
            # Rolling info
            champion_strategy=champion_strategy,
            rolling_cycles_count=len(strategy_data.get('cycle_results', [])),
            
            # Rolling metrics
            avg_cv_auc=strategy_data.get('avg_cv_auc', 0.0),
            std_cv_auc=strategy_data.get('std_cv_auc', 0.0),
            avg_monitor_auc=strategy_data.get('avg_monitor_auc', 0.0),
            std_monitor_auc=strategy_data.get('std_monitor_auc', 0.0),
            avg_monitor_f1_reject=strategy_data.get('avg_monitor_f1_reject', 0.0),
            std_monitor_f1_reject=strategy_data.get('std_monitor_f1_reject', 0.0),
            overall_score=strategy_data.get('overall_score', 0.0),
            
            # Model type
            model_type=champion_strategy,
            model_path=model_path,
            rolling_results_path=rolling_results_path,
        )
        
        # OOT metrics
        if oot_metrics:
            record.oot_auc = oot_metrics.get('auc')
            record.oot_f1 = oot_metrics.get('f1')
            record.oot_f1_reject = oot_metrics.get('f1_reject')
            record.oot_ks = oot_metrics.get('ks')
            record.oot_brier = oot_metrics.get('brier_score')
            
            # 同步到父類欄位
            record.auc_oot = record.oot_auc
            record.auc_test = record.avg_monitor_auc
            record.gini_test = record.avg_monitor_auc * 2 - 1 if record.avg_monitor_auc else 0
        
        # Threshold config
        if threshold_config:
            record.lower_threshold = threshold_config.get('lower_threshold', 0.4)
            record.upper_threshold = threshold_config.get('upper_threshold', 0.7)
        
        # Zone summary
        if zone_summary:
            for zone in zone_summary:
                if zone.get('zone_name') == '高通過機率區':
                    record.zone_high_count = zone.get('count', 0)
                    record.zone_high_approve_rate = zone.get('actual_approve_rate', 0.0)
                elif zone.get('zone_name') == '人工審核區':
                    record.zone_manual_count = zone.get('count', 0)
                    record.zone_manual_approve_rate = zone.get('actual_approve_rate', 0.0)
                elif zone.get('zone_name') == '低通過機率區':
                    record.zone_low_count = zone.get('count', 0)
                    record.zone_low_approve_rate = zone.get('actual_approve_rate', 0.0)
        
        # 儲存
        self.records[model_version] = record
        self._save_registry()
        
        logger.info(f"OK: Rolling Training 模型已註冊: {model_version}")
        logger.info(f"  Champion Strategy: {champion_strategy}")
        logger.info(f"  Rolling Cycles: {record.rolling_cycles_count}")
        logger.info(f"  Overall Score: {record.overall_score:.4f}")
        if record.oot_auc:
            logger.info(f"  OOT AUC: {record.oot_auc:.4f}")
        
        return record
    
    def compare_rolling_models(self, versions: List[str] = None) -> str:
        """比較 Rolling Training 模型"""
        if versions is None:
            versions = list(self.records.keys())
        
        lines = [
            "=" * 130,
            "Rolling Training Model Comparison",
            "=" * 130,
            f"{'Version':<25} {'Strategy':<15} {'Cycles':>6} {'Avg_CV_AUC':>10} {'Avg_Mon_AUC':>12} {'OOT_AUC':>10} {'OOT_F1_rej':>10} {'Score':>8}",
            "-" * 130,
        ]
        
        for version in sorted(versions, reverse=True):
            if version not in self.records:
                continue
            
            r = self.records[version]
            
            # 檢查是否為 RollingTrainedModelRecord
            strategy = getattr(r, 'champion_strategy', r.model_type)
            cycles = getattr(r, 'rolling_cycles_count', 0)
            avg_cv = getattr(r, 'avg_cv_auc', 0.0)
            avg_mon = getattr(r, 'avg_monitor_auc', 0.0)
            oot_auc = getattr(r, 'oot_auc', r.auc_oot)
            oot_f1_rej = getattr(r, 'oot_f1_reject', None)
            score = getattr(r, 'overall_score', 0.0)
            
            oot_auc_str = f"{oot_auc:.4f}" if oot_auc else "N/A"
            oot_f1_str = f"{oot_f1_rej:.4f}" if oot_f1_rej else "N/A"
            
            lines.append(
                f"{version:<25} {strategy:<15} {cycles:>6} {avg_cv:>10.4f} {avg_mon:>12.4f} {oot_auc_str:>10} {oot_f1_str:>10} {score:>8.4f}"
            )
        
        lines.append("=" * 130)
        return "\n".join(lines)


def load_best_model_name(model_bank_path: Path) -> Optional[str]:
    """讀取最佳模型名稱（簡化版本）"""
    best_model_file = Path(model_bank_path) / "best_model.txt"
    if best_model_file.exists():
        with open(best_model_file, 'r') as f:
            return f.read().strip()
    return None


# ============================================
# CLI 工具
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Registry CLI")
    parser.add_argument("--model-bank", type=str, default="model_bank", help="Model bank 路徑")
    parser.add_argument("--list", action="store_true", help="列出所有模型")
    parser.add_argument("--compare", action="store_true", help="比較所有模型")
    parser.add_argument("--compare-rolling", action="store_true", help="比較 Rolling Training 模型")
    parser.add_argument("--set-best", type=str, help="設定最佳模型")
    parser.add_argument("--set-production", type=str, help="設定 Production 模型")
    parser.add_argument("--auto-best", action="store_true", help="自動選擇最佳模型")
    parser.add_argument("--metric", type=str, default="auc_test", help="自動選擇的指標")
    
    args = parser.parse_args()
    
    # 使用 ExtendedModelRegistry
    registry = ExtendedModelRegistry(Path(args.model_bank))
    
    if args.list:
        print("Models in registry:")
        for v in registry.list_models():
            r = registry.records[v]
            status = []
            if r.is_best:
                status.append("BEST")
            if r.is_production:
                status.append("PROD")
            status_str = f" [{', '.join(status)}]" if status else ""
            print(f"  - {v}{status_str}")
    
    if args.compare:
        print(registry.compare_models())
    
    if args.compare_rolling:
        print(registry.compare_rolling_models())
    
    if args.set_best:
        registry.set_best_model(args.set_best)
    
    if args.set_production:
        registry.set_production_model(args.set_production)
    
    if args.auto_best:
        best = registry.auto_select_best(args.metric)
        print(f"Best model selected: {best}")
