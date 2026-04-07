"""
Model Training Module
=====================
XGBoost + Calibration for Credit Scoring

特點：
- 處理 Label Imbalance（scale_pos_weight, class_weight）
- Probability Calibration（CalibratedClassifierCV）
- Cross-Validation 訓練
- 與 transformation_artifacts 整合
- 完整的模型評估指標

Version: 1.0.0
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss,
    f1_score, precision_score, recall_score, log_loss
)
import xgboost as xgb

from pyspark.sql import SparkSession, DataFrame

from .config import ConfigManager, default_config, CONFIG_VERSION
from .transformation_artifacts import (
    TransformationPackage, load_transformation_artifacts
)
from .monitoring import AuditLogger, AuditRecord

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================
# Model Artifacts
# ============================================
@dataclass
class ModelArtifact:
    """模型 Artifact"""
    version: str = CONFIG_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    run_id: str = ""
    
    # 模型資訊
    model_type: str = "xgboost_calibrated"
    model_params: Dict = field(default_factory=dict)
    calibration_method: str = "isotonic"  # "isotonic" or "sigmoid"
    
    # 訓練資訊
    training_period: Dict[str, str] = field(default_factory=dict)
    training_row_count: int = 0
    feature_count: int = 0
    feature_names: List[str] = field(default_factory=list)
    
    # 類別不平衡資訊
    positive_ratio: float = 0.0
    scale_pos_weight: float = 1.0
    
    # 評估指標
    metrics: Dict[str, float] = field(default_factory=dict)
    cv_metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    # Calibration 資訊
    calibration_stats: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "model_type": self.model_type,
            "model_params": self.model_params,
            "calibration_method": self.calibration_method,
            "training_period": self.training_period,
            "training_row_count": self.training_row_count,
            "feature_count": self.feature_count,
            "feature_names": self.feature_names,
            "positive_ratio": self.positive_ratio,
            "scale_pos_weight": self.scale_pos_weight,
            "metrics": self.metrics,
            "cv_metrics": self.cv_metrics,
            "calibration_stats": self.calibration_stats,
        }
    
    def save(self, path: Path):
        """儲存 artifact metadata"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Model artifact saved to: {path}")
    
    @classmethod
    def load(cls, path: Path) -> "ModelArtifact":
        """載入 artifact metadata"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        artifact = cls()
        for key, value in data.items():
            if hasattr(artifact, key):
                setattr(artifact, key, value)
        return artifact


# ============================================
# Model Trainer
# ============================================
class CreditScoringModelTrainer:
    """信用評分模型訓練器"""
    
    def __init__(
        self, 
        config: ConfigManager = None,
        random_state: int = 42
    ):
        self.config = config or default_config
        self.random_state = random_state
        
        # 模型物件
        self.base_model: Optional[xgb.XGBClassifier] = None
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        
        # Artifact
        self.artifact = ModelArtifact()
        
    def get_default_xgb_params(self, scale_pos_weight: float = 1.0) -> Dict:
        """取得預設的 XGBoost 參數"""
        return {
            # 基本參數
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            
            # 正則化
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            
            # 類別不平衡
            "scale_pos_weight": scale_pos_weight,
            
            # 其他
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
    
    def calculate_scale_pos_weight(self, y: np.ndarray) -> float:
        """計算 scale_pos_weight 處理類別不平衡"""
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        
        if n_positive == 0:
            return 1.0
        
        scale_pos_weight = n_negative / n_positive
        logger.info(f"Label distribution: Positive={n_positive}, Negative={n_negative}")
        logger.info(f"Positive ratio: {n_positive / len(y):.2%}")
        logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")
        
        return scale_pos_weight
    
    def train_with_cv_calibration(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_splits: int = 5,
        calibration_method: str = "isotonic",
        xgb_params: Dict = None,
        early_stopping_rounds: int = 20,
    ) -> Tuple[CalibratedClassifierCV, ModelArtifact]:
        """
        使用 Cross-Validation 訓練並校正模型
        
        Args:
            X: 特徵矩陣
            y: 標籤
            feature_names: 特徵名稱
            n_splits: CV 折數
            calibration_method: "isotonic" 或 "sigmoid"
            xgb_params: XGBoost 參數（可選）
            early_stopping_rounds: Early stopping
            
        Returns:
            (calibrated_model, artifact)
        """
        logger.info("=" * 60)
        logger.info("開始訓練 XGBoost + Calibration")
        logger.info("=" * 60)
        
        # 1. 計算類別不平衡權重
        scale_pos_weight = self.calculate_scale_pos_weight(y)
        self.artifact.scale_pos_weight = scale_pos_weight
        self.artifact.positive_ratio = np.mean(y)
        
        # 2. 設定 XGBoost 參數
        if xgb_params is None:
            xgb_params = self.get_default_xgb_params(scale_pos_weight)
        else:
            xgb_params["scale_pos_weight"] = scale_pos_weight
        
        self.artifact.model_params = xgb_params
        self.artifact.calibration_method = calibration_method
        self.artifact.feature_names = feature_names
        self.artifact.feature_count = len(feature_names)
        self.artifact.training_row_count = len(y)
        
        logger.info(f"訓練樣本數: {len(y)}")
        logger.info(f"特徵數: {len(feature_names)}")
        logger.info(f"Calibration method: {calibration_method}")
        
        # 3. 建立 base model
        self.base_model = xgb.XGBClassifier(**xgb_params)
        
        # 4. 使用 CalibratedClassifierCV 進行 CV 訓練 + 校正
        logger.info(f"開始 {n_splits}-Fold CV Calibration...")
        
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_model,
            method=calibration_method,
            cv=n_splits,
            n_jobs=-1
        )
        
        # 訓練
        self.calibrated_model.fit(X, y)
        
        logger.info("✓ 模型訓練完成")
        
        # 5. 評估模型
        self._evaluate_model(X, y)
        
        # 6. 計算 CV 指標
        self._calculate_cv_metrics(X, y, n_splits)
        
        # 7. 計算 Calibration 統計
        self._calculate_calibration_stats(X, y)
        
        return self.calibrated_model, self.artifact
    
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray):
        """評估模型（in-sample）"""
        logger.info("計算評估指標...")
        
        y_pred_proba = self.calibrated_model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # 基本指標
        metrics = {
            "auc_roc": roc_auc_score(y, y_pred_proba),
            "auc_pr": average_precision_score(y, y_pred_proba),
            "brier_score": brier_score_loss(y, y_pred_proba),
            "log_loss": log_loss(y, y_pred_proba),
            "f1": f1_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
        }
        
        # 針對少數類（婉拒=0）的指標
        # 對於信用評分，找出「婉拒」案件更重要
        metrics["precision_reject"] = precision_score(y, y_pred, pos_label=0)
        metrics["recall_reject"] = recall_score(y, y_pred, pos_label=0)
        metrics["f1_reject"] = f1_score(y, y_pred, pos_label=0)
        
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        metrics["true_negative"] = int(tn)
        metrics["false_positive"] = int(fp)
        metrics["false_negative"] = int(fn)
        metrics["true_positive"] = int(tp)
        
        # 計算 KS statistic
        metrics["ks_statistic"] = self._calculate_ks(y, y_pred_proba)
        
        self.artifact.metrics = metrics
        
        logger.info("In-Sample Metrics:")
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
        logger.info(f"  KS Statistic: {metrics['ks_statistic']:.4f}")
        logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
        logger.info("")
        logger.info("  核准類 (Positive=1):")
        logger.info(f"    Precision: {metrics['precision']:.4f}")
        logger.info(f"    Recall: {metrics['recall']:.4f}")
        logger.info(f"    F1: {metrics['f1']:.4f}")
        logger.info("")
        logger.info("  婉拒類 (Negative=0) - 重要！:")
        logger.info(f"    Precision: {metrics['precision_reject']:.4f}")
        logger.info(f"    Recall: {metrics['recall_reject']:.4f}")
        logger.info(f"    F1: {metrics['f1_reject']:.4f}")
        logger.info("")
        logger.info(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    def _calculate_ks(self, y: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """計算 KS Statistic"""
        from scipy import stats
        pos_proba = y_pred_proba[y == 1]
        neg_proba = y_pred_proba[y == 0]
        ks_stat, _ = stats.ks_2samp(pos_proba, neg_proba)
        return float(ks_stat)
    
    def _calculate_cv_metrics(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_splits: int
    ):
        """計算 CV 指標（更完整版本）"""
        logger.info(f"計算 {n_splits}-Fold CV 指標...")
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        cv_metrics = {
            "auc_roc": [],
            "auc_pr": [],
            "brier_score": [],
            "f1": [],
            "f1_reject": [],  # 婉拒類 F1
            "precision_reject": [],
            "recall_reject": [],
            "ks_statistic": [],
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 訓練
            scale_pos_weight = self.calculate_scale_pos_weight(y_train)
            params = self.get_default_xgb_params(scale_pos_weight)
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            # 預測
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # 記錄指標
            cv_metrics["auc_roc"].append(roc_auc_score(y_val, y_pred_proba))
            cv_metrics["auc_pr"].append(average_precision_score(y_val, y_pred_proba))
            cv_metrics["brier_score"].append(brier_score_loss(y_val, y_pred_proba))
            cv_metrics["f1"].append(f1_score(y_val, y_pred))
            
            # 婉拒類指標
            cv_metrics["f1_reject"].append(f1_score(y_val, y_pred, pos_label=0))
            cv_metrics["precision_reject"].append(precision_score(y_val, y_pred, pos_label=0))
            cv_metrics["recall_reject"].append(recall_score(y_val, y_pred, pos_label=0))
            cv_metrics["ks_statistic"].append(self._calculate_ks(y_val, y_pred_proba))
            
            logger.info(f"  Fold {fold + 1}: AUC={cv_metrics['auc_roc'][-1]:.4f}, "
                       f"F1_reject={cv_metrics['f1_reject'][-1]:.4f}")
        
        self.artifact.cv_metrics = cv_metrics
        
        # 計算平均
        logger.info("")
        logger.info("CV Metrics (mean ± std):")
        logger.info("-" * 40)
        for name, values in cv_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"  {name}: {mean_val:.4f} ± {std_val:.4f}")
    
    def _calculate_calibration_stats(self, X: np.ndarray, y: np.ndarray):
        """計算 Calibration 統計"""
        logger.info("計算 Calibration 統計...")
        
        y_pred_proba = self.calibrated_model.predict_proba(X)[:, 1]
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10)
        
        # Expected Calibration Error (ECE)
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        self.artifact.calibration_stats = {
            "expected_calibration_error": float(ece),
            "prob_true_bins": prob_true.tolist(),
            "prob_pred_bins": prob_pred.tolist(),
        }
        
        logger.info(f"  Expected Calibration Error (ECE): {ece:.4f}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """取得特徵重要性"""
        if self.calibrated_model is None:
            raise ValueError("模型尚未訓練")
        
        # 從校正後的模型中取得 base estimators
        importances = []
        for calibrated_clf in self.calibrated_model.calibrated_classifiers_:
            base_clf = calibrated_clf.estimator
            importances.append(base_clf.feature_importances_)
        
        # 平均重要性
        mean_importance = np.mean(importances, axis=0)
        
        df = pd.DataFrame({
            "feature": self.artifact.feature_names,
            "importance": mean_importance
        })
        df = df.sort_values("importance", ascending=False)
        
        return df
    
    def save_model(self, output_dir: Path, run_id: str = None):
        """儲存模型和 artifacts"""
        if self.calibrated_model is None:
            raise ValueError("模型尚未訓練")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if run_id:
            self.artifact.run_id = run_id
        
        # 1. 儲存模型
        model_path = output_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.calibrated_model, f)
        logger.info(f"Model saved to: {model_path}")
        
        # 2. 儲存 base model（用於 SHAP 解釋）
        base_model_path = output_dir / "base_model.pkl"
        # 取第一個 calibrated classifier 的 base estimator
        base_estimator = self.calibrated_model.calibrated_classifiers_[0].estimator
        with open(base_model_path, 'wb') as f:
            pickle.dump(base_estimator, f)
        logger.info(f"Base model saved to: {base_model_path}")
        
        # 3. 儲存 artifact
        self.artifact.save(output_dir / "model_artifact.json")
        
        # 4. 儲存特徵重要性
        feature_importance = self.get_feature_importance()
        feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
        logger.info(f"Feature importance saved to: {output_dir / 'feature_importance.csv'}")
        
        # 5. 儲存評估報告
        report = {
            "run_id": self.artifact.run_id,
            "created_at": self.artifact.created_at,
            "model_type": self.artifact.model_type,
            "training_row_count": self.artifact.training_row_count,
            "feature_count": self.artifact.feature_count,
            "positive_ratio": self.artifact.positive_ratio,
            "scale_pos_weight": self.artifact.scale_pos_weight,
            "metrics": self.artifact.metrics,
            "cv_metrics_summary": {
                name: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
                for name, values in self.artifact.cv_metrics.items()
            },
            "calibration_stats": self.artifact.calibration_stats,
        }
        
        with open(output_dir / "training_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Training report saved to: {output_dir / 'training_report.json'}")
        
        return output_dir
    
    @classmethod
    def load_model(cls, model_dir: Path) -> Tuple["CreditScoringModelTrainer", Any]:
        """載入模型"""
        model_dir = Path(model_dir)
        
        trainer = cls()
        
        # 載入模型
        with open(model_dir / "model.pkl", 'rb') as f:
            trainer.calibrated_model = pickle.load(f)
        
        # 載入 artifact
        trainer.artifact = ModelArtifact.load(model_dir / "model_artifact.json")
        
        logger.info(f"Model loaded from: {model_dir}")
        return trainer


# ============================================
# Training Pipeline
# ============================================
def prepare_training_data(
    spark: SparkSession,
    gold_dev_path: Path,
    feature_columns: List[str],
    target_column: str = "授信結果_二元",
    key_columns: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    從 Gold Development 準備訓練資料
    
    Returns:
        (X, y, feature_names)
    """
    logger.info(f"讀取訓練資料: {gold_dev_path}")
    
    # 讀取 Parquet
    df = spark.read.parquet(str(gold_dev_path))
    logger.info(f"原始筆數: {df.count()}")
    
    # 選取需要的欄位
    select_cols = feature_columns + [target_column]
    if key_columns:
        select_cols = key_columns + select_cols
    
    # 過濾掉 target 為 NULL 的資料
    df = df.filter(df[target_column].isNotNull())
    logger.info(f"有效筆數（target 非 NULL）: {df.count()}")
    
    # 轉換為 Pandas
    pdf = df.select(*select_cols).toPandas()
    
    # 準備 X, y
    X = pdf[feature_columns].values.astype(np.float32)
    y = pdf[target_column].values.astype(np.int32)
    
    # 處理 NaN
    X = np.nan_to_num(X, nan=0.0)
    
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    
    return X, y, feature_columns


def run_model_training(
    project_root: Path,
    spark: SparkSession = None,
    config: ConfigManager = None,
    run_id: str = None,
    n_splits: int = 5,
    calibration_method: str = "isotonic",
    xgb_params: Dict = None,
) -> Path:
    """
    執行模型訓練 Pipeline
    
    Args:
        project_root: 專案根目錄
        spark: SparkSession
        config: 設定管理器
        run_id: 執行 ID
        n_splits: CV 折數
        calibration_method: Calibration 方法
        xgb_params: XGBoost 參數
        
    Returns:
        模型輸出路徑
    """
    if config is None:
        config = default_config
    
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 路徑設定
    gold_dev_path = project_root / "datamart" / "gold" / "development"
    model_output_path = project_root / "models" / run_id
    audit_path = project_root / "models" / "audit"
    
    # 初始化 Audit
    audit_logger = AuditLogger(audit_path)
    audit_record = AuditRecord(
        run_id=run_id,
        stage="model_train",
        action="run_model_training",
        input_path=str(gold_dev_path),
        output_path=str(model_output_path),
        config_version=CONFIG_VERSION,
    )
    
    # 建立 Spark Session
    should_stop_spark = False
    if spark is None:
        spark = SparkSession.builder.appName("model_train").getOrCreate()
        should_stop_spark = True
    
    try:
        logger.info("=" * 60)
        logger.info("開始模型訓練 Pipeline")
        logger.info("=" * 60)
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Gold Dev Path: {gold_dev_path}")
        
        # ============================================
        # 1. 準備特徵列表
        # ============================================
        feature_definition = config.feature_definition
        
        # 1.1 數值特徵（經過 MinMax Scaling）
        numeric_scaled = [f"{col}_scaled" for col in feature_definition.numeric_features_to_scale]
        
        # 1.2 序位特徵
        ordinal_features = feature_definition.ordinal_features
        
        # 1.3 二元特徵
        binary_features = feature_definition.binary_features
        
        # 1.4 交互特徵（Gold 階段產生）
        cross_features = [f"{name}_scaled" for name in feature_definition.cross_features.keys()]
        
        # 1.5 高基數頻率編碼特徵
        freq_features = [f"{col}_頻率_scaled" for col in feature_definition.high_cardinality_features]
        
        # 組合所有特徵
        feature_columns = (
            numeric_scaled +
            ordinal_features +
            binary_features +
            cross_features +
            freq_features
        )
        
        logger.info("特徵組成:")
        logger.info(f"  數值特徵 (scaled): {len(numeric_scaled)}")
        logger.info(f"  序位特徵: {len(ordinal_features)}")
        logger.info(f"  二元特徵: {len(binary_features)}")
        logger.info(f"  交互特徵: {len(cross_features)}")
        logger.info(f"  頻率編碼特徵: {len(freq_features)}")
        
        logger.info(f"特徵數: {len(feature_columns)}")
        
        # ============================================
        # 2. 準備訓練資料
        # ============================================
        X, y, feature_names = prepare_training_data(
            spark,
            gold_dev_path,
            feature_columns,
            target_column=feature_definition.target_column,
            key_columns=feature_definition.key_columns
        )
        
        audit_record.row_count_before = len(y)
        
        # ============================================
        # 3. 訓練模型
        # ============================================
        trainer = CreditScoringModelTrainer(config)
        
        calibrated_model, artifact = trainer.train_with_cv_calibration(
            X, y,
            feature_names=feature_names,
            n_splits=n_splits,
            calibration_method=calibration_method,
            xgb_params=xgb_params,
        )
        
        # ============================================
        # 4. 儲存模型
        # ============================================
        trainer.save_model(model_output_path, run_id)
        
        audit_record.row_count_after = len(y)
        audit_record.output_path = str(model_output_path)
        
        # 記錄關鍵指標
        audit_record.metrics = artifact.metrics
        
        # ============================================
        # 5. 儲存 Audit Log
        # ============================================
        audit_logger.add_record(audit_record)
        
        logger.info("=" * 60)
        logger.info("🎉 模型訓練完成！")
        logger.info("=" * 60)
        logger.info(f"模型輸出: {model_output_path}")
        logger.info(f"AUC-ROC: {artifact.metrics.get('auc_roc', 0):.4f}")
        logger.info(f"AUC-PR: {artifact.metrics.get('auc_pr', 0):.4f}")
        
        return model_output_path
        
    except Exception as e:
        audit_record.errors.append(str(e))
        audit_logger.add_record(audit_record)
        raise
        
    finally:
        if should_stop_spark:
            spark.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Credit Scoring Model Training")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID")
    parser.add_argument("--n-splits", type=int, default=5, help="CV folds")
    parser.add_argument("--calibration", type=str, default="isotonic", 
                        choices=["isotonic", "sigmoid"], help="Calibration method")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    run_model_training(
        project_root,
        run_id=args.run_id,
        n_splits=args.n_splits,
        calibration_method=args.calibration,
    )
