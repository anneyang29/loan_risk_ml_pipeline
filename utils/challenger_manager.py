"""
Challenger Manager
==================
統一管理所有 challenger 的核心邏輯。
由 main.py 呼叫，不依賴 experiments/ 目錄下的 script。

包含：
- run_c2_feature_pruning_challenger()   C2: 移除 6 個不穩定特徵 (25 -> 19 features)
- run_c3_decision_tuning_challenger()   C3: Decision-oriented + anti-overfitting tuning
- compare_against_baseline()            與 baseline v1 做指標對比
- generate_routing_report()             輸出可讀的分流分析報告

Usage (from main.py):
    python main.py --run-c2
    python main.py --run-c3
    python main.py --compare-baseline --challenger c2
    python main.py --compare-baseline --challenger c3
"""

import json
import logging
import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.four_phase_trainer import (
    FourPhaseTrainer,
    evaluate_threshold_grid,
    score_threshold_policy,
    assign_score_zone,
    RANDOM_STATE,
)
from utils.config import (
    ConfigManager,
    BusinessConstraintConfig,
)

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    brier_score_loss,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Reference Metrics
# ============================================================

BASELINE_V1_METRICS: Dict = {
    "auc": 0.8953,
    "f1_reject": 0.3975,
    "brier_score": 0.0313,
    "ks": 0.6345,
    "precision_reject": 0.6879,
    "recall_reject": 0.2795,
    "low_zone_reject_precision": 0.5298,
    "manual_review_ratio": 0.0955,
    "high_zone_ratio": 0.8725,
    "low_zone_ratio": 0.0319,
    "rolling_stability_score": 0.0308,
    "avg_monitor_f1_reject": 0.2714,
}

C2_METRICS: Dict = {
    "auc": 0.9016,
    "f1_reject": 0.3903,
    "brier_score": 0.0316,
    "ks": 0.6472,
    "precision_reject": 0.6467,
    "recall_reject": 0.2795,
    "low_zone_reject_precision": 0.5478,
    "manual_review_ratio": 0.0868,
    "high_zone_ratio": 0.8794,
    "low_zone_ratio": 0.0338,
    "stability_score": 0.0286,
    "avg_monitor_f1_reject": 0.2712,
}

# C2 feature drop list
C2_FEATURES_TO_DROP: List[str] = [
    "車齡_異常旗標",
    "內部往來次數_是否特殊值",
    "近半年同業查詢次數_是否缺失",
    "教育程度_是否缺失",
    "負債月所得比_scaled",
    "教育_所得交互_scaled",
]

# C3 tuning candidates
C3_TUNING_CANDIDATES: List[Dict] = [
    {
        "config_id": "c2_baseline",
        "description": "C2 original config (depth=3, moderate reg)",
        "model_type": "xgboost",
        "max_depth": 3, "min_child_weight": 20,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 1.0, "reg_lambda": 5.0,
        "learning_rate": 0.03, "n_estimators": 300,
        "gamma": 1.0, "max_delta_step": 1,
    },
    {
        "config_id": "conservative_strong_reg",
        "description": "Strong regularization: depth=2, high alpha/lambda",
        "model_type": "xgboost",
        "max_depth": 2, "min_child_weight": 30,
        "subsample": 0.65, "colsample_bytree": 0.6,
        "reg_alpha": 3.0, "reg_lambda": 12.0,
        "learning_rate": 0.02, "n_estimators": 500,
        "gamma": 2.5, "max_delta_step": 2,
    },
    {
        "config_id": "conservative_high_gamma",
        "description": "High gamma (2.0) + depth=3",
        "model_type": "xgboost",
        "max_depth": 3, "min_child_weight": 25,
        "subsample": 0.7, "colsample_bytree": 0.65,
        "reg_alpha": 2.0, "reg_lambda": 8.0,
        "learning_rate": 0.025, "n_estimators": 400,
        "gamma": 2.0, "max_delta_step": 2,
    },
    {
        "config_id": "depth4_moderate_reg",
        "description": "depth=4 + moderate regularization",
        "model_type": "xgboost",
        "max_depth": 4, "min_child_weight": 15,
        "subsample": 0.75, "colsample_bytree": 0.7,
        "reg_alpha": 1.5, "reg_lambda": 6.0,
        "learning_rate": 0.03, "n_estimators": 300,
        "gamma": 1.0, "max_delta_step": 1,
    },
    {
        "config_id": "depth4_strong_reg",
        "description": "depth=4 + strong regularization",
        "model_type": "xgboost",
        "max_depth": 4, "min_child_weight": 25,
        "subsample": 0.7, "colsample_bytree": 0.6,
        "reg_alpha": 2.5, "reg_lambda": 10.0,
        "learning_rate": 0.02, "n_estimators": 400,
        "gamma": 1.5, "max_delta_step": 2,
    },
    {
        "config_id": "high_delta_step",
        "description": "max_delta_step=3 to strengthen minority class updates",
        "model_type": "xgboost",
        "max_depth": 3, "min_child_weight": 20,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 1.5, "reg_lambda": 6.0,
        "learning_rate": 0.03, "n_estimators": 350,
        "gamma": 1.0, "max_delta_step": 3,
    },
    {
        "config_id": "low_lr_many_trees",
        "description": "learning_rate=0.01, n_estimators=800",
        "model_type": "xgboost",
        "max_depth": 3, "min_child_weight": 20,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 1.5, "reg_lambda": 6.0,
        "learning_rate": 0.01, "n_estimators": 800,
        "gamma": 1.0, "max_delta_step": 2,
    },
    {
        "config_id": "rf_conservative",
        "description": "Random Forest: low depth, high min_samples",
        "model_type": "random_forest",
        "n_estimators": 500, "max_depth": 5,
        "min_samples_split": 80, "min_samples_leaf": 30,
        "max_features": "sqrt",
    },
]

C3_CALIBRATION_METHODS: List[str] = ["isotonic", "sigmoid"]


# ============================================================
# Decision Scoring Config
# ============================================================

@dataclass
class DecisionScoringConfig:
    w_low_zone_reject_precision: float = 0.25
    w_high_zone_approve_precision: float = 0.15
    w_manual_review: float = 0.15
    w_holdout_f1_reject: float = 0.15
    w_holdout_brier: float = 0.10
    w_holdout_auc: float = 0.10
    w_holdout_ks: float = 0.10
    overfitting_penalty_weight: float = 2.0
    max_train_monitor_auc_gap: float = 0.03
    max_monitor_holdout_auc_gap: float = 0.04
    max_stability_score: float = 0.03
    robustness_penalty_weight: float = 1.5
    max_f1r_gap: float = 0.05
    max_calibration_gap: float = 0.10
    min_high_zone_approve_precision: float = 0.98
    max_manual_review_ratio: float = 0.12
    min_auto_decision_rate: float = 0.85
    max_hard_train_monitor_auc_gap: float = 0.08
    target_low_zone_reject_precision: float = 0.65

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================
# Shared Trainers (feature-pruned)
# ============================================================

class PrunedFeatureTrainer(FourPhaseTrainer):
    """FourPhaseTrainer with a configurable feature drop list."""

    def __init__(self, features_to_drop: List[str], **kwargs):
        super().__init__(**kwargs)
        self.features_to_drop = features_to_drop

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        base = super()._get_feature_columns(df)
        pruned = [f for f in base if f not in self.features_to_drop]
        dropped = [f for f in self.features_to_drop if f in base]
        if dropped:
            logger.info("Feature pruning: %d -> %d (dropped: %s)",
                        len(base), len(pruned), dropped)
        return pruned


# ============================================================
# Zone Metric Helpers
# ============================================================

def compute_zone_metrics_from_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
) -> Dict:
    """Calculate zone-level metrics from raw predictions."""
    zones = assign_score_zone(y_pred_proba, lower_threshold, upper_threshold)
    total = len(y_true)
    result: Dict = {}

    # High zone (approve zone): precision = fraction that are actual positives
    high_mask = (zones == 2)
    result["high_zone_ratio"] = float(high_mask.sum() / total)
    result["high_zone_approve_precision"] = (
        float(np.mean(y_true[high_mask])) if high_mask.sum() > 0 else 0.0
    )

    # Low zone (reject zone): precision = fraction that are actual negatives
    low_mask = (zones == 0)
    result["low_zone_ratio"] = float(low_mask.sum() / total)
    result["low_zone_reject_precision"] = (
        float(np.mean(1 - y_true[low_mask])) if low_mask.sum() > 0 else 0.0
    )

    # Manual zone
    manual_mask = (zones == 1)
    result["manual_zone_ratio"] = float(manual_mask.sum() / total)
    result["manual_review_ratio"] = result["manual_zone_ratio"]  # alias

    # Aggregated KPIs
    result["human_review_workload_ratio"] = (
        result["manual_zone_ratio"] + result["low_zone_ratio"]
    )
    result["auto_light_touch_ratio"] = result["high_zone_ratio"]
    result["auto_decision_rate"] = result["high_zone_ratio"] + result["low_zone_ratio"]

    return result


def compute_zone_metrics_from_csv(holdout_pred_path: Path) -> Dict:
    """Calculate zone metrics from a saved holdout_predictions CSV."""
    df = pd.read_csv(holdout_pred_path)
    total = len(df)
    result: Dict = {}

    def _reject_col(sub: pd.DataFrame) -> pd.Series:
        if "actual_label" in sub.columns:
            return sub["actual_label"]
        if "授信結果_二元" in sub.columns:
            return sub["授信結果_二元"]
        return pd.Series(dtype=int)

    for zone_name in df["zone_name"].unique() if "zone_name" in df.columns else []:
        zdf = df[df["zone_name"] == zone_name]
        col = _reject_col(zdf)
        result[zone_name] = {
            "count": len(zdf),
            "ratio": len(zdf) / total,
            "reject_rate": float((col == 0).sum() / len(zdf)) if len(zdf) > 0 else 0.0,
        }

    low = df[df["zone_name"] == "低通過機率區"] if "zone_name" in df.columns else pd.DataFrame()
    manual = df[df["zone_name"] == "人工審核區"] if "zone_name" in df.columns else pd.DataFrame()
    high = df[df["zone_name"] == "高通過機率區"] if "zone_name" in df.columns else pd.DataFrame()

    col_low = _reject_col(low)
    result["low_zone_reject_precision"] = (
        float((col_low == 0).sum() / len(low)) if len(low) > 0 else 0.0
    )
    result["manual_zone_ratio"] = len(manual) / total
    result["manual_review_ratio"] = result["manual_zone_ratio"]
    result["high_zone_ratio"] = len(high) / total
    result["low_zone_ratio"] = len(low) / total
    result["human_review_workload_ratio"] = result["manual_zone_ratio"] + result["low_zone_ratio"]
    result["auto_light_touch_ratio"] = result["high_zone_ratio"]

    return result


def compute_rolling_stability(rolling_results_path: Path) -> Dict:
    """Summarise rolling-window stability from rolling_results.csv."""
    df = pd.read_csv(rolling_results_path)
    if "model_name" in df.columns:
        for candidate in df["model_name"].unique():
            if "xgboost" in candidate.lower():
                df = df[df["model_name"] == candidate]
                break
        else:
            df = df[df["model_name"] == df["model_name"].iloc[0]]

    result = {
        "avg_monitor_auc":      float(df["monitor_auc"].mean()) if "monitor_auc" in df.columns else None,
        "avg_monitor_f1_reject": float(df["monitor_f1_reject"].mean()) if "monitor_f1_reject" in df.columns else None,
        "avg_monitor_ks":        float(df["monitor_ks"].mean()) if "monitor_ks" in df.columns else None,
        "avg_monitor_brier":     float(df["monitor_brier"].mean()) if "monitor_brier" in df.columns else None,
        "std_monitor_auc":       float(df["monitor_auc"].std()) if "monitor_auc" in df.columns else None,
        "stability_score":       float(df["monitor_auc"].std()) if "monitor_auc" in df.columns else None,
    }
    return result


# ============================================================
# Upgrade Candidate Evaluation
# ============================================================

def evaluate_upgrade_candidate(
    challenger: Dict,
    baseline: Dict,
    scoring_cfg: DecisionScoringConfig,
) -> Tuple[str, str, bool, bool]:
    """
    Determine upgrade_candidate status.

    Returns:
        (status, reason, suitable_for_routing, suitable_for_replacing_classifier)
        status: "yes" | "conditional" | "no"
    """
    bl_workload = (
        baseline.get("manual_review_ratio", 0.0)
        + baseline.get("low_zone_ratio", 0.0)
    )
    lzp_improved   = challenger.get("low_zone_reject_precision", 0) > baseline.get("low_zone_reject_precision", 0) + 0.005
    hz_ok          = challenger.get("high_zone_approve_precision", 0) >= scoring_cfg.min_high_zone_approve_precision
    workload_ok    = challenger.get("human_review_workload_ratio", 1) <= bl_workload + 0.02
    f1r_ok         = challenger.get("holdout_f1_reject", 0) >= baseline.get("f1_reject", 0) - 0.005
    no_overfit     = challenger.get("overfitting_penalty", 1) < 0.01

    if lzp_improved and hz_ok and workload_ok and f1r_ok and no_overfit:
        return (
            "yes",
            "low_zone_reject_precision improved, F1_reject not regressed, "
            "high zone precision meets threshold, human workload not worsened, "
            "overfitting controlled. Candidate qualifies for full upgrade.",
            True, True,
        )
    if lzp_improved and hz_ok and workload_ok:
        return (
            "conditional",
            "low_zone_reject_precision improved and routing metrics acceptable, "
            "but F1_reject regressed vs baseline. "
            "Suitable for routing improvement but not for full baseline replacement.",
            True, False,
        )
    return (
        "no",
        "Challenger did not meet core upgrade conditions "
        "(low_zone_reject_precision improvement + high zone precision + workload).",
        False, False,
    )


# ============================================================
# Comparison Report Generator
# ============================================================

def compare_against_baseline(
    challenger_id: str,
    challenger_desc: str,
    challenger_holdout: Dict,
    challenger_zone: Dict,
    challenger_stability: Dict,
    challenger_features_dropped: List[str],
    challenger_features_remaining: int,
    baseline_metrics: Dict,
    output_dir: Path,
    upgrade_status: Optional[str] = None,
    upgrade_reason: Optional[str] = None,
    suitable_routing: Optional[bool] = None,
    suitable_replace: Optional[bool] = None,
) -> str:
    """
    Generate a plain-text Markdown comparison report.
    No emoji or icon characters in output.
    """
    lines: List[str] = []
    lines.append(f"# Challenger vs Baseline Comparison: {challenger_id}")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Challenger: {challenger_id}")
    lines.append(f"Description: {challenger_desc}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Feature changes
    lines.append("## 1. Feature Changes")
    lines.append("")
    lines.append("| | Baseline v1 | Challenger |")
    lines.append("|--|-------------|------------|")
    lines.append(f"| Feature count | 25 | {challenger_features_remaining} |")
    lines.append(f"| Features dropped | 0 | {len(challenger_features_dropped)} |")
    lines.append("")
    if challenger_features_dropped:
        lines.append("Features removed:")
        for f in challenger_features_dropped:
            lines.append(f"  - {f}")
    lines.append("")

    # Classifier metrics
    lines.append("## 2. Classifier Metrics (Final Holdout)")
    lines.append("")
    lines.append("| Metric | Baseline v1 | Challenger | Delta | Result |")
    lines.append("|--------|-------------|------------|-------|--------|")

    classifier_pairs = [
        ("AUC",             "auc",             "auc",             True),
        ("F1_reject",       "f1_reject",        "f1_reject",       True),
        ("Brier",           "brier_score",      "brier_score",     False),
        ("KS",              "ks",               "ks",              True),
        ("Precision_reject","precision_reject", "precision_reject",True),
        ("Recall_reject",   "recall_reject",    "recall_reject",   True),
    ]
    for name, bl_key, ch_key, higher_better in classifier_pairs:
        bv = baseline_metrics.get(bl_key)
        cv = challenger_holdout.get(ch_key)
        if bv is None or cv is None:
            continue
        delta = cv - bv
        if higher_better:
            verdict = "IMPROVED" if delta > 0.001 else ("REGRESSED" if delta < -0.001 else "UNCHANGED")
        else:
            verdict = "IMPROVED" if delta < -0.001 else ("REGRESSED" if delta > 0.001 else "UNCHANGED")
        lines.append(f"| {name} | {bv:.4f} | {cv:.4f} | {delta:+.4f} | {verdict} |")
    lines.append("")

    # Zone precision vs ratio (clearly separated)
    lines.append("## 3. Routing / Zone Metrics")
    lines.append("")
    lines.append("> NOTE: Zone Precision and Zone Ratio are different concepts.")
    lines.append("> Precision = purity within a zone; Ratio = that zone's share of all cases.")
    lines.append("")
    lines.append("### 3-A. Zone Precision (purity within each zone)")
    lines.append("")
    lines.append("| Metric | Description | Baseline v1 | Challenger | Delta | Result |")
    lines.append("|--------|-------------|-------------|------------|-------|--------|")

    bv = baseline_metrics.get("low_zone_reject_precision")
    cv = challenger_zone.get("low_zone_reject_precision")
    if bv is not None and cv is not None:
        d = cv - bv
        v = "IMPROVED" if d > 0.001 else ("REGRESSED" if d < -0.001 else "UNCHANGED")
        lines.append(f"| low_zone_reject_precision | Fraction of true rejects in low zone | {bv:.4f} | {cv:.4f} | {d:+.4f} | {v} |")

    cv_hz = challenger_zone.get("high_zone_approve_precision")
    if cv_hz is not None:
        lines.append(f"| high_zone_approve_precision | Fraction of true approvals in high zone | N/A | {cv_hz:.4f} | — | — |")

    lines.append("")
    lines.append("### 3-B. Zone Ratio (each zone as fraction of total cases)")
    lines.append("")
    lines.append("| Zone | Baseline v1 | Challenger | Delta | Note |")
    lines.append("|------|-------------|------------|-------|------|")

    bl_h = baseline_metrics.get("high_zone_ratio", 0.0)
    bl_m = baseline_metrics.get("manual_review_ratio", 0.0)
    bl_l = baseline_metrics.get("low_zone_ratio", 0.0)
    bl_wl = bl_m + bl_l

    ch_h = challenger_zone.get("high_zone_ratio", 0.0)
    ch_m = challenger_zone.get("manual_zone_ratio", challenger_zone.get("manual_review_ratio", 0.0))
    ch_l = challenger_zone.get("low_zone_ratio", 0.0)
    ch_wl = challenger_zone.get("human_review_workload_ratio", ch_m + ch_l)

    lines.append(f"| high_zone_ratio  | {bl_h:.1%} | {ch_h:.1%} | {(ch_h-bl_h):+.1%} | Higher = more auto-approve |")
    lines.append(f"| manual_zone_ratio | {bl_m:.1%} | {ch_m:.1%} | {(ch_m-bl_m):+.1%} | Lower = less human review |")
    lines.append(f"| low_zone_ratio   | {bl_l:.1%} | {ch_l:.1%} | {(ch_l-bl_l):+.1%} | Strict-review pool size |")
    lines.append(f"| **human_review_workload_ratio** | **{bl_wl:.1%}** | **{ch_wl:.1%}** | **{(ch_wl-bl_wl):+.1%}** | **Core KPI: manual + low** |")
    lines.append("")
    lines.append(
        f"Routing layout: High {ch_h:.1%} | Manual {ch_m:.1%} | Low {ch_l:.1%} | "
        f"Total human workload {ch_wl:.1%}"
    )
    lines.append("")

    # Stability
    lines.append("## 4. Rolling Stability")
    lines.append("")
    lines.append("| Metric | Baseline v1 | Challenger | Delta | Result |")
    lines.append("|--------|-------------|------------|-------|--------|")

    bl_stab = {"avg_monitor_auc": 0.8644, "avg_monitor_f1_reject": 0.2714, "stability_score": 0.0308}
    for name, bl_key, higher_better in [
        ("Avg Monitor AUC",    "avg_monitor_auc",      True),
        ("Avg Monitor F1r",    "avg_monitor_f1_reject", True),
        ("Stability (AUC std)","stability_score",       False),
    ]:
        bv = bl_stab.get(bl_key)
        cv = challenger_stability.get(bl_key)
        if bv is None or cv is None:
            continue
        d = cv - bv
        v = ("IMPROVED" if (d > 0.001 if higher_better else d < -0.001)
             else ("REGRESSED" if (d < -0.001 if higher_better else d > 0.001) else "UNCHANGED"))
        lines.append(f"| {name} | {bv:.4f} | {cv:.4f} | {d:+.4f} | {v} |")
    lines.append("")

    # Upgrade candidate
    lines.append("## 5. Upgrade Candidate Evaluation")
    lines.append("")
    if upgrade_status:
        status_label = {
            "yes":         "YES - Qualifies for full upgrade",
            "conditional": "CONDITIONAL - Routing improvement only; not recommended as full replacement",
            "no":          "NO - Does not meet upgrade criteria",
        }.get(upgrade_status, upgrade_status)
        lines.append(f"upgrade_candidate: {status_label}")
        lines.append("")
        lines.append(f"Reason: {upgrade_reason or 'N/A'}")
        lines.append("")
        lines.append("| Dimension | Recommendation |")
        lines.append("|-----------|----------------|")
        lines.append(f"| Auto-approve support | {'YES' if ch_hz is not None and ch_hz >= 0.98 else 'REVIEW NEEDED'} |")

        lzp_val = challenger_zone.get("low_zone_reject_precision", 0.0)
        lines.append(f"| Low zone enhanced review pool | {'YES' if lzp_val >= 0.55 else 'MARGINAL'} |")
        lines.append(f"| Fully auto-reject low zone | {'YES' if lzp_val >= 0.65 else 'NOT RECOMMENDED'} |")
        lines.append(f"| Suitable for routing improvement | {'YES' if suitable_routing else 'NO'} |")
        lines.append(f"| Suitable for replacing baseline classifier | {'YES' if suitable_replace else 'NO'} |")

        if lzp_val < 0.65:
            lines.append("")
            lines.append(
                f"NOTE: low_zone_reject_precision = {lzp_val:.4f}, below target 0.65. "
                "Recommend using low zone as enhanced review / fast-track reject review pool. "
                "Direct fully auto-reject is not recommended at this precision level."
            )
    lines.append("")

    # Baseline comparison conclusion
    lines.append("## 6. Conclusion vs Baseline")
    lines.append("")
    improved, regressed = [], []

    checks = [
        ("low_zone_reject_precision", challenger_zone.get("low_zone_reject_precision", 0),
         baseline_metrics.get("low_zone_reject_precision", 0), True, "routing quality"),
        ("AUC", challenger_holdout.get("auc", 0), baseline_metrics.get("auc", 0), True, "classifier"),
        ("KS",  challenger_holdout.get("ks", 0),  baseline_metrics.get("ks", 0),  True, "classifier"),
        ("F1_reject", challenger_holdout.get("f1_reject", 0), baseline_metrics.get("f1_reject", 0), True, "classifier"),
        ("manual_zone_ratio", ch_m, bl_m, False, "human workload"),
        ("human_review_workload_ratio", ch_wl, bl_wl, False, "human workload"),
        ("stability_score", challenger_stability.get("stability_score", 1),
         bl_stab.get("stability_score", 1), False, "stability"),
    ]
    for metric, cv, bv, higher_better, category in checks:
        delta = cv - bv
        if higher_better and delta > 0.001:
            improved.append(f"  {metric} ({category}): {bv:.4f} -> {cv:.4f} ({delta:+.4f})")
        elif not higher_better and delta < -0.001:
            improved.append(f"  {metric} ({category}): {bv:.4f} -> {cv:.4f} ({delta:+.4f})")
        elif higher_better and delta < -0.001:
            regressed.append(f"  {metric} ({category}): {bv:.4f} -> {cv:.4f} ({delta:+.4f})")
        elif not higher_better and delta > 0.001:
            regressed.append(f"  {metric} ({category}): {bv:.4f} -> {cv:.4f} ({delta:+.4f})")

    lines.append("### Improvements vs Baseline v1")
    lines += (improved if improved else ["  None"])
    lines.append("")
    lines.append("### Regressions vs Baseline v1")
    lines += (regressed if regressed else ["  None"])
    lines.append("")

    if upgrade_status == "yes":
        lines.append("RECOMMENDATION: Proceed with upgrade. Challenger outperforms baseline across key routing and classifier metrics.")
    elif upgrade_status == "conditional":
        lines.append("RECOMMENDATION: Deploy for routing improvement only.")
        lines.append("  - Routing improvement = YES: low zone is cleaner, human workload acceptable.")
        lines.append("  - Classifier replacement = NO: F1_reject regressed; full baseline replacement carries risk.")
        lines.append("")
        lines.append("Next steps:")
        lines.append("  1. Run C1 Reject-Targeted Feature Challenger to improve both routing AND classifier metrics.")
        lines.append("  2. In the interim, challenger can be deployed as routing-only alongside existing baseline.")
    else:
        lines.append("RECOMMENDATION: Do not upgrade. Challenger did not demonstrate sufficient improvement.")
        lines.append("  Consider C1 reject-targeted feature engineering as the next step.")

    lines.append("")
    lines.append("---")
    lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    report = "\n".join(lines)
    out_path = output_dir / f"{challenger_id}_vs_baseline_comparison.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info("Comparison report saved: %s", out_path)
    return report


# ============================================================
# Routing Report Generator
# ============================================================

def generate_routing_report(
    challenger_id: str,
    best_candidate: Dict,
    baseline_metrics: Dict,
    output_dir: Path,
) -> str:
    """
    Generate a concise human-readable routing report (plain text, no emoji).
    """
    hz  = best_candidate.get("high_zone_ratio", 0.0)
    mz  = best_candidate.get("manual_zone_ratio", best_candidate.get("manual_review_ratio", 0.0))
    lz  = best_candidate.get("low_zone_ratio", 0.0)
    wl  = best_candidate.get("human_review_workload_ratio", mz + lz)
    lzp = best_candidate.get("low_zone_reject_precision", 0.0)
    hzp = best_candidate.get("high_zone_approve_precision", 0.0)
    f1r = best_candidate.get("holdout_f1_reject", best_candidate.get("f1_reject", 0.0))
    auc = best_candidate.get("holdout_auc", best_candidate.get("auc", 0.0))
    uc  = best_candidate.get("upgrade_candidate", "N/A")

    bl_wl = baseline_metrics.get("manual_review_ratio", 0.0) + baseline_metrics.get("low_zone_ratio", 0.0)

    lines = [
        f"# Routing Report: {challenger_id}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Zone Distribution",
        f"  High zone (auto-approve eligible) : {hz:.1%}",
        f"  Manual zone (standard review)      : {mz:.1%}",
        f"  Low zone (enhanced review pool)    : {lz:.1%}",
        f"  -----------------------------------------------",
        f"  Total human review workload        : {wl:.1%}  (manual + low)",
        f"  vs Baseline v1 human workload      : {bl_wl:.1%}",
        f"  Delta                              : {(wl - bl_wl):+.1%}",
        "",
        "## Zone Precision",
        f"  high_zone_approve_precision : {hzp:.4f}  (target >= 0.98)",
        f"  low_zone_reject_precision   : {lzp:.4f}  (target >= 0.65)",
        "",
        "## Classifier Metrics",
        f"  Holdout AUC      : {auc:.4f}  (baseline: {baseline_metrics.get('auc', 0):.4f})",
        f"  Holdout F1_reject: {f1r:.4f}  (baseline: {baseline_metrics.get('f1_reject', 0):.4f})",
        "",
        "## Upgrade Candidate",
        f"  upgrade_candidate: {uc}",
        "",
        "## Low Zone Operational Guidance",
    ]

    if lzp >= 0.65:
        lines.append(
            f"  low_zone_reject_precision = {lzp:.4f} meets target 0.65. "
            "Low zone may be considered for fully auto-reject review."
        )
    else:
        lines.append(
            f"  low_zone_reject_precision = {lzp:.4f}, below target 0.65."
        )
        lines.append(
            "  RECOMMENDATION: Treat low zone as enhanced review / fast-track reject review pool."
        )
        lines.append(
            "  Do NOT apply fully auto-reject at this precision level."
        )

    lines += [
        "",
        "## Improvement vs Baseline",
        f"  - low_zone_reject_precision: {baseline_metrics.get('low_zone_reject_precision', 0):.4f} -> {lzp:.4f}",
        f"  - human_review_workload_ratio: {bl_wl:.1%} -> {wl:.1%}",
        f"  - AUC: {baseline_metrics.get('auc', 0):.4f} -> {auc:.4f}",
        f"  - F1_reject: {baseline_metrics.get('f1_reject', 0):.4f} -> {f1r:.4f}",
        "",
        "---",
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]

    report = "\n".join(lines)
    out_path = output_dir / f"{challenger_id}_routing_report.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info("Routing report saved: %s", out_path)
    return report


# ============================================================
# C2: Feature Pruning Challenger
# ============================================================

def run_c2_feature_pruning_challenger(
    project_root: Optional[Path] = None,
    features_to_drop: Optional[List[str]] = None,
) -> Dict:
    """
    Run the C2 Feature Pruning challenger.

    - Drops 6 unstable / low-importance features (25 -> 19)
    - Reruns the full four-phase pipeline
    - Compares against baseline v1
    - Saves comparison report + metadata to model_bank/experiments/c2_feature_pruning/

    Returns dict with output_dir, comparison report path, and key metrics.
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    if features_to_drop is None:
        features_to_drop = C2_FEATURES_TO_DROP

    logger.info("=" * 70)
    logger.info("C2 Feature Pruning Challenger")
    logger.info("  Dropping %d features: %s", len(features_to_drop), features_to_drop)
    logger.info("=" * 70)

    config_path = project_root / "config" / "pipeline_config.yaml"
    config = ConfigManager(config_path) if config_path.exists() else None

    trainer = PrunedFeatureTrainer(
        features_to_drop=features_to_drop,
        project_root=project_root,
        imbalance_strategy="scale_weight",
        config=config,
    )

    results = trainer.run_full_pipeline(
        model_names=None,
        use_calibration=True,
        lower_threshold=0.5,
        upper_threshold=0.85,
    )
    output_dir = Path(results["output_dir"])
    logger.info("C2 training complete. Results: %s", output_dir)

    # Collect metrics
    holdout_metrics_path = output_dir / "final_holdout_metrics.json"
    c2_holdout = json.loads(holdout_metrics_path.read_text(encoding="utf-8"))

    c2_zone = compute_zone_metrics_from_csv(output_dir / "final_holdout_predictions.csv")
    c2_stability = compute_rolling_stability(output_dir / "rolling_results.csv")

    feature_names = json.loads((output_dir / "feature_names.json").read_text(encoding="utf-8"))

    # Upgrade evaluation
    combined = {**c2_holdout, **c2_zone}
    scoring_cfg = DecisionScoringConfig()
    uc_status, uc_reason, suitable_r, suitable_c = evaluate_upgrade_candidate(
        combined, BASELINE_V1_METRICS, scoring_cfg
    )

    # Reports
    challenger_output = project_root / "model_bank" / "experiments" / "c2_feature_pruning"
    challenger_output.mkdir(parents=True, exist_ok=True)

    report = compare_against_baseline(
        challenger_id="C2_feature_pruning",
        challenger_desc="Remove 6 unstable/low-importance features (25 -> 19)",
        challenger_holdout=c2_holdout,
        challenger_zone=c2_zone,
        challenger_stability=c2_stability,
        challenger_features_dropped=features_to_drop,
        challenger_features_remaining=len(feature_names),
        baseline_metrics=BASELINE_V1_METRICS,
        output_dir=challenger_output,
        upgrade_status=uc_status,
        upgrade_reason=uc_reason,
        suitable_routing=suitable_r,
        suitable_replace=suitable_c,
    )

    routing_report = generate_routing_report(
        challenger_id="C2_feature_pruning",
        best_candidate={**c2_holdout, **c2_zone, "upgrade_candidate": uc_status},
        baseline_metrics=BASELINE_V1_METRICS,
        output_dir=challenger_output,
    )

    # Metadata
    meta = {
        "challenger_id": "C2_feature_pruning",
        "source_run": str(output_dir),
        "features_dropped": features_to_drop,
        "feature_count": len(feature_names),
        "features_remaining": feature_names,
        "baseline_reference": "baseline_v1",
        "holdout_metrics": c2_holdout,
        "zone_metrics": c2_zone,
        "stability_metrics": c2_stability,
        "upgrade_candidate": uc_status,
        "upgrade_reason": uc_reason,
        "suitable_for_routing_improvement": suitable_r,
        "suitable_for_replacing_baseline_classifier": suitable_c,
        "created_at": datetime.now().isoformat(),
    }
    (challenger_output / "c2_challenger_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    logger.info("C2 output dir: %s", challenger_output)
    return {
        "output_dir": str(challenger_output),
        "training_dir": str(output_dir),
        "report": report,
        "routing_report": routing_report,
        "metrics": {**c2_holdout, **c2_zone, **c2_stability},
        "upgrade_candidate": uc_status,
    }


# ============================================================
# C3: Decision-Oriented Tuning Challenger
# ============================================================

def _build_model(model_type: str, params: Dict, scale_pos_weight: float, class_weight: Dict) -> Any:
    if model_type == "xgboost":
        return xgb.XGBClassifier(
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
            objective="binary:logistic",
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            min_samples_split=params.get("min_samples_split", 50),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            max_features=params.get("max_features", "sqrt"),
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def _evaluate_c3_candidate(
    candidate_cfg: Dict,
    calibration_method: str,
    trainer: PrunedFeatureTrainer,
    X_dev: np.ndarray, y_dev: np.ndarray,
    X_policy: np.ndarray, y_policy: np.ndarray,
    X_holdout: np.ndarray, y_holdout: np.ndarray,
    scale_pos_weight: float,
    class_weight: Dict,
    feature_cols: List[str],
    scoring_cfg: DecisionScoringConfig,
    bc_cfg: BusinessConstraintConfig,
) -> Optional[Dict]:
    """Evaluate a single C3 candidate configuration."""
    config_id = candidate_cfg["config_id"]
    model_type = candidate_cfg.get("model_type", "xgboost")
    label = f"{config_id}_{calibration_method}"
    logger.info("Evaluating C3 candidate: %s", label)

    base_model = _build_model(model_type, candidate_cfg, scale_pos_weight, class_weight)

    # Rolling window evaluation
    monitor_aucs, monitor_f1rs, monitor_ks_list, monitor_briers, train_aucs = [], [], [], [], []

    for window in trainer.window_definitions:
        try:
            train_df   = trainer.load_development_data(start_date=window.train_start,   end_date=window.train_end)
            monitor_df = trainer.load_development_data(start_date=window.monitor_start, end_date=window.monitor_end)
            if len(train_df) == 0 or len(monitor_df) == 0:
                continue
            Xtr, ytr, _ = trainer._prepare_xy(train_df, feature_cols)
            Xmo, ymo, _ = trainer._prepare_xy(monitor_df, feature_cols)
            m = deepcopy(base_model)
            if model_type == "xgboost":
                m.set_params(early_stopping_rounds=30)
                m.fit(Xtr, ytr, eval_set=[(Xmo, ymo)], verbose=False)
            else:
                m.fit(Xtr, ytr)
            train_aucs.append(trainer.metrics_calc.calculate_all_metrics(ytr, m.predict_proba(Xtr)[:, 1])["auc"])
            mo_m = trainer.metrics_calc.calculate_all_metrics(ymo, m.predict_proba(Xmo)[:, 1])
            monitor_aucs.append(mo_m["auc"])
            monitor_f1rs.append(mo_m["f1_reject"])
            monitor_ks_list.append(mo_m["ks"])
            monitor_briers.append(mo_m["brier_score"])
        except Exception as exc:
            logger.debug("Window failed for %s: %s", label, exc)

    if not monitor_aucs:
        logger.warning("No valid rolling windows for %s", label)
        return None

    avg_train_auc    = float(np.mean(train_aucs))
    avg_monitor_auc  = float(np.mean(monitor_aucs))
    avg_monitor_f1r  = float(np.mean(monitor_f1rs))
    avg_monitor_ks   = float(np.mean(monitor_ks_list))
    avg_monitor_brier= float(np.mean(monitor_briers))
    stability_score  = float(np.std(monitor_aucs))
    std_monitor_f1r  = float(np.std(monitor_f1rs))
    train_monitor_auc_gap = float(avg_train_auc - avg_monitor_auc)

    # Full retrain
    full_model = deepcopy(base_model)
    if calibration_method != "none":
        cal_model = CalibratedClassifierCV(estimator=full_model, method=calibration_method, cv=5, n_jobs=-1)
        cal_model.fit(X_dev, y_dev)
        final_model = cal_model
    else:
        if model_type == "xgboost":
            from sklearn.model_selection import train_test_split as tts
            Xtr2, Xes2, ytr2, yes2 = tts(X_dev, y_dev, test_size=0.15, random_state=RANDOM_STATE, stratify=y_dev)
            full_model.set_params(early_stopping_rounds=30)
            full_model.fit(Xtr2, ytr2, eval_set=[(Xes2, yes2)], verbose=False)
        else:
            full_model.fit(X_dev, y_dev)
        final_model = full_model

    dev_m = trainer.metrics_calc.calculate_all_metrics(y_dev, final_model.predict_proba(X_dev)[:, 1])
    train_auc_full   = dev_m["auc"]
    train_brier_full = dev_m["brier_score"]

    # Per-candidate threshold selection
    y_policy_pred  = final_model.predict_proba(X_policy)[:, 1]
    th_results     = evaluate_threshold_grid(
        y_policy, y_policy_pred,
        lower_thresholds=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        upper_thresholds=[0.70, 0.80, 0.85, 0.90, 0.95],
    )
    scored = score_threshold_policy(th_results, bc_cfg)
    passing_th = [r for r in scored if r.passes_hard_constraints]
    best_th = passing_th[0] if passing_th else (scored[0] if scored else None)

    if best_th is None:
        best_lower, best_upper = 0.7, 0.9
    else:
        best_lower = best_th.lower_threshold
        best_upper = best_th.upper_threshold

    # Holdout evaluation
    y_holdout_pred  = final_model.predict_proba(X_holdout)[:, 1]
    holdout_m       = trainer.metrics_calc.calculate_all_metrics(y_holdout, y_holdout_pred)
    holdout_auc     = holdout_m["auc"]
    holdout_f1r     = holdout_m["f1_reject"]
    holdout_brier   = holdout_m["brier_score"]
    holdout_ks      = holdout_m["ks"]

    zone_m = compute_zone_metrics_from_predictions(y_holdout, y_holdout_pred, best_lower, best_upper)

    # Scoring
    monitor_holdout_auc_gap = float(avg_monitor_auc - holdout_auc)
    calibration_gap         = float(avg_monitor_brier - train_brier_full)
    f1r_gap                 = float(avg_monitor_f1r - holdout_f1r)
    sc = scoring_cfg

    decision_score = (
        sc.w_low_zone_reject_precision  * zone_m["low_zone_reject_precision"]
        + sc.w_high_zone_approve_precision * zone_m["high_zone_approve_precision"]
        + sc.w_manual_review              * (1.0 - zone_m["manual_review_ratio"])
        + sc.w_holdout_f1_reject          * holdout_f1r
        + sc.w_holdout_brier              * (1.0 - holdout_brier)
        + sc.w_holdout_auc                * holdout_auc
        + sc.w_holdout_ks                 * holdout_ks
    )
    overfitting_penalty = sc.overfitting_penalty_weight * (
        max(train_monitor_auc_gap - sc.max_train_monitor_auc_gap, 0)
        + max(abs(monitor_holdout_auc_gap) - sc.max_monitor_holdout_auc_gap, 0)
        + max(stability_score - sc.max_stability_score, 0)
    )
    robustness_penalty = sc.robustness_penalty_weight * (
        max(f1r_gap - sc.max_f1r_gap, 0)
        + max(calibration_gap - sc.max_calibration_gap, 0)
    )
    final_selection_score = decision_score - overfitting_penalty - robustness_penalty

    # Hard constraint check
    passes_hard = True
    violations: List[str] = []
    if zone_m["high_zone_approve_precision"] < sc.min_high_zone_approve_precision:
        passes_hard = False
        violations.append(f"high_zone_approve_precision={zone_m['high_zone_approve_precision']:.4f} < {sc.min_high_zone_approve_precision}")
    if zone_m["manual_review_ratio"] > sc.max_manual_review_ratio:
        passes_hard = False
        violations.append(f"manual_review_ratio={zone_m['manual_review_ratio']:.4f} > {sc.max_manual_review_ratio}")
    if zone_m["auto_decision_rate"] < sc.min_auto_decision_rate:
        passes_hard = False
        violations.append(f"auto_decision_rate={zone_m['auto_decision_rate']:.4f} < {sc.min_auto_decision_rate}")
    if train_monitor_auc_gap > sc.max_hard_train_monitor_auc_gap:
        passes_hard = False
        violations.append(f"train_monitor_auc_gap={train_monitor_auc_gap:.4f} > {sc.max_hard_train_monitor_auc_gap} (extreme overfitting)")

    # Upgrade evaluation
    combined = {
        "holdout_f1_reject": holdout_f1r,
        "overfitting_penalty": overfitting_penalty,
        **zone_m,
    }
    uc_status, uc_reason, suitable_r, suitable_c = evaluate_upgrade_candidate(
        combined, BASELINE_V1_METRICS, sc
    )

    return {
        "config_id": config_id,
        "model_type": model_type,
        "calibration_method": calibration_method,
        "description": candidate_cfg.get("description", ""),
        "best_lower_threshold": best_lower,
        "best_upper_threshold": best_upper,
        # Zone precision
        "low_zone_reject_precision": zone_m["low_zone_reject_precision"],
        "high_zone_approve_precision": zone_m["high_zone_approve_precision"],
        # Zone ratio
        "high_zone_ratio": zone_m["high_zone_ratio"],
        "manual_zone_ratio": zone_m["manual_zone_ratio"],
        "low_zone_ratio": zone_m["low_zone_ratio"],
        # Human workload
        "human_review_workload_ratio": zone_m["human_review_workload_ratio"],
        "auto_light_touch_ratio": zone_m["auto_light_touch_ratio"],
        "auto_decision_rate": zone_m["auto_decision_rate"],
        "manual_review_ratio": zone_m["manual_review_ratio"],
        # Holdout
        "holdout_auc": holdout_auc,
        "holdout_f1_reject": holdout_f1r,
        "holdout_brier": holdout_brier,
        "holdout_ks": holdout_ks,
        # Monitor
        "avg_train_auc": avg_train_auc,
        "avg_monitor_auc": avg_monitor_auc,
        "avg_monitor_f1_reject": avg_monitor_f1r,
        "avg_monitor_ks": avg_monitor_ks,
        "avg_monitor_brier": avg_monitor_brier,
        "stability_score": stability_score,
        "std_monitor_f1_reject": std_monitor_f1r,
        # Gaps
        "train_monitor_auc_gap": train_monitor_auc_gap,
        "monitor_holdout_auc_gap": monitor_holdout_auc_gap,
        "calibration_gap": calibration_gap,
        "f1r_gap": f1r_gap,
        # Scoring
        "decision_score": decision_score,
        "overfitting_penalty": overfitting_penalty,
        "robustness_penalty": robustness_penalty,
        "final_selection_score": final_selection_score,
        # Upgrade
        "upgrade_candidate": uc_status,
        "upgrade_reason": uc_reason,
        "suitable_for_routing_improvement": suitable_r,
        "suitable_for_replacing_baseline_classifier": suitable_c,
        # Constraints
        "passes_hard_constraints": passes_hard,
        "constraint_violations": "; ".join(violations),
        "params": {k: v for k, v in candidate_cfg.items() if k not in ["config_id", "description", "model_type"]},
    }


def run_c3_decision_tuning_challenger(
    project_root: Optional[Path] = None,
    features_to_drop: Optional[List[str]] = None,
    tuning_candidates: Optional[List[Dict]] = None,
    calibration_methods: Optional[List[str]] = None,
) -> Dict:
    """
    Run the C3 Decision-Oriented + Anti-Overfitting Tuning challenger.

    Based on the C2 19-feature set.
    For each candidate x calibration, evaluates zone metrics, overfitting penalties,
    and selects the best per-candidate threshold policy.

    Saves:
      decision_tuning_comparison.csv
      challenger_summary.json
      C3_decision_tuning_routing_report.md
      C3_decision_tuning_vs_baseline_comparison.md

    Returns dict with output_dir and best candidate metrics.
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    if features_to_drop is None:
        features_to_drop = C2_FEATURES_TO_DROP
    if tuning_candidates is None:
        tuning_candidates = C3_TUNING_CANDIDATES
    if calibration_methods is None:
        calibration_methods = C3_CALIBRATION_METHODS

    logger.info("=" * 70)
    logger.info("C3 Decision-Oriented + Anti-Overfitting Tuning")
    logger.info("  Feature set: C2 (19 features)")
    logger.info("  Candidates: %d x %d calibrations = %d total",
                len(tuning_candidates), len(calibration_methods),
                len(tuning_candidates) * len(calibration_methods))
    logger.info("=" * 70)

    config_path = project_root / "config" / "pipeline_config.yaml"
    config = ConfigManager(config_path) if config_path.exists() else None
    scoring_cfg = DecisionScoringConfig()
    bc_cfg = BusinessConstraintConfig(
        max_manual_review_ratio=0.12,
        min_auto_decision_rate=0.85,
        min_low_zone_ratio=0.02,
        min_high_zone_precision=0.98,
        min_low_zone_reject_precision=0.65,
        target_auto_decision_rate=0.90,
    )

    trainer = PrunedFeatureTrainer(
        features_to_drop=features_to_drop,
        project_root=project_root,
        imbalance_strategy="scale_weight",
        config=config,
    )

    # Load data
    dev_df = trainer.load_development_data()
    X_dev, y_dev, feature_cols = trainer._prepare_xy(dev_df)
    trainer.feature_names = feature_cols
    logger.info("Features: %d  |  Dev rows: %d", len(feature_cols), len(y_dev))

    scale_pos_weight = trainer.imbalance_handler.calculate_scale_pos_weight(y_dev)
    class_weight     = trainer.imbalance_handler.calculate_class_weight(y_dev)

    # OOT split
    spark    = trainer._get_spark()
    oot_df   = spark.read.parquet(str(trainer.oot_path)).toPandas()
    oot_df["進件日"] = pd.to_datetime(oot_df["進件日"])
    oot_df   = oot_df.sort_values("進件日")
    cut      = int(len(oot_df) * 2 / 3)
    policy_df  = oot_df.iloc[:cut].copy()
    holdout_df = oot_df.iloc[cut:].copy()
    X_policy,  y_policy,  _ = trainer._prepare_xy(policy_df,  feature_cols)
    X_holdout, y_holdout, _ = trainer._prepare_xy(holdout_df, feature_cols)
    logger.info("Policy validation: %d rows  |  Holdout: %d rows", len(y_policy), len(y_holdout))

    # Rolling windows
    trainer.load_rolling_window_definition()

    # Evaluate all candidates
    all_records: List[Dict] = []
    total = len(tuning_candidates) * len(calibration_methods)
    count = 0
    for cand in tuning_candidates:
        for cal in calibration_methods:
            count += 1
            logger.info("[%d/%d] %s x %s", count, total, cand["config_id"], cal)
            try:
                rec = _evaluate_c3_candidate(
                    cand, cal, trainer,
                    X_dev, y_dev, X_policy, y_policy, X_holdout, y_holdout,
                    scale_pos_weight, class_weight, feature_cols, scoring_cfg, bc_cfg,
                )
                if rec is not None:
                    all_records.append(rec)
            except Exception as exc:
                logger.warning("Candidate %s_%s failed: %s", cand["config_id"], cal, exc)

    if not all_records:
        logger.error("All C3 candidates failed.")
        return {}

    passing = sorted(
        [r for r in all_records if r["passes_hard_constraints"]],
        key=lambda x: x["final_selection_score"], reverse=True,
    )
    failing = sorted(
        [r for r in all_records if not r["passes_hard_constraints"]],
        key=lambda x: x["final_selection_score"], reverse=True,
    )
    sorted_records = passing + failing

    # Output
    output_dir = project_root / "model_bank" / "experiments" / "c3_decision_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    priority_cols = [
        "config_id", "model_type", "calibration_method", "passes_hard_constraints",
        "upgrade_candidate", "upgrade_reason",
        "suitable_for_routing_improvement", "suitable_for_replacing_baseline_classifier",
        "final_selection_score", "decision_score", "overfitting_penalty", "robustness_penalty",
        "low_zone_reject_precision", "high_zone_approve_precision",
        "high_zone_ratio", "manual_zone_ratio", "low_zone_ratio",
        "human_review_workload_ratio", "auto_light_touch_ratio", "auto_decision_rate",
        "holdout_auc", "holdout_f1_reject", "holdout_brier", "holdout_ks",
        "best_lower_threshold", "best_upper_threshold",
        "train_monitor_auc_gap", "monitor_holdout_auc_gap",
        "stability_score", "calibration_gap", "f1r_gap",
        "avg_monitor_auc", "avg_monitor_f1_reject", "constraint_violations",
    ]
    csv_records = [
        {**{k: v for k, v in r.items() if k != "params"},
         **{f"param_{k}": v for k, v in r.get("params", {}).items()}}
        for r in sorted_records
    ]
    df_csv = pd.DataFrame(csv_records)
    existing = [c for c in priority_cols if c in df_csv.columns]
    other    = [c for c in df_csv.columns if c not in existing]
    df_csv   = df_csv[existing + other]
    csv_path = output_dir / "decision_tuning_comparison.csv"
    df_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Comparison CSV saved: %s", csv_path)

    # Summary JSON + reports using best candidate
    best = passing[0] if passing else sorted_records[0]

    bl_workload = BASELINE_V1_METRICS["manual_review_ratio"] + BASELINE_V1_METRICS["low_zone_ratio"]

    summary = {
        "challenger_id": "C3_decision_tuning",
        "created_at": datetime.now().isoformat(),
        "total_candidates_evaluated": len(all_records),
        "candidates_passing_hard_constraints": len(passing),
        "feature_count": len(feature_cols),
        "features_dropped": features_to_drop,
        "best_candidate": {
            "config_id": best["config_id"],
            "model_type": best["model_type"],
            "calibration_method": best["calibration_method"],
            "threshold": {"lower": best["best_lower_threshold"], "upper": best["best_upper_threshold"]},
            "final_selection_score": best["final_selection_score"],
            "holdout_metrics": {
                "auc": best["holdout_auc"],
                "f1_reject": best["holdout_f1_reject"],
                "brier": best["holdout_brier"],
                "ks": best["holdout_ks"],
            },
            "zone_precision": {
                "low_zone_reject_precision": best["low_zone_reject_precision"],
                "high_zone_approve_precision": best["high_zone_approve_precision"],
            },
            "zone_ratio": {
                "high_zone_ratio": best["high_zone_ratio"],
                "high_zone_pct": f"{best['high_zone_ratio']:.1%}",
                "manual_zone_ratio": best["manual_zone_ratio"],
                "manual_zone_pct": f"{best['manual_zone_ratio']:.1%}",
                "low_zone_ratio": best["low_zone_ratio"],
                "low_zone_pct": f"{best['low_zone_ratio']:.1%}",
            },
            "human_workload": {
                "human_review_workload_ratio": best["human_review_workload_ratio"],
                "human_review_workload_pct": f"{best['human_review_workload_ratio']:.1%}",
                "note": "manual zone + low zone; both require human intervention at different intensities",
                "vs_baseline": {
                    "baseline_human_workload": bl_workload,
                    "c3_human_workload": best["human_review_workload_ratio"],
                    "delta": best["human_review_workload_ratio"] - bl_workload,
                },
            },
            "upgrade_candidate": best["upgrade_candidate"],
            "upgrade_reason": best["upgrade_reason"],
            "suitable_for_routing_improvement": best["suitable_for_routing_improvement"],
            "suitable_for_replacing_baseline_classifier": best["suitable_for_replacing_baseline_classifier"],
            "business_recommendation": {
                "auto_approve_support": (
                    "YES: high zone precision meets threshold (>=0.98)"
                    if best["high_zone_approve_precision"] >= 0.98
                    else "REVIEW NEEDED: high zone precision below 0.98"
                ),
                "low_zone_usage": (
                    "Enhanced review / fast-track reject review pool. "
                    + ("Fully auto-reject may be considered."
                       if best["low_zone_reject_precision"] >= 0.65
                       else f"Do NOT auto-reject: precision={best['low_zone_reject_precision']:.4f} < 0.65.")
                ),
                "full_baseline_replacement": (
                    "YES: challenger outperforms baseline on all key metrics"
                    if best["upgrade_candidate"] == "yes"
                    else "NOT RECOMMENDED: F1_reject regressed; routing improvement only"
                ),
            },
        },
    }
    (output_dir / "challenger_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Challenger summary saved: %s", output_dir / "challenger_summary.json")

    # Comparison and routing reports
    compare_against_baseline(
        challenger_id="C3_decision_tuning",
        challenger_desc="Decision-oriented + anti-overfitting tuning on C2 feature set",
        challenger_holdout={
            "auc": best["holdout_auc"],
            "f1_reject": best["holdout_f1_reject"],
            "brier_score": best["holdout_brier"],
            "ks": best["holdout_ks"],
        },
        challenger_zone={
            "low_zone_reject_precision": best["low_zone_reject_precision"],
            "high_zone_approve_precision": best["high_zone_approve_precision"],
            "high_zone_ratio": best["high_zone_ratio"],
            "manual_zone_ratio": best["manual_zone_ratio"],
            "low_zone_ratio": best["low_zone_ratio"],
            "human_review_workload_ratio": best["human_review_workload_ratio"],
        },
        challenger_stability={
            "avg_monitor_auc": best["avg_monitor_auc"],
            "avg_monitor_f1_reject": best["avg_monitor_f1_reject"],
            "stability_score": best["stability_score"],
        },
        challenger_features_dropped=features_to_drop,
        challenger_features_remaining=len(feature_cols),
        baseline_metrics=BASELINE_V1_METRICS,
        output_dir=output_dir,
        upgrade_status=best["upgrade_candidate"],
        upgrade_reason=best["upgrade_reason"],
        suitable_routing=best["suitable_for_routing_improvement"],
        suitable_replace=best["suitable_for_replacing_baseline_classifier"],
    )

    generate_routing_report(
        challenger_id="C3_decision_tuning",
        best_candidate=best,
        baseline_metrics=BASELINE_V1_METRICS,
        output_dir=output_dir,
    )

    try:
        trainer._stop_spark()
    except Exception:
        pass

    logger.info("C3 output dir: %s", output_dir)
    return {
        "output_dir": str(output_dir),
        "best_candidate": best,
        "all_records": sorted_records,
        "upgrade_candidate": best["upgrade_candidate"],
    }
