"""
Final Decision Report Generator
=================================
Produces a complete, business-friendly governance report after challenger evaluation.

The report is centred on the *final chosen candidate* — the active baseline when no
challenger qualifies, or the best challenger when one does.  A single
``FinalDecision`` object carries that choice through every downstream step so that
plots, tables, and narrative all tell the same story.

Generates:
  1. Confusion matrix plot           (plots/confusion_matrix.png)
  2. Zone distribution bar chart     (plots/zone_distribution.png)
  3. Zone outcome heatmap            (plots/zone_outcome_heatmap.png)
  4. SHAP summary (beeswarm) plot    (plots/shap_summary.png)
  5. SHAP top-feature bar chart      (plots/shap_feature_importance.png)
  6. model_governance_decision_report.md
  7. final_baseline_decision_summary.json   (machine-readable single source of truth)

All artifacts are written under:
  model_bank/experiments/final_decision_report/

Public API:
  finalize_model_decision(project_root)    -> FinalDecision
  build_final_decision_artifacts(decision) -> dict
  run_final_decision_report(project_root)  -> dict   (main entry point)
"""

import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")           # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as _fm
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CJK font setup — applied once at module import time
# ─────────────────────────────────────────────────────────────

def _configure_cjk_font() -> None:
    """
    Detect and configure a CJK-capable font for matplotlib so that Traditional /
    Simplified Chinese characters in feature names, axis labels, and plot titles
    render correctly in all saved PNG images — on both Windows and Linux/Docker.

    Priority order — Linux/Docker names first, then Windows fallbacks:
      1. Noto Sans CJK TC   – fonts-noto-cjk on Debian/Ubuntu (Traditional Chinese)
      2. Noto Sans CJK SC   – fonts-noto-cjk on Debian/Ubuntu (Simplified Chinese)
      3. Noto Sans CJK HK   – fonts-noto-cjk on Debian/Ubuntu (Hong Kong)
      4. Noto Sans CJK JP   – fonts-noto-cjk on Debian/Ubuntu (Japanese; covers CJK)
      5. WenQuanYi Zen Hei  – fonts-wqy-zenhei on Debian/Ubuntu
      6. WenQuanYi Micro Hei– alternative WQY package
      7. Microsoft JhengHei – Windows Traditional Chinese system font
      8. Noto Sans TC        – Windows Noto (without "CJK" in name)
      9. Noto Sans HK        – Windows Noto HK
     10. Microsoft YaHei     – Windows Simplified Chinese
     11. SimHei              – Windows Simplified Chinese (broad coverage)
     12. SimSun              – Windows fallback
     13. Arial Unicode MS    – broad Unicode fallback (Office)
     14. Malgun Gothic       – Korean (last resort; covers some CJK)

    Also sets ``axes.unicode_minus = False`` so that minus signs (−) in axis
    tick labels and annotations are rendered by the selected font rather than
    as the Unicode minus glyph, which many CJK fonts do not include.
    """
    candidates = [
        # ── Linux / Docker (fonts-noto-cjk Debian package) ──────────
        "Noto Sans CJK TC",      # Traditional Chinese — exact match on Linux
        "Noto Sans CJK SC",      # Simplified Chinese
        "Noto Sans CJK HK",      # Hong Kong
        "Noto Sans CJK JP",      # Japanese (covers all CJK codepoints)
        "WenQuanYi Zen Hei",     # fonts-wqy-zenhei
        "WenQuanYi Micro Hei",   # fonts-wqy-microhei
        # ── Windows ─────────────────────────────────────────────────
        "Microsoft JhengHei",    # Windows Traditional Chinese
        "Noto Sans TC",          # Windows Noto (no "CJK" in name)
        "Noto Sans HK",          # Windows Noto HK
        "Microsoft YaHei",       # Windows Simplified Chinese
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "Malgun Gothic",
    ]

    available = {f.name for f in _fm.fontManager.ttflist}
    chosen: Optional[str] = None
    for name in candidates:
        if name in available:
            chosen = name
            break

    if chosen:
        matplotlib.rcParams["font.family"] = "sans-serif"
        # Prepend the chosen font so it is tried first; keep DejaVu as fallback
        # for Latin/ASCII characters the CJK font might not cover perfectly.
        current_sans = matplotlib.rcParams.get("font.sans-serif", [])
        new_sans = [chosen] + [f for f in current_sans if f != chosen]
        matplotlib.rcParams["font.sans-serif"] = new_sans
        logger.info("CJK font configured: %s", chosen)
    else:
        logger.warning(
            "No CJK font found in matplotlib font manager. "
            "Chinese characters may appear as squares in generated plots. "
            "Install one of: %s", ", ".join(candidates)
        )

    # Prevent minus signs from being rendered as the Unicode minus glyph (U+2212),
    # which many CJK fonts lack — use ASCII hyphen-minus instead.
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_cjk_font()


# ─────────────────────────────────────────────────────────────
# FinalDecision dataclass  — single source of truth
# ─────────────────────────────────────────────────────────────

@dataclass
class FinalDecision:
    """
    Carries the resolved final governance decision and all artefact references
    needed to drive plots, the Markdown report, and the JSON summary.

    decision_type values:
      "RETAIN_BASELINE"        – no challenger beat the baseline
      "FULL_UPGRADE"           – challenger beats on both routing AND classifier
      "ROUTING_ONLY_UPGRADE"   – challenger improves routing but classifier regressed
    """
    # Core identity
    active_baseline_name: str = ""
    chosen_candidate_id: str = ""          # e.g. "baseline_v1", "C2_feature_pruning"
    chosen_candidate_label: str = ""       # human-readable, e.g. "Baseline (baseline_v1)"
    decision_type: str = "RETAIN_BASELINE"
    decision_reason: str = ""

    # Artefact paths for the *chosen* candidate
    holdout_csv: Optional[Path] = None     # final_holdout_predictions.csv
    model_pkl:   Optional[object] = None   # loaded model object
    feature_names: Optional[List[str]] = None

    # Baseline metadata (always the active baseline, regardless of chosen candidate)
    baseline_meta: dict = field(default_factory=dict)

    # Challenger evaluation summary (full dict from challenger_evaluation_summary.json)
    eval_summary: dict = field(default_factory=dict)

    # Individual challenger raw metadata (for report tables)
    c2_meta:    dict = field(default_factory=dict)
    c3_summary: dict = field(default_factory=dict)

    # Thresholds resolved from chosen candidate
    lower_threshold: float = 0.5
    upper_threshold: float = 0.85

    # Whether challenger artifacts were found (affects report completeness)
    has_c2: bool = False
    has_c3: bool = False

# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _find_baseline_dir(project_root: Path) -> Tuple[Optional[Path], str]:
    """Return (baseline_dir, baseline_name) for the active baseline."""
    active_file = project_root / "model_bank" / "baselines" / "active_baseline.json"
    if not active_file.exists():
        return None, ""
    active = _read_json(active_file)
    name = active.get("active_baseline", "")
    bl_dir = project_root / "model_bank" / "baselines" / name
    return (bl_dir if bl_dir.exists() else None), name


def _load_holdout_predictions(baseline_dir: Path) -> Optional[pd.DataFrame]:
    """Load final_holdout_predictions.csv from the baseline predictions folder."""
    path = baseline_dir / "predictions" / "final_holdout_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def _load_gold_oot(project_root: Path) -> Optional[pd.DataFrame]:
    """Load the full gold OOT dataset (all months, all feature columns + label)."""
    oot_dir = project_root / "datamart" / "gold" / "oot"
    if not oot_dir.exists():
        return None
    parts = [pd.read_parquet(str(p)) for p in oot_dir.glob("**/*.parquet")]
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def _load_model_and_features(
    baseline_dir: Path,
) -> Tuple[Optional[object], Optional[List[str]]]:
    """Load the champion model pkl and feature name list from a baseline directory."""
    model_path   = baseline_dir / "model" / "final_champion_model.pkl"
    feature_path = baseline_dir / "artifacts" / "feature_names.json"
    model = None
    features = None
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    if feature_path.exists():
        features = json.loads(feature_path.read_text(encoding="utf-8"))
    return model, features


# ─────────────────────────────────────────────────────────────
# Core orchestration — FinalDecision resolution
# ─────────────────────────────────────────────────────────────

def finalize_model_decision(project_root: Path) -> Optional["FinalDecision"]:
    """
    Resolve the final governance decision and return a populated FinalDecision.

    Decision logic (mirrors challenger_manager.evaluate_upgrade_candidate):
      1. Load active baseline + challenger evaluation summary.
      2. Determine the final chosen candidate:
           - If best_challenger is None          → chosen = active baseline
           - If best_challenger is C2/C3         → chosen = that challenger
      3. Load artefacts (holdout CSV, model pkl, feature names) for the
         chosen candidate, falling back to the active baseline when challenger
         artefacts are not available.
      4. Return a fully-populated FinalDecision.

    Artefact resolution for challengers:
      - Holdout CSV: challenger output_dir / predictions / final_holdout_predictions.csv
                     (fallback: active baseline holdout CSV)
      - Model pkl  : challenger output_dir / model / final_champion_model.pkl
                     (fallback: active baseline model pkl)
      - Features   : challenger output_dir / artifacts / feature_names.json
                     (fallback: active baseline feature_names.json)

    Returns None if no active baseline is found.
    """
    # ── Active baseline ───────────────────────────────────────────────
    baseline_dir, baseline_name = _find_baseline_dir(project_root)
    if baseline_dir is None:
        logger.error("finalize_model_decision: no active baseline found.")
        return None

    baseline_meta = _read_json(baseline_dir / "baseline_metadata.json")
    tp = baseline_meta.get("threshold_policy", {})
    bl_lower = tp.get("lower_threshold", 0.5)
    bl_upper = tp.get("upper_threshold", 0.85)

    # ── Challenger evaluation summary ─────────────────────────────────
    eval_path = project_root / "model_bank" / "experiments" / "challenger_evaluation_summary.json"
    eval_summary = _read_json(eval_path)

    c2_meta    = _read_json(
        project_root / "model_bank" / "experiments" / "c2_feature_pruning" / "c2_challenger_metadata.json"
    )
    c3_summary = _read_json(
        project_root / "model_bank" / "experiments" / "c3_decision_tuning" / "challenger_summary.json"
    )

    best_chal  = eval_summary.get("best_challenger")       # None | "C2_feature_pruning" | "C3_decision_tuning"
    c2_status  = eval_summary.get("c2_feature_pruning",  {}).get("upgrade_candidate", "reject")
    c3_status  = eval_summary.get("c3_decision_tuning",  {}).get("upgrade_candidate", "reject")
    c2_reason  = eval_summary.get("c2_feature_pruning",  {}).get("upgrade_reason", "")
    c3_reason  = eval_summary.get("c3_decision_tuning",  {}).get("upgrade_reason", "")
    overall    = eval_summary.get("overall_recommendation", "KEEP BASELINE")

    # ── Map best_challenger → decision_type ───────────────────────────
    if best_chal is None:
        decision_type = "RETAIN_BASELINE"
        chosen_id     = baseline_name
        chosen_label  = f"Baseline ({baseline_name})"
        decision_reason = (c2_reason or c3_reason or
                           "Neither challenger met routing-first promotion criteria.")
    elif "full_upgrade" in (c2_status if best_chal == "C2_feature_pruning" else c3_status):
        decision_type  = "FULL_UPGRADE"
        chosen_id      = best_chal
        chosen_label   = ("C2 Feature Pruning" if best_chal == "C2_feature_pruning"
                          else "C3 Decision Tuning")
        decision_reason = (c2_reason if best_chal == "C2_feature_pruning" else c3_reason)
    else:
        decision_type  = "ROUTING_ONLY_UPGRADE"
        chosen_id      = best_chal
        chosen_label   = ("C2 Feature Pruning" if best_chal == "C2_feature_pruning"
                          else "C3 Decision Tuning")
        decision_reason = (c2_reason if best_chal == "C2_feature_pruning" else c3_reason)

    # ── Locate challenger artefact directory ──────────────────────────
    challenger_dir: Optional[Path] = None
    if best_chal == "C2_feature_pruning":
        candidate_output = eval_summary.get("c2_feature_pruning", {}).get("output_dir", "")
        if candidate_output:
            p = Path(candidate_output)
            if p.exists():
                challenger_dir = p
        if challenger_dir is None:
            challenger_dir = project_root / "model_bank" / "experiments" / "c2_feature_pruning"
    elif best_chal == "C3_decision_tuning":
        candidate_output = eval_summary.get("c3_decision_tuning", {}).get("output_dir", "")
        if candidate_output:
            p = Path(candidate_output)
            if p.exists():
                challenger_dir = p
        if challenger_dir is None:
            challenger_dir = project_root / "model_bank" / "experiments" / "c3_decision_tuning"

    # ── Load holdout CSV for the chosen candidate ─────────────────────
    holdout_df: Optional[pd.DataFrame] = None
    lower_threshold = bl_lower
    upper_threshold = bl_upper

    if challenger_dir is not None:
        # Try challenger-specific holdout predictions first
        for cand_path in [
            challenger_dir / "predictions" / "final_holdout_predictions.csv",
            challenger_dir / "final_holdout_predictions.csv",
        ]:
            if cand_path.exists():
                holdout_df = pd.read_csv(cand_path)
                logger.info("Loaded challenger holdout: %s", cand_path)
                break

    if holdout_df is None:
        # Fall back to active baseline holdout (always available after training)
        holdout_df = _load_holdout_predictions(baseline_dir)
        if holdout_df is not None:
            logger.info("Using baseline holdout predictions (challenger holdout not found).")

    # ── Load model + feature names for the chosen candidate ───────────
    model: Optional[object] = None
    feature_names: Optional[List[str]] = None

    if challenger_dir is not None:
        model, feature_names = _load_model_and_features(challenger_dir)

    if model is None:
        # Challenger has no model pkl — use baseline model for SHAP
        model, feature_names = _load_model_and_features(baseline_dir)
        if model is not None:
            logger.info("Using baseline model for SHAP (challenger model not found).")

    # ── Build and return FinalDecision ───────────────────────────────
    return FinalDecision(
        active_baseline_name  = baseline_name,
        chosen_candidate_id   = chosen_id,
        chosen_candidate_label= chosen_label,
        decision_type         = decision_type,
        decision_reason       = decision_reason,
        holdout_csv           = holdout_df,     # DataFrame, not a path
        model_pkl             = model,
        feature_names         = feature_names,
        baseline_meta         = baseline_meta,
        eval_summary          = eval_summary,
        c2_meta               = c2_meta,
        c3_summary            = c3_summary,
        lower_threshold       = lower_threshold,
        upper_threshold       = upper_threshold,
        has_c2                = bool(c2_meta),
        has_c3                = bool(c3_summary),
    )


# ─────────────────────────────────────────────────────────────
# Plot 1 — Confusion Matrix
# ─────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    holdout_df: pd.DataFrame,
    lower_threshold: float,
    upper_threshold: float,
    candidate_label: str,
    out_path: Path,
) -> Dict:
    """
    Generate a confusion matrix plot using the final holdout data.

    Zone mapping (pred_zone): 0 = low, 1 = manual, 2 = high
    Binary decision: zone 2 (high / score >= upper) → predicted approve (1)
                     zone 0 (low  / score <  lower) → predicted reject  (0)
                     zone 1 (manual)                → predicted approve  (1)
                                                       [conservative: manual treated as approve]

    The confusion matrix is computed against actual_label.
    """
    y_true = holdout_df["actual_label"].values
    # Binary prediction: zones 1 & 2 → approve=1 (positive), zone 0 → reject=0
    y_pred = (holdout_df["pred_zone"] >= 1).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    classes = ["Reject (0)", "Approve (1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — Final Blind Holdout\n({candidate_label})", fontsize=11)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=13)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", out_path)

    precision_reject = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_reject    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "true_positive":  int(tp),
        "true_negative":  int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "precision_reject_zone": round(precision_reject, 4),
        "recall_reject_zone":    round(recall_reject, 4),
    }


# ─────────────────────────────────────────────────────────────
# Plot 2 — Zone Distribution
# ─────────────────────────────────────────────────────────────

def _plot_zone_distribution(
    holdout_df: pd.DataFrame,
    baseline_meta: dict,
    candidate_label: str,
    out_path: Path,
) -> None:
    """Grouped bar chart: zone population ratios — candidate vs baseline."""
    total = len(holdout_df)
    cand_high   = (holdout_df["pred_zone"] == 2).sum() / total
    cand_manual = (holdout_df["pred_zone"] == 1).sum() / total
    cand_low    = (holdout_df["pred_zone"] == 0).sum() / total

    bc = baseline_meta.get("business_constraints", {})
    bl_high   = bc.get("high_zone_ratio", 0)
    bl_manual = bc.get("review_zone_ratio", bc.get("manual_review_load", 0))
    bl_low    = bc.get("low_zone_ratio", 0)

    zones  = ["High Zone\n(Auto-Approve)", "Manual Zone\n(Human Review)", "Low Zone\n(Auto-Reject)"]
    cand_v = [cand_high, cand_manual, cand_low]
    bl_v   = [bl_high,   bl_manual,   bl_low]

    x = np.arange(len(zones))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, [v * 100 for v in cand_v], width,
                   label=candidate_label, color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, [v * 100 for v in bl_v],   width,
                   label="Baseline", color="#DD8452", alpha=0.85)

    ax.set_ylabel("Population (%)", fontsize=11)
    ax.set_title("Zone Distribution — Final Blind Holdout vs Baseline", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(zones, fontsize=10)
    ax.legend(fontsize=10)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Zone distribution chart saved: %s", out_path)


# ─────────────────────────────────────────────────────────────
# Plot 3 — Zone Outcome Heatmap
# ─────────────────────────────────────────────────────────────

def _plot_zone_outcome_heatmap(
    holdout_df: pd.DataFrame,
    candidate_label: str,
    out_path: Path,
) -> None:
    """Heatmap of actual approve/reject rates per routing zone."""
    zone_map = {2: "High Zone", 1: "Manual Zone", 0: "Low Zone"}
    rows = []
    for zone_code in [2, 1, 0]:
        sub = holdout_df[holdout_df["pred_zone"] == zone_code]
        if len(sub) == 0:
            approve_pct = reject_pct = 0.0
        else:
            approve_pct = sub["actual_label"].mean() * 100
            reject_pct  = (1 - sub["actual_label"].mean()) * 100
        rows.append([approve_pct, reject_pct])

    data = np.array(rows)
    zone_labels = ["High Zone\n(Auto-Approve)", "Manual Zone\n(Human Review)", "Low Zone\n(Auto-Reject)"]
    outcome_labels = ["Actual Approve %", "Actual Reject %"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    import seaborn as sns
    sns.heatmap(
        data, annot=True, fmt=".1f", cmap="RdYlGn",
        xticklabels=outcome_labels, yticklabels=zone_labels,
        ax=ax, linewidths=0.5, cbar_kws={"label": "%"},
        vmin=0, vmax=100,
    )
    ax.set_title(f"Zone Outcome Summary — Final Blind Holdout\n({candidate_label})", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Zone outcome heatmap saved: %s", out_path)


# ─────────────────────────────────────────────────────────────
# Plots 4 & 5 — SHAP
# ─────────────────────────────────────────────────────────────

def _plot_shap(
    model: object,
    X_sample: pd.DataFrame,
    feature_names: List[str],
    candidate_label: str,
    out_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Generate SHAP beeswarm summary + bar feature importance chart.

    Uses TreeExplainer for XGBoost / RandomForest (fast, exact).
    Falls back to a plain model feature_importances_ bar chart if SHAP fails.

    Returns (shap_summary_path, shap_bar_path).
    """
    try:
        import shap

        # Extract underlying estimator if wrapped in CalibratedClassifierCV
        raw_model = model
        if hasattr(model, "calibrated_classifiers_"):
            raw_model = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, "base_estimator"):
            raw_model = model.base_estimator
        elif hasattr(model, "estimator"):
            raw_model = model.estimator

        explainer = shap.TreeExplainer(raw_model)
        shap_values = explainer.shap_values(X_sample)

        # For binary classifiers shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        # ── Beeswarm summary ──────────────────────────────────────────
        summary_path = out_dir / "shap_summary.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(sv, X_sample, feature_names=feature_names,
                          show=False, plot_size=None)
        plt.title(f"SHAP Feature Impact — {candidate_label}", fontsize=11, pad=10)
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info("SHAP summary saved: %s", summary_path)

        # ── Bar chart (mean |SHAP|) ───────────────────────────────────
        bar_path = out_dir / "shap_feature_importance.png"
        mean_abs = np.abs(sv).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1]
        top_n = min(15, len(feature_names))
        top_idx   = sorted_idx[:top_n]
        top_names = [feature_names[i] for i in top_idx]
        top_vals  = mean_abs[top_idx]

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
        ax.barh(range(top_n), top_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(f"Top {top_n} Features by SHAP Importance\n({candidate_label})", fontsize=11)
        plt.tight_layout()
        fig.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("SHAP bar chart saved: %s", bar_path)

        return summary_path, bar_path

    except Exception as exc:
        logger.warning("SHAP generation failed (%s); falling back to model feature_importances_.", exc)
        return _plot_feature_importances_fallback(model, feature_names, candidate_label, out_dir)


def _plot_feature_importances_fallback(
    model: object,
    feature_names: List[str],
    candidate_label: str,
    out_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Fallback: plot model.feature_importances_ if SHAP is unavailable."""
    raw = model
    for attr in ("calibrated_classifiers_", "base_estimator", "estimator"):
        if hasattr(model, attr):
            cand = getattr(model, attr)
            raw = cand[0].estimator if isinstance(cand, list) else cand
            break

    if not hasattr(raw, "feature_importances_"):
        return None, None

    importances = raw.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_names))
    top_idx   = sorted_idx[:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = importances[top_idx]

    bar_path = out_dir / "shap_feature_importance.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
    ax.barh(range(top_n), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title(f"Top {top_n} Features by Model Importance\n({candidate_label})", fontsize=11)
    plt.tight_layout()
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance fallback chart saved: %s", bar_path)
    return None, bar_path


# ─────────────────────────────────────────────────────────────
# Markdown Report Writer
# ─────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v:.2%}"

def _fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"

def _delta_str(new: float, old: float, higher_is_better: bool = True) -> str:
    d = new - old
    symbol = ("+" if d >= 0 else "") + _fmt(d)
    if d == 0:
        return f"({symbol}) —"
    good = (d > 0) == higher_is_better
    return f"({symbol}) {'BETTER' if good else 'WORSE'}"


def _write_report(
    decision: "FinalDecision",
    report_dir: Path,
    plots_dir: Path,
    cm_values: dict,
    shap_summary_path: Optional[Path],
    shap_bar_path: Optional[Path],
) -> Path:
    """
    Render the full 7-section Markdown governance report.

    The report narrative is centred on ``decision.chosen_candidate_label`` —
    the active baseline when no challenger qualifies, or the best challenger
    when one does.  All evidence sections (confusion matrix, zone tables, SHAP)
    reflect the artefacts of that chosen candidate.
    """
    out_path = report_dir / "model_governance_decision_report.md"

    baseline_name  = decision.active_baseline_name
    candidate_label= decision.chosen_candidate_label
    decision_type  = decision.decision_type
    decision_reason= decision.decision_reason
    holdout_df     = decision.holdout_csv
    baseline_meta  = decision.baseline_meta
    eval_summary   = decision.eval_summary
    c2_meta        = decision.c2_meta
    c3_summary     = decision.c3_summary

    h    = baseline_meta.get("holdout_metrics", {})
    rc   = baseline_meta.get("rolling_cv_metrics", {})
    tp   = baseline_meta.get("threshold_policy", {})
    bc   = baseline_meta.get("business_constraints", {})
    diag = baseline_meta.get("diagnostics", {})
    cm_model = baseline_meta.get("champion_model", {})
    tuning   = baseline_meta.get("tuning", {})

    overall_rec = eval_summary.get("overall_recommendation", "")
    c2_status   = eval_summary.get("c2_feature_pruning", {}).get("upgrade_candidate", "unknown")
    c2_reason   = eval_summary.get("c2_feature_pruning", {}).get("upgrade_reason", "")
    c3_status   = eval_summary.get("c3_decision_tuning", {}).get("upgrade_candidate", "unknown")
    c3_reason   = eval_summary.get("c3_decision_tuning", {}).get("upgrade_reason", "")

    # Decision type → human-readable headline
    _dt_label = {
        "RETAIN_BASELINE":      "RETAIN BASELINE",
        "FULL_UPGRADE":         "FULL UPGRADE RECOMMENDED",
        "ROUTING_ONLY_UPGRADE": "ROUTING-ONLY UPGRADE CANDIDATE",
    }
    decision_headline = _dt_label.get(decision_type, decision_type)

    # ── Zone stats from chosen candidate's holdout ───────────────────
    total    = len(holdout_df)
    high_n   = (holdout_df["pred_zone"] == 2).sum()
    manual_n = (holdout_df["pred_zone"] == 1).sum()
    low_n    = (holdout_df["pred_zone"] == 0).sum()

    def _zone_approve_pct(zone_code: int) -> float:
        sub = holdout_df[holdout_df["pred_zone"] == zone_code]
        return sub["actual_label"].mean() if len(sub) > 0 else 0.0

    hi_approve  = _zone_approve_pct(2)
    man_approve = _zone_approve_pct(1)
    low_approve = _zone_approve_pct(0)

    # ── Relative image paths (from report file location) ─────────────
    def _rel(p: Optional[Path]) -> str:
        if p is None:
            return "_not available_"
        try:
            return str(p.relative_to(report_dir)).replace("\\", "/")
        except ValueError:
            return str(p).replace("\\", "/")

    cm_img   = _rel(plots_dir / "confusion_matrix.png")
    zone_img = _rel(plots_dir / "zone_distribution.png")
    heat_img = _rel(plots_dir / "zone_outcome_heatmap.png")
    shap_img = _rel(shap_summary_path)
    bar_img  = _rel(shap_bar_path)

    # ── C2 detail ────────────────────────────────────────────────────
    c2h = c2_meta.get("holdout_metrics", {})
    c2z = c2_meta.get("zone_metrics", {})

    # ── C3 detail ────────────────────────────────────────────────────
    c3b  = c3_summary.get("best_candidate", {})
    c3hm = c3b.get("holdout_metrics", {})
    c3zp = c3b.get("zone_precision", {})
    c3zr = c3b.get("zone_ratio", {})

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Executive summary blurb ───────────────────────────────────────
    if decision_type == "RETAIN_BASELINE":
        exec_blurb = (
            "The active baseline was evaluated against two challengers (C2 Feature Pruning, "
            "C3 Decision Tuning). Neither challenger met the routing-first promotion criteria. "
            f"**The current baseline `{baseline_name}` is retained as the production model.**"
        )
    elif decision_type == "FULL_UPGRADE":
        exec_blurb = (
            f"**{candidate_label}** has demonstrated sufficient improvement over the active "
            f"baseline `{baseline_name}` on both routing quality and classifier guardrails. "
            "A full baseline replacement is recommended."
        )
    else:  # ROUTING_ONLY_UPGRADE
        exec_blurb = (
            f"**{candidate_label}** shows meaningful improvement in routing quality over the "
            f"active baseline `{baseline_name}`. However, classifier metrics show minor regression. "
            "A routing-policy-only update is recommended; full model replacement is not advised at this stage."
        )

    # ── Section 5 header note for chosen candidate ───────────────────
    if decision_type == "RETAIN_BASELINE":
        evidence_note = (
            f"> Evidence below is for **{candidate_label}** — the current production model.  \n"
            "> This dataset was never used for training, validation, or threshold tuning."
        )
    else:
        evidence_note = (
            f"> Evidence below is for **{candidate_label}** — the recommended promotion candidate.  \n"
            f"> The active baseline is `{baseline_name}`.  \n"
            "> This dataset was never used for training, validation, or threshold tuning."
        )

    lines: List[str] = [
        "# Model Governance Decision Report",
        "",
        f"> **Generated**: {now}  ",
        f"> **Pipeline**: Four-Phase Credit Risk ML  ",
        f"> **Active Baseline**: `{baseline_name}`  ",
        f"> **Final Chosen Candidate**: **{candidate_label}**  ",
        f"> **Decision**: **{decision_headline}**",
        "",
        "---",
        "",
        # ============================================================
        "## 1. Executive Summary",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Evaluation date | {now} |",
        f"| Active baseline | `{baseline_name}` |",
        f"| Challengers evaluated | C2 Feature Pruning, C3 Decision Tuning |",
        f"| C2 outcome | **{c2_status.upper()}** |",
        f"| C3 outcome | **{c3_status.upper()}** |",
        f"| Final chosen candidate | **{candidate_label}** |",
        f"| Decision type | **{decision_headline}** |",
        "",
        exec_blurb,
        "",
        "---",
        "",
        # ============================================================
        "## 2. Current Baseline Profile",
        "",
        "> This section always describes the **active baseline**, regardless of which candidate was chosen.",
        "",
        f"**Baseline**: `{baseline_name}`  ",
        f"**Model type**: {cm_model.get('model_type', '')} + {cm_model.get('calibration_method', '')}  ",
        f"**Features**: {cm_model.get('feature_count', '')}  ",
        f"**Tuning config**: `{tuning.get('config_id', '')}`  ",
        "",
        "### 2a. Classifier Metrics (Final Blind Holdout)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| AUC | {_fmt(h.get('auc', 0))} |",
        f"| F1 (Reject class) | {_fmt(h.get('f1_reject', 0))} |",
        f"| Precision (Reject) | {_fmt(h.get('precision_reject', 0))} |",
        f"| Recall (Reject) | {_fmt(h.get('recall_reject', 0))} |",
        f"| KS | {_fmt(h.get('ks', 0))} |",
        f"| Brier Score | {_fmt(h.get('brier_score', 0))} |",
        "",
        "### 2b. Routing / Zone Profile",
        "",
        "| Zone | Population Ratio | Precision |",
        "|---|---|---|",
        f"| High Zone (Auto-Approve) | {_pct(bc.get('high_zone_ratio', 0))} | {_fmt(bc.get('high_zone_precision', 0))} |",
        f"| Manual Zone (Human Review) | {_pct(bc.get('review_zone_ratio', bc.get('manual_review_load', 0)))} | — |",
        f"| Low Zone (Auto-Reject) | {_pct(bc.get('low_zone_ratio', 0))} | {_fmt(bc.get('low_zone_precision', 0))} |",
        "",
        "### 2c. Threshold Policy",
        "",
        f"- Lower threshold: **{tp.get('lower_threshold', '')}**",
        f"- Upper threshold: **{tp.get('upper_threshold', '')}**",
        f"- Source: {tp.get('source', '')}",
        f"- Passes hard constraints: {'YES' if tp.get('passes_hard_constraints') else 'NO'}",
        "",
        "### 2d. Rolling Stability",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Avg Monitor AUC (CV) | {_fmt(rc.get('avg_monitor_auc', 0))} |",
        f"| Avg Monitor F1 Reject (CV) | {_fmt(rc.get('avg_monitor_f1_reject', 0))} |",
        f"| Stability Score (std) | {_fmt(rc.get('stability_score', 0))} |",
        "",
        "### 2e. Diagnostics",
        "",
        f"- Overfitting: {'WARNING — ' + diag.get('overfitting_severity', '') if diag.get('is_overfitting') else 'OK — None detected'}",
        f"- Calibration issue: {'WARNING' if diag.get('has_calibration_issue') else 'OK'}",
        f"- Reject detection issue: {'WARNING' if diag.get('has_reject_detection_issue') else 'OK'}",
        "",
        "---",
        "",
        # ============================================================
        "## 3. Challenger Summary",
        "",
        "> Delta columns compare challenger vs the active baseline.",
        "",
        "### 3a. C2 — Feature Pruning (25 → 19 features)",
        "",
        "| Metric | Baseline | C2 | Delta |",
        "|---|---|---|---|",
        f"| AUC | {_fmt(h.get('auc', 0))} | {_fmt(c2h.get('auc', 0))} "
        f"| {_delta_str(c2h.get('auc', 0), h.get('auc', 0))} |",
        f"| F1 Reject | {_fmt(h.get('f1_reject', 0))} | {_fmt(c2h.get('f1_reject', 0))} "
        f"| {_delta_str(c2h.get('f1_reject', 0), h.get('f1_reject', 0))} |",
        f"| KS | {_fmt(h.get('ks', 0))} | {_fmt(c2h.get('ks', 0))} "
        f"| {_delta_str(c2h.get('ks', 0), h.get('ks', 0))} |",
        f"| Low Zone Precision | {_fmt(bc.get('low_zone_precision', 0))} "
        f"| {_fmt(c2z.get('low_zone_reject_precision', 0))} "
        f"| {_delta_str(c2z.get('low_zone_reject_precision', 0), bc.get('low_zone_precision', 0))} |",
        f"| Human Workload | {_pct(bc.get('review_zone_ratio', 0) + bc.get('low_zone_ratio', 0))} "
        f"| {_pct(c2z.get('human_review_workload_ratio', 0))} | — |",
        "",
        f"**Decision**: {c2_status.upper()}  ",
        f"**Reason**: {c2_reason or 'See comparison report.'}",
        "",
        "### 3b. C3 — Decision Tuning",
        "",
        "| Metric | Baseline | C3 | Delta |",
        "|---|---|---|---|",
        f"| AUC | {_fmt(h.get('auc', 0))} | {_fmt(c3hm.get('auc', 0))} "
        f"| {_delta_str(c3hm.get('auc', 0), h.get('auc', 0))} |",
        f"| F1 Reject | {_fmt(h.get('f1_reject', 0))} | {_fmt(c3hm.get('f1_reject', 0))} "
        f"| {_delta_str(c3hm.get('f1_reject', 0), h.get('f1_reject', 0))} |",
        f"| KS | {_fmt(h.get('ks', 0))} | {_fmt(c3hm.get('ks', 0))} "
        f"| {_delta_str(c3hm.get('ks', 0), h.get('ks', 0))} |",
        f"| Low Zone Precision | {_fmt(bc.get('low_zone_precision', 0))} "
        f"| {_fmt(c3zp.get('low_zone_reject_precision', 0))} "
        f"| {_delta_str(c3zp.get('low_zone_reject_precision', 0), bc.get('low_zone_precision', 0))} |",
        f"| High Zone Precision | {_fmt(bc.get('high_zone_precision', 0))} "
        f"| {_fmt(c3zp.get('high_zone_approve_precision', 0))} "
        f"| {_delta_str(c3zp.get('high_zone_approve_precision', 0), bc.get('high_zone_precision', 0))} |",
        f"| High Zone Ratio | {_pct(bc.get('high_zone_ratio', 0))} | {_pct(c3zr.get('high_zone_ratio', 0))} | — |",
        f"| Human Workload | {_pct(bc.get('review_zone_ratio', 0) + bc.get('low_zone_ratio', 0))} "
        f"| {_pct(c3b.get('human_workload', {}).get('human_review_workload_ratio', 0))} | — |",
        "",
        f"**Decision**: {c3_status.upper()}  ",
        f"**Reason**: {c3_reason or 'See comparison report.'}",
        "",
        "---",
        "",
        # ============================================================
        "## 4. Final Comparison and Decision",
        "",
        "### Routing-First Evaluation Logic",
        "",
        "Promotion decisions follow a routing-first framework:",
        "",
        "1. **Primary check** — Does the challenger improve routing quality?",
        "   - Low zone reject precision must increase (target > baseline + 0.005)",
        "   - Human review workload ratio must not worsen beyond 2 pp tolerance",
        "2. **Guardrails** — AUC ≥ baseline − 0.01, F1_reject ≥ baseline − 0.02, "
        "high zone approve precision ≥ 0.98",
        "3. **Decision labels**:",
        "   - `full_upgrade` — routing improved AND all classifier guardrails pass",
        "   - `routing_only_upgrade` — routing improved, minor classifier regression accepted",
        "   - `reject` — routing did not improve or hard guardrails failed",
        "",
        "### Decision Summary Table",
        "",
        "| Challenger | Routing Improved | Guardrails Passed | Decision |",
        "|---|---|---|---|",
        f"| C2 Feature Pruning | {'YES' if c2_status != 'reject' else 'NO'} "
        f"| {'YES' if c2_status == 'full_upgrade' else 'NO'} | **{c2_status.upper()}** |",
        f"| C3 Decision Tuning | {'YES' if c3_status != 'reject' else 'NO'} "
        f"| {'YES' if c3_status == 'full_upgrade' else 'NO'} | **{c3_status.upper()}** |",
        "",
        f"> **Final chosen candidate: {candidate_label}**  ",
        f"> **Decision: {decision_headline}**",
        "",
        (
            "Neither challenger improved the routing profile meaningfully over the baseline. "
            "The baseline's low zone reject precision and human workload ratio remain the reference standard. "
            f"The baseline `{baseline_name}` is retained as the production model."
            if decision_type == "RETAIN_BASELINE" else
            f"**{candidate_label}** has demonstrated {('both routing improvement and all classifier guardrails' if decision_type == 'FULL_UPGRADE' else 'routing improvement (classifier guardrails show minor regression)')}. "
            "Please review the detailed comparison report in `model_bank/experiments/` before activating."
        ),
        "",
        "---",
        "",
        # ============================================================
        "## 5. Final Holdout Evidence",
        "",
        "> **All metrics in this section are computed on the Final Blind Holdout (Phase 4 — last 2 months).**  ",
        evidence_note,
        "",
        "### 5a. Confusion Matrix",
        "",
        f"![Confusion Matrix — {candidate_label}]({cm_img})",
        "",
        "| | Predicted Approve | Predicted Reject |",
        "|---|---|---|",
        f"| **Actual Approve** | TP = {cm_values.get('true_positive', '?'):,} "
        f"| FN = {cm_values.get('false_negative', '?'):,} |",
        f"| **Actual Reject** | FP = {cm_values.get('false_positive', '?'):,} "
        f"| TN = {cm_values.get('true_negative', '?'):,} |",
        "",
        "**What this means:**",
        f"- **True Positives ({cm_values.get('true_positive', 0):,})**: "
        "Correctly approved loans — revenue-generating decisions.",
        f"- **True Negatives ({cm_values.get('true_negative', 0):,})**: "
        "Correctly flagged rejections — credit risk avoided.",
        f"- **False Positives ({cm_values.get('false_positive', 0):,})**: "
        "Approved loans that actually defaulted — credit risk exposure.",
        f"- **False Negatives ({cm_values.get('false_negative', 0):,})**: "
        "Rejected loans that would have repaid — missed business opportunity.",
        "",
        "### 5b. Zone Distribution",
        "",
        f"![Zone Distribution — {candidate_label}]({zone_img})",
        "",
        "| Zone | Count | Population % | Actual Approve Rate |",
        "|---|---|---|---|",
        f"| High Zone (Auto-Approve) | {high_n:,} | {_pct(high_n / total)} | {_pct(hi_approve)} |",
        f"| Manual Zone (Human Review) | {manual_n:,} | {_pct(manual_n / total)} | {_pct(man_approve)} |",
        f"| Low Zone (Auto-Reject) | {low_n:,} | {_pct(low_n / total)} | {_pct(low_approve)} |",
        "",
        "### 5c. Zone Outcome Heatmap",
        "",
        f"![Zone Outcome Heatmap — {candidate_label}]({heat_img})",
        "",
        "The heatmap shows actual approve and reject rates within each routing zone. "
        "A well-calibrated model shows high approve rates in the High Zone and high reject rates in the Low Zone.",
        "",
        "---",
        "",
        # ============================================================
        "## 6. Model Explainability",
        "",
        f"> SHAP values below reflect the **{candidate_label}** model.",
        "> SHAP (SHapley Additive exPlanations) shows how much each feature pushed a",
        "> specific prediction higher or lower.",
        "",
        "### 6a. SHAP Summary Plot (Global Feature Impact)",
        "",
        f"![SHAP Summary — {candidate_label}]({shap_img})",
        "",
        "Each dot is one holdout sample. Features are ranked by total impact. "
        "Red = high feature value, Blue = low feature value. "
        "Dots to the right pushed the score toward approval.",
        "",
        "### 6b. Top Feature Importance",
        "",
        f"![Feature Importance — {candidate_label}]({bar_img})",
        "",
        "The bar chart shows the mean |SHAP value| per feature — "
        "which features the model relies on most when making decisions.",
        "",
        "---",
        "",
        # ============================================================
        "## 7. Decision Rationale",
        "",
    ]

    if decision_type == "RETAIN_BASELINE":
        lines += [
            f"### Why `{baseline_name}` is Retained",
            "",
            f"The active baseline `{baseline_name}` is retained as the production model because:",
            "",
            "1. **C2 Feature Pruning** did not pass promotion criteria.",
            f"   - {c2_reason or 'No meaningful routing improvement detected.'}",
            "",
            "2. **C3 Decision Tuning** did not pass promotion criteria.",
            f"   - {c3_reason or 'No meaningful routing improvement detected.'}",
            "",
            "3. The routing-first framework requires a challenger to demonstrably improve:",
            "   - Low zone reject precision (catching more true rejects in the auto-reject zone), and/or",
            "   - Human review workload ratio (reducing the manual review burden),",
            "   before promotion is considered — regardless of AUC change.",
            "",
            "4. No automatic baseline replacement has occurred.",
            "   To manually promote a challenger: `python main.py --create-baseline`",
            "",
            "### If You Want to Re-Evaluate",
            "",
            "- Re-run challengers: `python main.py --run-c2` or `--run-c3`",
            "- Regenerate comparison reports: `python main.py --compare-baseline --challenger c2`",
            "- Full lifecycle rerun: `python main.py`",
        ]
    elif decision_type == "FULL_UPGRADE":
        lines += [
            f"### Why {candidate_label} is the Recommended Full Upgrade",
            "",
            f"**{candidate_label}** qualifies for a full baseline replacement because:",
            "",
            f"- **Decision**: {decision_reason}",
            "",
            "- Both routing quality AND classifier guardrails are satisfied:",
            "  - Low zone reject precision improved beyond threshold (+0.005)",
            "  - Human review workload is within tolerance (±2 pp)",
            "  - AUC regression < 0.01 and F1_reject regression < 0.02",
            "",
            "### How to Promote",
            "",
            "No automatic promotion has occurred. To activate the challenger as the new baseline:",
            "",
            "```",
            "python main.py --create-baseline --baseline-name baseline_v2",
            "```",
            "",
            "After promotion, all future challenger evaluations will compare against the new baseline.",
        ]
    else:  # ROUTING_ONLY_UPGRADE
        lines += [
            f"### Why {candidate_label} is a Routing-Only Upgrade Candidate",
            "",
            f"**{candidate_label}** shows routing improvement but classifier metrics regressed:",
            "",
            f"- **Decision**: {decision_reason}",
            "",
            "- This means:",
            "  - Routing quality (low zone reject precision / human workload) improved",
            "  - However, AUC or F1_reject regressed beyond the allowed tolerance",
            "  - A full model replacement is **not** recommended at this stage",
            "  - The routing policy parameters may be updated independently",
            "",
            "### Recommended Action",
            "",
            "Consider reviewing whether the routing thresholds from this challenger can be",
            "applied to the existing baseline model without replacing the full model.",
            "If further retraining resolves the classifier regression, re-run evaluation.",
        ]

    lines += [
        "",
        "---",
        "",
        f"*This report was auto-generated by `utils/final_decision_report.py` on {now}.*  ",
        "*No baseline has been replaced. All promotion decisions require an explicit operator action.*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Governance report saved: %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────
# JSON Summary Writer
# ─────────────────────────────────────────────────────────────

def _write_decision_summary_json(
    decision: "FinalDecision",
    report_dir: Path,
    plots_dir: Path,
    report_path: Path,
    cm_values: dict,
    shap_summary_path: Optional[Path],
    shap_bar_path: Optional[Path],
) -> Path:
    """
    Write final_baseline_decision_summary.json — machine-readable single source of truth.

    Keys:
      generated_at, active_baseline, chosen_candidate_id, chosen_candidate_label,
      decision_type, decision_reason, c2_result, c3_result,
      final_holdout_evidence, artifact_paths
    """
    def _spath(p: Optional[Path]) -> str:
        return str(p) if p is not None else ""

    summary = {
        "generated_at":           datetime.now().isoformat(),
        "active_baseline":        decision.active_baseline_name,
        "chosen_candidate_id":    decision.chosen_candidate_id,
        "chosen_candidate_label": decision.chosen_candidate_label,
        "decision_type":          decision.decision_type,
        "decision_reason":        decision.decision_reason,
        "c2_result": {
            "upgrade_candidate": decision.eval_summary.get("c2_feature_pruning", {}).get("upgrade_candidate", "unknown"),
            "upgrade_reason":    decision.eval_summary.get("c2_feature_pruning", {}).get("upgrade_reason", ""),
        },
        "c3_result": {
            "upgrade_candidate": decision.eval_summary.get("c3_decision_tuning", {}).get("upgrade_candidate", "unknown"),
            "upgrade_reason":    decision.eval_summary.get("c3_decision_tuning", {}).get("upgrade_reason", ""),
        },
        "overall_recommendation": decision.eval_summary.get("overall_recommendation", ""),
        "final_holdout_evidence": {
            "confusion_matrix":   cm_values,
            "zone_distribution": {
                "high_zone_count":   int((decision.holdout_csv["pred_zone"] == 2).sum()),
                "manual_zone_count": int((decision.holdout_csv["pred_zone"] == 1).sum()),
                "low_zone_count":    int((decision.holdout_csv["pred_zone"] == 0).sum()),
                "total":             len(decision.holdout_csv),
            },
        },
        "artifact_paths": {
            "report_md":              _spath(report_path),
            "confusion_matrix_png":   _spath(plots_dir / "confusion_matrix.png"),
            "zone_distribution_png":  _spath(plots_dir / "zone_distribution.png"),
            "zone_heatmap_png":       _spath(plots_dir / "zone_outcome_heatmap.png"),
            "shap_summary_png":       _spath(shap_summary_path),
            "shap_bar_png":           _spath(shap_bar_path),
            "challenger_eval_json":   _spath(
                report_dir.parent / "challenger_evaluation_summary.json"
            ),
        },
    }

    json_path = report_dir / "final_baseline_decision_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Decision summary JSON saved: %s", json_path)
    return json_path


# ─────────────────────────────────────────────────────────────
# build_final_decision_artifacts  — clean orchestration layer
# ─────────────────────────────────────────────────────────────

def build_final_decision_artifacts(
    decision: "FinalDecision",
    project_root: Path,
) -> dict:
    """
    Generate all final decision artefacts from a resolved FinalDecision.

    This is the single orchestration point for:
      - All 5 plots (confusion matrix, zone distribution, heatmap, SHAP x2)
      - model_governance_decision_report.md
      - final_baseline_decision_summary.json

    All artefacts are written under:
      model_bank/experiments/final_decision_report/

    The evidence sections (plots, tables, narratives) all reflect the
    *chosen candidate* stored in ``decision`` — not always the baseline.

    Returns a dict with keys:
      report_path, json_summary_path, plots_dir, cm_values, decision_type
    """
    # ── Resolve output directories ────────────────────────────────────
    report_dir = project_root / "model_bank" / "experiments" / "final_decision_report"
    plots_dir  = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    holdout_df      = decision.holdout_csv
    candidate_label = decision.chosen_candidate_label
    baseline_meta   = decision.baseline_meta

    # ── Plot 1: Confusion Matrix ──────────────────────────────────────
    print(f"\n  [1/5] Confusion matrix ({candidate_label}) ...")
    cm_values = _plot_confusion_matrix(
        holdout_df=holdout_df,
        lower_threshold=decision.lower_threshold,
        upper_threshold=decision.upper_threshold,
        candidate_label=candidate_label,
        out_path=plots_dir / "confusion_matrix.png",
    )

    # ── Plot 2: Zone Distribution ─────────────────────────────────────
    print(f"  [2/5] Zone distribution ({candidate_label}) ...")
    _plot_zone_distribution(
        holdout_df=holdout_df,
        baseline_meta=baseline_meta,
        candidate_label=candidate_label,
        out_path=plots_dir / "zone_distribution.png",
    )

    # ── Plot 3: Zone Outcome Heatmap ──────────────────────────────────
    print(f"  [3/5] Zone outcome heatmap ({candidate_label}) ...")
    _plot_zone_outcome_heatmap(
        holdout_df=holdout_df,
        candidate_label=candidate_label,
        out_path=plots_dir / "zone_outcome_heatmap.png",
    )

    # ── Plots 4 & 5: SHAP ────────────────────────────────────────────
    print(f"  [4/5] SHAP explainability ({candidate_label}) ...")
    shap_summary_path: Optional[Path] = None
    shap_bar_path:     Optional[Path] = None

    model        = decision.model_pkl
    feature_names = decision.feature_names

    if model is not None and feature_names is not None:
        oot_df = _load_gold_oot(project_root)
        if oot_df is not None:
            available_features = [f for f in feature_names if f in oot_df.columns]
            if available_features:
                if "案件編號" in oot_df.columns and "案件編號" in holdout_df.columns:
                    oot_sub = oot_df[oot_df["案件編號"].isin(holdout_df["案件編號"])]
                else:
                    oot_sub = oot_df

                X_all = oot_sub[available_features].dropna()
                sample_size = min(2000, len(X_all))
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_all), size=sample_size, replace=False)
                X_sample = X_all.iloc[idx].reset_index(drop=True)

                print(f"       SHAP sample: {len(X_sample):,} rows × {len(available_features)} features")
                shap_summary_path, shap_bar_path = _plot_shap(
                    model=model,
                    X_sample=X_sample,
                    feature_names=available_features,
                    candidate_label=candidate_label,
                    out_dir=plots_dir,
                )
            else:
                print("       WARNING: Feature columns not found in OOT data; SHAP skipped.")
        else:
            print("       WARNING: Gold OOT data not found; SHAP skipped.")
    else:
        print("       WARNING: Model or feature names not loaded; SHAP skipped.")

    # ── Write Markdown report ─────────────────────────────────────────
    print("  [5/5] Writing governance report ...")
    report_path = _write_report(
        decision=decision,
        report_dir=report_dir,
        plots_dir=plots_dir,
        cm_values=cm_values,
        shap_summary_path=shap_summary_path,
        shap_bar_path=shap_bar_path,
    )

    # ── Write JSON summary ────────────────────────────────────────────
    json_summary_path = _write_decision_summary_json(
        decision=decision,
        report_dir=report_dir,
        plots_dir=plots_dir,
        report_path=report_path,
        cm_values=cm_values,
        shap_summary_path=shap_summary_path,
        shap_bar_path=shap_bar_path,
    )

    print(f"\n  Report          : {report_path}")
    print(f"  Decision JSON   : {json_summary_path}")
    print(f"  Plots           : {plots_dir}")
    print(f"  Decision        : {decision.decision_type} — {decision.chosen_candidate_label}")

    return {
        "report_path":       str(report_path),
        "json_summary_path": str(json_summary_path),
        "plots_dir":         str(plots_dir),
        "cm_values":         cm_values,
        "decision_type":     decision.decision_type,
        "chosen_candidate":  decision.chosen_candidate_label,
        "decision_reason":   decision.decision_reason,
    }


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def run_final_decision_report(project_root: Optional[Path] = None) -> dict:
    """
    Generate the complete final decision report with plots.

    This is the main public entry point, intended to be called from main.py.

    Steps:
      1. Call finalize_model_decision() to determine the final chosen candidate
         and load all required artefacts.
      2. Call build_final_decision_artifacts() to generate all plots,
         model_governance_decision_report.md, and
         final_baseline_decision_summary.json.

    Reads:
      - model_bank/baselines/<active>/baseline_metadata.json
      - model_bank/baselines/<active>/predictions/final_holdout_predictions.csv
      - model_bank/baselines/<active>/model/final_champion_model.pkl
      - model_bank/baselines/<active>/artifacts/feature_names.json
      - model_bank/experiments/challenger_evaluation_summary.json
      - model_bank/experiments/c2_feature_pruning/c2_challenger_metadata.json
      - model_bank/experiments/c3_decision_tuning/challenger_summary.json
      - datamart/gold/oot/**/*.parquet  (for SHAP feature matrix)

    Writes to model_bank/experiments/final_decision_report/:
      plots/confusion_matrix.png
      plots/zone_distribution.png
      plots/zone_outcome_heatmap.png
      plots/shap_summary.png
      plots/shap_feature_importance.png
      model_governance_decision_report.md
      final_baseline_decision_summary.json

    Returns a dict with keys:
      report_path, json_summary_path, plots_dir, cm_values,
      decision_type, chosen_candidate, decision_reason
    """
    if project_root is None:
        from pathlib import Path as _P
        project_root = _P(__file__).parent.parent

    print("\n" + "=" * 70)
    print("  Final Decision Report")
    print("=" * 70)

    # ── Step 1: Resolve final decision ───────────────────────────────
    decision = finalize_model_decision(project_root)
    if decision is None:
        print("  ERROR: Could not resolve final decision. Run training first.")
        return {}

    if decision.holdout_csv is None:
        print("  ERROR: final_holdout_predictions.csv not found for chosen candidate.")
        return {}

    print(f"  Active baseline      : {decision.active_baseline_name}")
    print(f"  Chosen candidate     : {decision.chosen_candidate_label}")
    print(f"  Decision type        : {decision.decision_type}")
    print(f"  Holdout rows         : {len(decision.holdout_csv):,}")
    if not decision.eval_summary:
        print("  WARNING: challenger_evaluation_summary.json not found.")
        print("  Run challenger evaluation first: python main.py --challengers-only")

    # ── Step 2: Build all artefacts ──────────────────────────────────
    result = build_final_decision_artifacts(decision=decision, project_root=project_root)

    print("\n" + "=" * 70)
    print("  Final Decision Report complete.")
    print("=" * 70)

    return result
