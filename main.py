"""
ML Pipeline main entry point
============================
Uses the Four Phase Trainer architecture for credit risk model training.

Four-phase flow:
1. Phase 1: Model Development (18-month rolling training)
2. Phase 2: Champion Retraining (full development data -> Final Champion Artifact)
3. Phase 3: Policy Validation (4 months, threshold / zone policy tuning)
4. Phase 4: Final Blind Holdout (last 2 months, model and threshold frozen)

Challenger experiments:
- C2 Feature Pruning: remove 6 unstable / low-importance features (25 -> 19)
- C3 Decision Tuning: decision-oriented + anti-overfitting hyperparameter search

Baseline lifecycle:
- After the first successful training run, baseline_v1 is auto-created and activated.
- Use --create-baseline to manually create a named baseline from the latest run.
- Challengers compare against the active baseline (dynamic loading, not hardcoded).

Standard full lifecycle (python main.py):
  A. Data Pipeline       Bronze -> Silver -> Gold
  B. Baseline Pipeline   Four-phase training + auto-create/activate baseline
  C. Challenger Stage    Run C2 -> Run C3 -> compare each against active baseline
  D. Decision Summary    challenger_evaluation_summary.{json,md} in model_bank/experiments/
  E. (optional)          Production Monitoring  [--all only]

Usage:
    python main.py                             # Data + Training + Challenger Evaluation
    python main.py --all                       # Data + Training + Challengers + Monitoring
    python main.py --data-only                 # Bronze -> Silver -> Gold only
    python main.py --train-only                # Four-phase training only
    python main.py --challengers-only          # Challenger evaluation only (requires baseline)
    python main.py --monitor-only              # Production monitoring only
    python main.py --run-c2                    # C2 Feature Pruning challenger (manual)
    python main.py --run-c3                    # C3 Decision Tuning challenger (manual)
    python main.py --compare-baseline --challenger c2   # Comparison report only
    python main.py --compare-baseline --challenger c3   # Comparison report only
    python main.py --create-baseline                    # Create baseline from latest run
    python main.py --create-baseline --baseline-name v2 # Name the new baseline
    python main.py --lower-threshold 0.3       # Custom threshold (fallback)
    python main.py --imbalance scale_weight    # Imbalance handling method
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# 添加專案根目錄到 Python 路徑
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_processing_bronze_table import run_bronze_pipeline
from utils.data_processing_silver_table import run_silver_pipeline
from utils.data_processing_gold_table import run_gold_pipeline
from utils.four_phase_trainer import (
    FourPhaseTrainer,
    run_four_phase_pipeline,
    PhaseConfig,
)
from utils.production_monitor import (
    ProductionMonitor,
    ProductionMonitoringConfig,
    generate_retraining_data_window,
)
from utils.challenger_manager import (
    run_c2_feature_pruning_challenger,
    run_c3_decision_tuning_challenger,
    compare_against_baseline,
)

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================
# Baseline Helpers
# ============================================

def _maybe_create_first_baseline(project_root: Path, training_results: dict):
    """
    Auto-create baseline_v1 after the first successful training run.

    - If no baselines exist at all: create baseline_v1 and activate it.
    - If baselines exist but active_baseline.json is missing: activate the most
      recent baseline so the pipeline is not left in an ambiguous state.
    - If an active baseline is already set: no-op.
    """
    from utils.baseline_manager import BaselineManager

    mgr = BaselineManager(str(project_root / "model_bank"))
    existing = mgr.list_baselines()

    if existing:
        # Baselines exist — check whether an active one is set
        active = mgr.get_active_baseline()
        if active is None:
            # Orphaned state: folders exist but no active pointer — recover silently
            most_recent = existing[-1]  # list_baselines() returns sorted names
            try:
                mgr.activate_baseline(most_recent)
                print(f"\n  [Baseline] No active baseline was set; activated '{most_recent}'.")
            except Exception as exc:
                logger.warning("Could not recover active baseline: %s", exc)
        return  # either already active, or just recovered above

    # No baselines exist at all — auto-create baseline_v1
    run_dir = (training_results or {}).get("output_dir", "")
    if not run_dir:
        logger.warning("Cannot auto-create baseline: training results do not include output_dir.")
        return

    run_id = Path(run_dir).name
    try:
        mgr.create_baseline(
            run_id=run_id,
            baseline_name="baseline_v1",
            description="Auto-created from first successful training run",
            auto_activate=True,
        )
        print(f"\n  [Baseline] Auto-created baseline_v1 from run: {run_id}")
        print("  [Baseline] baseline_v1 is now the active baseline.")
        print("  [Baseline] Challengers will compare against baseline_v1 by default.")
    except Exception as exc:
        logger.warning("Could not auto-create baseline_v1: %s", exc)


def _create_named_baseline(project_root: Path, baseline_name: str = None):
    """
    Create a named baseline from the latest training run.

    If baseline_name is None, auto-increments: baseline_v1, baseline_v2, ...
    """
    import json
    from utils.baseline_manager import BaselineManager

    model_bank = project_root / "model_bank"
    run_dirs = sorted(
        [d for d in model_bank.iterdir()
         if d.is_dir() and (d.name.startswith("four_phase_") or d.name.startswith("rolling_training_v2_"))],
        reverse=True,
    )
    if not run_dirs:
        print("  ERROR: No training result directories found. Run training first.")
        return None

    latest = run_dirs[0]
    run_id = latest.name

    mgr = BaselineManager(str(model_bank))
    if baseline_name is None:
        existing = mgr.list_baselines()
        n = len(existing) + 1
        baseline_name = f"baseline_v{n}"

    try:
        mgr.create_baseline(
            run_id=run_id,
            baseline_name=baseline_name,
            description=f"Created from run {run_id}",
            auto_activate=True,
        )
        print(f"\n  [Baseline] Created and activated '{baseline_name}' from run: {run_id}")
        return baseline_name
    except Exception as exc:
        print(f"  ERROR: Could not create baseline '{baseline_name}': {exc}")
        return None


# ============================================
# Data Pipeline
# ============================================
def run_data_pipeline(project_root: Path = None):
    """
    Run full data processing pipeline: Bronze -> Silver -> Gold.
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    print("\n" + "=" * 70)
    print("  Data Processing Pipeline (Bronze -> Silver -> Gold)")
    print("=" * 70)

    print("\n[1/3] Bronze layer ...")
    bronze_path = run_bronze_pipeline(project_root)
    print(f"  Bronze complete: {bronze_path}")

    print("\n[2/3] Silver layer ...")
    silver_path = run_silver_pipeline(project_root)
    print(f"  Silver complete: {silver_path}")

    print("\n[3/3] Gold layer ...")
    gold_output_paths = run_gold_pipeline(project_root)
    print("  Gold complete")
    for name, path in gold_output_paths.items():
        if name not in ("artifact_files",):
            print(f"    {name}: {path}")

    if "manifest" in gold_output_paths:
        print(f"\n  Gold Split Manifest: {gold_output_paths['manifest']}")

    print("\n" + "=" * 70)
    print("  Data processing complete.")
    print("=" * 70)

    return gold_output_paths


# ============================================
# Training Pipeline
# ============================================
def run_training_pipeline(
    project_root: Path = None,
    lower_threshold: float = None,
    upper_threshold: float = None,
    imbalance_strategy: str = "scale_weight",
    use_calibration: bool = True,
    model_names: list = None,
):
    """
    Run the four-phase training pipeline.

    lower/upper_threshold are fallback values; Phase 3 recommended thresholds take priority.
    If None, defaults of 0.5 / 0.85 are used.
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    lower_threshold = lower_threshold or 0.5
    upper_threshold = upper_threshold or 0.85

    print("\n" + "=" * 70)
    print("  Four Phase Training Pipeline")
    print("=" * 70)
    print(f"  Imbalance Strategy : {imbalance_strategy}")
    print(f"  Lower Threshold    : {lower_threshold} (fallback; Phase 3 recommendation takes priority)")
    print(f"  Upper Threshold    : {upper_threshold} (fallback; Phase 3 recommendation takes priority)")
    print(f"  Calibration        : {use_calibration}")
    print("=" * 70)

    results = run_four_phase_pipeline(
        project_root=project_root,
        model_names=model_names,
        use_calibration=use_calibration,
        imbalance_strategy=imbalance_strategy,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )

    # Auto-create baseline_v1 on first training run
    _maybe_create_first_baseline(project_root, results or {})

    print("\n" + "=" * 70)
    print("  Four-phase training complete.")
    print("=" * 70)

    return results


# ============================================
# Monitoring Pipeline
# ============================================
def run_monitoring_pipeline(
    project_root: Path = None,
    lower_threshold: float = None,
    upper_threshold: float = None,
):
    """
    Run production monitoring.

    Reference source resolution order:
      1. Active baseline (model_bank/baselines/active_baseline.json)
         - Uses active baseline's policy thresholds and policy_validation_predictions.csv
         - This is the correct path once a baseline has been established
      2. Latest training run directory (fallback when no baseline exists yet)

    Threshold override resolution (applied on top of reference source):
      1. CLI --lower-threshold / --upper-threshold (if provided)
      2. Active baseline thresholds  (from active_baseline.json)
      3. zone_policy_summary.json from run dir  (fallback)
      4. champion_summary.json threshold_config  (fallback)
      5. ProductionMonitoringConfig defaults

    Monitoring target (demo mode): final_holdout_predictions.csv from the reference run.
    In production, replace with actual production batch scoring data.
    """
    import json
    import numpy as np
    import pandas as pd
    from utils.baseline_manager import BaselineManager

    if project_root is None:
        project_root = PROJECT_ROOT

    print("\n" + "=" * 70)
    print("  Production Monitoring")
    print("=" * 70)

    model_bank = project_root / "model_bank"
    if not model_bank.exists():
        print("  WARNING: model_bank not found; run training pipeline first.")
        return None

    # ── Step 1: Resolve reference run directory ───────────────────────
    # Prefer active baseline; fall back to latest training run.
    reference_dir = None
    reference_source = None
    active_thresholds = {}

    mgr = BaselineManager(str(model_bank))
    active_info = mgr.get_active_baseline()

    if active_info:
        baseline_name = active_info.get("active_baseline", "")
        baseline_path = model_bank / "baselines" / baseline_name
        if baseline_path.exists():
            # Resolve the original run folder for holdout predictions
            try:
                meta = mgr.get_baseline_metadata(baseline_name)
                run_id = meta.get("source_run", {}).get("run_folder", "")
                run_folder = model_bank / run_id if run_id else None
                if run_folder and run_folder.exists():
                    reference_dir = run_folder
                    reference_source = f"active baseline '{baseline_name}' -> run {run_id}"
                else:
                    # Fall back to the copied predictions inside the baseline dir
                    reference_dir = baseline_path
                    reference_source = f"active baseline '{baseline_name}' (copied files)"
            except Exception as exc:
                logger.warning("Could not resolve baseline run folder: %s", exc)

            active_thresholds = {
                "lower": active_info.get("thresholds", {}).get("lower"),
                "upper": active_info.get("thresholds", {}).get("upper"),
            }
            print(f"  Reference : {reference_source}")
        else:
            print(f"  WARNING: Active baseline path not found: {baseline_path}")

    if reference_dir is None:
        run_dirs = sorted(
            [d for d in model_bank.iterdir()
             if d.is_dir() and (d.name.startswith("four_phase_") or d.name.startswith("rolling_training_v2_"))],
            reverse=True,
        )
        if not run_dirs:
            print("  WARNING: No training result directories found.")
            return None
        reference_dir = run_dirs[0]
        reference_source = f"latest run '{reference_dir.name}' (no active baseline)"
        print(f"  Reference : {reference_source}")

    # ── Step 2: Resolve thresholds ────────────────────────────────────
    resolved_lower = lower_threshold   # CLI takes top priority
    resolved_upper = upper_threshold
    threshold_source = "CLI argument" if lower_threshold is not None else None

    if resolved_lower is None and active_thresholds.get("lower") is not None:
        resolved_lower   = active_thresholds["lower"]
        resolved_upper   = active_thresholds["upper"]
        threshold_source = f"active baseline '{active_info.get('active_baseline', '')}' thresholds"

    if resolved_lower is None:
        zone_policy_path = reference_dir / "zone_policy_summary.json"
        champion_path    = reference_dir / "champion_summary.json"
        if zone_policy_path.exists():
            zone_policy = json.loads(zone_policy_path.read_text(encoding="utf-8"))
            if "selected_lower_threshold" in zone_policy:
                resolved_lower   = zone_policy["selected_lower_threshold"]
                resolved_upper   = zone_policy["selected_upper_threshold"]
                threshold_source = "zone_policy_summary.json (Phase 3 recommended)"
            elif "recommended_lower_threshold" in zone_policy:
                resolved_lower   = zone_policy["recommended_lower_threshold"]
                resolved_upper   = zone_policy["recommended_upper_threshold"]
                threshold_source = "zone_policy_summary.json (recommended)"
        elif champion_path.exists():
            champion_info = json.loads(champion_path.read_text(encoding="utf-8"))
            tc = champion_info.get("threshold_config", {})
            if tc.get("lower_threshold") is not None:
                resolved_lower   = tc["lower_threshold"]
                resolved_upper   = tc["upper_threshold"]
                threshold_source = "champion_summary.json"

    if resolved_lower is None:
        resolved_lower = ProductionMonitoringConfig().default_lower_threshold
        resolved_upper = ProductionMonitoringConfig().default_upper_threshold
        threshold_source = "ProductionMonitoringConfig defaults"

    print(f"\n  Monitoring threshold:")
    print(f"    lower : {resolved_lower}")
    print(f"    upper : {resolved_upper}")
    print(f"    source: {threshold_source}")

    # ── Step 3: Locate baseline predictions file ──────────────────────
    # Prefer policy/policy_validation_predictions.csv inside baseline dir,
    # then root-level policy_validation_predictions.csv in the run folder.
    def _find_pv_csv(d: Path) -> Path:
        candidates = [
            d / "policy" / "policy_validation_predictions.csv",
            d / "policy_validation_predictions.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    pv_pred_path = _find_pv_csv(reference_dir)
    if pv_pred_path is None:
        print("  WARNING: policy_validation_predictions.csv not found in reference dir.")
        return None

    print(f"\n  Baseline predictions: {pv_pred_path}")
    pv_df = pd.read_csv(pv_pred_path)
    baseline_scores = pv_df["pred_prob"].values
    print(f"    Rows: {len(baseline_scores)}  Mean score: {baseline_scores.mean():.4f}")

    # ── Step 4: Set up monitor ────────────────────────────────────────
    config  = ProductionMonitoringConfig(
        default_lower_threshold=resolved_lower,
        default_upper_threshold=resolved_upper,
    )
    monitor = ProductionMonitor(config=config)
    monitor.set_baseline(baseline_scores, resolved_lower, resolved_upper)

    # Set last training date from champion metadata
    champion_path = reference_dir / "champion_summary.json"
    if not champion_path.exists():
        champion_path = reference_dir / "metadata" / "champion_summary.json"
    if champion_path.exists():
        champion_info = json.loads(champion_path.read_text(encoding="utf-8"))
        monitor.set_last_training_date(champion_info.get("created_at", "")[:10])

    # ── Step 5: Run monitoring against holdout data ───────────────────
    def _find_holdout_csv(d: Path) -> Path:
        candidates = [
            d / "predictions" / "final_holdout_predictions.csv",
            d / "final_holdout_predictions.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    holdout_pred_path = _find_holdout_csv(reference_dir)
    if holdout_pred_path is None:
        print("  WARNING: final_holdout_predictions.csv not found; monitoring skipped.")
        return None

    print(f"\n  Monitoring target (demo): {holdout_pred_path}")
    print("  NOTE: Replace with production batch scoring data in live deployment.")

    holdout_df  = pd.read_csv(holdout_pred_path)
    predictions = holdout_df["pred_prob"].values
    labels      = holdout_df["actual_label"].values if "actual_label" in holdout_df.columns else None

    result = monitor.run_production_monitoring(
        predictions=predictions,
        labels=labels,
        model_version=reference_dir.name,
        period_start=str(holdout_df["進件日"].min()) if "進件日" in holdout_df.columns else "",
        period_end=str(holdout_df["進件日"].max()) if "進件日" in holdout_df.columns else "",
        lower_threshold=resolved_lower,
        upper_threshold=resolved_upper,
    )

    output_dir = project_root / "model_bank" / "monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"monitoring_result_{ts}.json"

    result_dict = result.to_dict()
    result_dict["threshold_config"] = {
        "lower_threshold": resolved_lower,
        "upper_threshold": resolved_upper,
        "source": threshold_source,
    }
    result_dict["reference_source"] = reference_source
    if active_info:
        result_dict["active_baseline"] = active_info.get("active_baseline")

    result_path.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Monitoring result saved: {result_path}")

    decision = {
        "timestamp": ts,
        "needs_retraining": result.needs_retraining,
        "trigger_reason": result.retraining_trigger_reason,
        "alerts": result.alerts,
        "warnings": result.warnings,
    }
    if result.needs_retraining:
        window = generate_retraining_data_window(
            current_date=datetime.now().date()
        )
        decision["retraining_window"] = window

    decision_path = output_dir / f"retraining_decision_{ts}.json"
    decision_path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Retraining decision saved: {decision_path}")

    return result


# ============================================
# Challenger Evaluation Pipeline
# ============================================

def run_challenger_evaluation_pipeline(project_root: Path = None) -> dict:
    """
    Run the full challenger evaluation stage (C2 + C3) against the active baseline.

    Steps:
      1. Verify an active baseline exists — abort with a clear message if not.
      2. Run C2 Feature Pruning challenger.
      3. Run C3 Decision Tuning challenger.
      4. Build a structured evaluation summary.
      5. Write challenger_evaluation_summary.json and .md to
         model_bank/experiments/.

    Returns a dict with keys:
      active_baseline, c2, c3, overall_recommendation, summary_path
    """
    import json
    from utils.challenger_manager import load_active_baseline_metrics

    if project_root is None:
        project_root = PROJECT_ROOT

    print("\n" + "=" * 70)
    print("  Challenger Evaluation Stage")
    print("=" * 70)

    # ── Guard: need an active baseline ───────────────────────────────
    baseline_metrics = load_active_baseline_metrics(project_root)
    bl_label = (
        baseline_metrics.get("_source", "")
        .replace("active_baseline:", "")
        .strip()
        or "Active Baseline"
    )
    if bl_label == "Active Baseline":
        print("  WARNING: No active baseline found.")
        print("  Run training first so that baseline_v1 is created, then re-run.")
        return {}

    print(f"\n  Active baseline : {bl_label}")
    print(f"  Baseline AUC    : {baseline_metrics.get('auc', 0):.4f}")
    print(f"  Baseline F1_rej : {baseline_metrics.get('f1_reject', 0):.4f}")

    # ── C2 ────────────────────────────────────────────────────────────
    print("\n[C2] Feature Pruning challenger ...")
    c2_result = {}
    try:
        c2_result = run_c2_feature_pruning_challenger(project_root=project_root) or {}
        c2_status = c2_result.get("upgrade_candidate", "unknown")
        print(f"  C2 result : {c2_status}")
    except Exception as exc:
        logger.error("C2 challenger failed: %s", exc)
        c2_status = "error"
        c2_result = {"upgrade_candidate": "error", "upgrade_reason": str(exc)}

    # ── C3 ────────────────────────────────────────────────────────────
    print("\n[C3] Decision Tuning challenger ...")
    c3_result = {}
    try:
        c3_result = run_c3_decision_tuning_challenger(project_root=project_root) or {}
        c3_status = c3_result.get("upgrade_candidate", "unknown")
        print(f"  C3 result : {c3_status}")
    except Exception as exc:
        logger.error("C3 challenger failed: %s", exc)
        c3_status = "error"
        c3_result = {"upgrade_candidate": "error", "upgrade_reason": str(exc)}

    # ── Overall recommendation ────────────────────────────────────────
    # Priority: full_upgrade > routing_only_upgrade > reject
    _rank = {"full_upgrade": 3, "routing_only_upgrade": 2, "reject": 1,
             "unknown": 0, "error": 0}

    best_challenger = None
    best_status = "reject"
    for cid, res in [("C2_feature_pruning", c2_result), ("C3_decision_tuning", c3_result)]:
        st = res.get("upgrade_candidate", "reject")
        if _rank.get(st, 0) > _rank.get(best_status, 0):
            best_status = st
            best_challenger = cid

    if best_status == "full_upgrade":
        overall = f"FULL UPGRADE CANDIDATE: {best_challenger} — recommend replacing baseline"
    elif best_status == "routing_only_upgrade":
        overall = f"ROUTING UPGRADE CANDIDATE: {best_challenger} — recommend routing policy update only"
    else:
        overall = "KEEP BASELINE — no challenger meets promotion criteria"

    # ── Build summary dict ────────────────────────────────────────────
    def _challenger_block(result: dict, cid: str) -> dict:
        best = result.get("best_candidate", {}) if cid == "C3_decision_tuning" else {}
        return {
            "upgrade_candidate": result.get("upgrade_candidate", "unknown"),
            "upgrade_reason":    result.get("upgrade_reason",
                                   (best or {}).get("upgrade_reason", "")),
            "output_dir":        result.get("output_dir", ""),
        }

    summary = {
        "generated_at":           datetime.now().isoformat(),
        "active_baseline":        bl_label,
        "baseline_auc":           baseline_metrics.get("auc", 0),
        "baseline_f1_reject":     baseline_metrics.get("f1_reject", 0),
        "c2_feature_pruning":     _challenger_block(c2_result, "C2_feature_pruning"),
        "c3_decision_tuning":     _challenger_block(c3_result, "C3_decision_tuning"),
        "best_challenger":        best_challenger,
        "overall_recommendation": overall,
    }

    # ── Write artifacts ───────────────────────────────────────────────
    experiments_dir = project_root / "model_bank" / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    json_path = experiments_dir / "challenger_evaluation_summary.json"
    md_path   = experiments_dir / "challenger_evaluation_summary.md"

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_evaluation_summary_md(summary, md_path)

    print(f"\n  Summary JSON : {json_path}")
    print(f"  Summary MD   : {md_path}")
    print(f"\n  Overall      : {overall}")
    print("\n" + "=" * 70)
    print("  Challenger Evaluation complete.")
    print("=" * 70)

    return {**summary, "summary_path": str(json_path)}


def _write_evaluation_summary_md(summary: dict, path: Path):
    """Write challenger_evaluation_summary.md."""

    def _status_icon(st: str) -> str:
        return {"full_upgrade": "FULL UPGRADE",
                "routing_only_upgrade": "ROUTING UPGRADE",
                "reject": "REJECT"}.get(st, st.upper())

    c2 = summary.get("c2_feature_pruning", {})
    c3 = summary.get("c3_decision_tuning", {})

    lines = [
        "# Challenger Evaluation Summary",
        "",
        f"> Generated: {summary.get('generated_at', '')[:19]}  ",
        f"> Active baseline: **{summary.get('active_baseline', '')}**  ",
        f"> Baseline AUC: {summary.get('baseline_auc', 0):.4f}  |  "
        f"Baseline F1_reject: {summary.get('baseline_f1_reject', 0):.4f}",
        "",
        "---",
        "",
        "## Challenger Results",
        "",
        "| Challenger | Decision | Reason |",
        "|---|---|---|",
        f"| C2 Feature Pruning | **{_status_icon(c2.get('upgrade_candidate', ''))}** "
        f"| {c2.get('upgrade_reason', '')} |",
        f"| C3 Decision Tuning | **{_status_icon(c3.get('upgrade_candidate', ''))}** "
        f"| {c3.get('upgrade_reason', '')} |",
        "",
        "---",
        "",
        "## Overall Recommendation",
        "",
        f"> **{summary.get('overall_recommendation', '')}**",
        "",
        "---",
        "",
        "## Output Locations",
        "",
        f"- C2: `{c2.get('output_dir', '')}`",
        f"- C3: `{c3.get('output_dir', '')}`",
        "",
        "---",
        "",
        "*Auto-generated by run_challenger_evaluation_pipeline(). "
        "No baseline has been replaced — promotion requires an explicit --create-baseline call.*",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================
# Full Pipeline
# ============================================
def run_full_pipeline(
    project_root: Path = None,
    lower_threshold: float = None,
    upper_threshold: float = None,
    imbalance_strategy: str = "scale_weight",
    use_calibration: bool = True,
    model_names: list = None,
    include_challengers: bool = True,
    include_monitoring: bool = False,
):
    """
    Run the full pipeline.

    Stage A  Data processing (Bronze -> Silver -> Gold)
    Stage B  Four-phase training + auto-create/activate baseline
    Stage C  Challenger evaluation: C2 + C3 vs active baseline  [include_challengers=True]
    Stage D  Decision summary artifact written to model_bank/experiments/
    Stage E  Production monitoring                               [include_monitoring=True]

    Args:
        lower_threshold:      Fallback threshold; Phase 3 recommended value takes priority.
        upper_threshold:      Fallback threshold; Phase 3 recommended value takes priority.
        include_challengers:  Run C2 + C3 challenger evaluation stage (default True).
        include_monitoring:   Run production monitoring after challengers (default False).
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    print("\n" + "=" * 80)
    print("  ML PIPELINE - Full Run")
    print("  Four Phase Training Architecture")
    stages = ["Data", "Training", "Challenger Evaluation"]
    if include_monitoring:
        stages.append("Monitoring")
    print(f"  Stages: {' -> '.join(stages)}")
    print("=" * 80)

    # Stage A: Data processing
    run_data_pipeline(project_root)

    # Stage B: Four-phase training + baseline
    results = run_training_pipeline(
        project_root=project_root,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        imbalance_strategy=imbalance_strategy,
        use_calibration=use_calibration,
        model_names=model_names,
    )

    # Stage C + D: Challenger evaluation
    challenger_summary = {}
    if include_challengers:
        challenger_summary = run_challenger_evaluation_pipeline(project_root=project_root)

    # Stage E: Production monitoring (optional)
    monitoring_result = None
    if include_monitoring:
        selected_lower = None
        selected_upper = None
        if results and isinstance(results, dict):
            selected_lower = results.get("selected_lower_threshold")
            selected_upper = results.get("selected_upper_threshold")

        monitoring_result = run_monitoring_pipeline(
            project_root=project_root,
            lower_threshold=selected_lower,
            upper_threshold=selected_upper,
        )

    print("\n" + "=" * 80)
    print("  Full pipeline complete.")
    print(f"  Stages completed: {' -> '.join(stages)}")
    print("=" * 80)

    return {
        "training": results,
        "challengers": challenger_summary,
        "monitoring": monitoring_result,
    }


# ============================================
# CLI Entry Point
# ============================================
def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ML Pipeline - Four Phase Training Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Data + Training + Challenger Evaluation (default)
  python main.py --all                        # Data + Training + Challengers + Monitoring
  python main.py --data-only                  # Data processing only
  python main.py --train-only                 # Training only (requires Gold data)
  python main.py --challengers-only           # Challenger evaluation only (requires baseline)
  python main.py --monitor-only               # Production monitoring only
  python main.py --run-c2                     # Run C2 Feature Pruning challenger (manual)
  python main.py --run-c3                     # Run C3 Decision Tuning challenger (manual)
  python main.py --compare-baseline --challenger c2   # Regenerate C2 comparison report
  python main.py --compare-baseline --challenger c3   # Regenerate C3 comparison report
  python main.py --create-baseline            # Create and activate baseline from latest run
  python main.py --lower-threshold 0.35       # Custom threshold
  python main.py --imbalance smote            # Use SMOTE resampling
        """
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--data-only",
        action="store_true",
        help="Run data processing pipeline only (Bronze -> Silver -> Gold)",
    )
    mode_group.add_argument(
        "--train-only",
        action="store_true",
        help="Run four-phase training only (requires Gold data)",
    )
    mode_group.add_argument(
        "--challengers-only",
        action="store_true",
        help="Run challenger evaluation only: C2 + C3 vs active baseline (requires baseline)",
    )
    mode_group.add_argument(
        "--monitor-only",
        action="store_true",
        help="Run production monitoring only (requires trained model)",
    )
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Run full pipeline: Data + Training + Challenger Evaluation + Monitoring",
    )
    mode_group.add_argument(
        "--run-c2",
        action="store_true",
        help="Run C2 Feature Pruning challenger manually",
    )
    mode_group.add_argument(
        "--run-c3",
        action="store_true",
        help="Run C3 Decision Tuning challenger manually",
    )
    mode_group.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Regenerate baseline comparison report for a challenger (requires --challenger)",
    )
    mode_group.add_argument(
        "--create-baseline",
        action="store_true",
        help="Create and activate a baseline from the latest training run",
    )

    # Challenger selection (used with --compare-baseline)
    parser.add_argument(
        "--challenger",
        type=str,
        choices=["c2", "c3"],
        help="Challenger ID for --compare-baseline",
    )

    # Baseline name (used with --create-baseline)
    parser.add_argument(
        "--baseline-name",
        type=str,
        default=None,
        dest="baseline_name",
        help="Baseline name for --create-baseline (default: auto-increment baseline_v1, v2, ...)",
    )

    # Thresholds (fallback; Phase 3 recommended values take priority)
    parser.add_argument(
        "--lower-threshold",
        type=float,
        default=None,
        help="Zone lower threshold (default: auto-detect from Phase 3 outputs, fallback 0.5)",
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=None,
        help="Zone upper threshold (default: auto-detect from Phase 3 outputs, fallback 0.85)",
    )

    # Imbalance handling
    parser.add_argument(
        "--imbalance",
        type=str,
        default="scale_weight",
        choices=["scale_weight", "class_weight", "smote", "undersample", "none"],
        help="Class imbalance strategy (default: scale_weight)",
    )

    # Calibration
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable probability calibration",
    )

    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Candidate model names (e.g., xgboost random_forest logistic_regression)",
    )

    args = parser.parse_args()

    if args.data_only:
        run_data_pipeline()

    elif args.train_only:
        run_training_pipeline(
            lower_threshold=args.lower_threshold,
            upper_threshold=args.upper_threshold,
            imbalance_strategy=args.imbalance,
            use_calibration=not args.no_calibration,
            model_names=args.models,
        )

    elif args.challengers_only:
        run_challenger_evaluation_pipeline(project_root=PROJECT_ROOT)

    elif args.monitor_only:
        run_monitoring_pipeline(
            lower_threshold=args.lower_threshold,
            upper_threshold=args.upper_threshold,
        )

    elif args.all:
        run_full_pipeline(
            lower_threshold=args.lower_threshold,
            upper_threshold=args.upper_threshold,
            imbalance_strategy=args.imbalance,
            use_calibration=not args.no_calibration,
            model_names=args.models,
            include_challengers=True,
            include_monitoring=True,
        )

    elif args.run_c2:
        result = run_c2_feature_pruning_challenger(project_root=PROJECT_ROOT)
        if result:
            print(f"\nC2 challenger complete.")
            print(f"  Output : {result.get('output_dir', 'N/A')}")
            print(f"  Upgrade: {result.get('upgrade_candidate', 'N/A')}")
            if result.get("upgrade_reason"):
                print(f"  Reason : {result['upgrade_reason']}")

    elif args.run_c3:
        result = run_c3_decision_tuning_challenger(project_root=PROJECT_ROOT)
        if result:
            print(f"\nC3 challenger complete.")
            print(f"  Output    : {result.get('output_dir', 'N/A')}")
            print(f"  Best model: {result.get('best_candidate', 'N/A')}")
            print(f"  Upgrade   : {result.get('upgrade_candidate', 'N/A')}")
            if result.get("upgrade_reason"):
                print(f"  Reason    : {result['upgrade_reason']}")

    elif args.compare_baseline:
        if not args.challenger:
            parser.error("--compare-baseline requires --challenger {c2,c3}")
        import json
        from utils.challenger_manager import load_active_baseline_metrics

        challenger_id = args.challenger.lower()
        active_baseline = load_active_baseline_metrics(PROJECT_ROOT)
        bl_label = (
            active_baseline.get("_source", "")
            .replace("active_baseline:", "")
            .strip()
            or "Active Baseline"
        )

        if challenger_id == "c2":
            exp_dir = PROJECT_ROOT / "model_bank" / "experiments" / "c2_feature_pruning"
            meta_path = exp_dir / "c2_challenger_metadata.json"
            if not meta_path.exists():
                print(f"ERROR: c2_challenger_metadata.json not found at {exp_dir}")
                print("       Run --run-c2 first.")
                return
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            compare_against_baseline(
                challenger_id="C2_feature_pruning",
                challenger_desc="Remove 6 unstable/low-importance features (25 -> 19)",
                challenger_holdout=meta.get("holdout_metrics", {}),
                challenger_zone=meta.get("zone_metrics", {}),
                challenger_stability=meta.get("stability_metrics", {}),
                challenger_features_dropped=meta.get("features_dropped", []),
                challenger_features_remaining=meta.get("feature_count", 0),
                baseline_metrics=active_baseline,
                output_dir=exp_dir,
                upgrade_status=meta.get("upgrade_candidate"),
                upgrade_reason=meta.get("upgrade_reason"),
                suitable_routing=meta.get("suitable_for_routing_improvement"),
                suitable_replace=meta.get("suitable_for_replacing_baseline_classifier"),
                baseline_label=bl_label,
            )
            print(f"\nC2 comparison report regenerated against {bl_label}.")

        elif challenger_id == "c3":
            exp_dir = PROJECT_ROOT / "model_bank" / "experiments" / "c3_decision_tuning"
            summary_path = exp_dir / "challenger_summary.json"
            if not summary_path.exists():
                print(f"ERROR: challenger_summary.json not found at {exp_dir}")
                print("       Run --run-c3 first.")
                return
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            best = summary.get("best_candidate", {})
            holdout = best.get("holdout_metrics", {})
            zone_p  = best.get("zone_precision", {})
            zone_r  = best.get("zone_ratio", {})
            workload = best.get("human_workload", {})
            stab    = best.get("stability", {})
            compare_against_baseline(
                challenger_id="C3_decision_tuning",
                challenger_desc="Decision-oriented + anti-overfitting tuning on C2 feature set",
                challenger_holdout={
                    "auc":        holdout.get("auc", 0),
                    "f1_reject":  holdout.get("f1_reject", 0),
                    "brier_score": holdout.get("brier", 0),
                    "ks":          holdout.get("ks", 0),
                },
                challenger_zone={
                    "low_zone_reject_precision":  zone_p.get("low_zone_reject_precision", 0),
                    "high_zone_approve_precision": zone_p.get("high_zone_approve_precision", 0),
                    "high_zone_ratio":             zone_r.get("high_zone_ratio", 0),
                    "manual_zone_ratio":           zone_r.get("manual_zone_ratio", 0),
                    "low_zone_ratio":              zone_r.get("low_zone_ratio", 0),
                    "human_review_workload_ratio": workload.get("human_review_workload_ratio", 0),
                },
                challenger_stability={
                    "avg_monitor_auc":       stab.get("avg_monitor_auc", 0),
                    "avg_monitor_f1_reject": stab.get("avg_monitor_f1_reject", 0),
                    "stability_score":       stab.get("stability_score", 0),
                },
                challenger_features_dropped=summary.get("features_dropped", []),
                challenger_features_remaining=summary.get("feature_count", 0),
                baseline_metrics=active_baseline,
                output_dir=exp_dir,
                upgrade_status=best.get("upgrade_candidate"),
                upgrade_reason=best.get("upgrade_reason"),
                suitable_routing=best.get("suitable_for_routing_improvement"),
                suitable_replace=best.get("suitable_for_replacing_baseline_classifier"),
                baseline_label=bl_label,
            )
            print(f"\nC3 comparison report regenerated against {bl_label}.")

        else:
            print(f"ERROR: Unknown challenger '{args.challenger}'. Use 'c2' or 'c3'.")

    elif args.create_baseline:
        name = _create_named_baseline(PROJECT_ROOT, baseline_name=args.baseline_name)
        if name:
            print(f"\nBaseline '{name}' is now active.")
            print("Future challengers will compare against this baseline.")

    else:
        # Default: Data + Training + Challenger Evaluation (no monitoring)
        run_full_pipeline(
            lower_threshold=args.lower_threshold,
            upper_threshold=args.upper_threshold,
            imbalance_strategy=args.imbalance,
            use_calibration=not args.no_calibration,
            model_names=args.models,
            include_challengers=True,
            include_monitoring=False,
        )


if __name__ == "__main__":
    main()
