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

Usage:
    python main.py                             # Data + Training (default)
    python main.py --all                       # Data + Training + Monitoring
    python main.py --data-only                 # Bronze -> Silver -> Gold only
    python main.py --train-only                # Four-phase training only
    python main.py --monitor-only              # Production monitoring only
    python main.py --run-c2                    # C2 Feature Pruning challenger
    python main.py --run-c3                    # C3 Decision Tuning challenger
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
    generate_routing_report,
    C2_METRICS,
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

    If any baseline already exists in the model_bank, this is a no-op.
    If no baseline exists, creates 'baseline_v1' from the training run and activates it.
    """
    from utils.baseline_manager import BaselineManager

    mgr = BaselineManager(str(project_root / "model_bank"))
    if mgr.list_baselines():
        return  # at least one baseline already exists

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

    Threshold resolution order:
      1. zone_policy_summary.json (Phase 3 recommended)
      2. champion_summary.json threshold_config
      3. CLI --lower-threshold / --upper-threshold
      4. ProductionMonitoringConfig defaults

    Baseline: policy_validation_predictions.csv (Phase 3, last labelled window before deployment).
    Monitoring target (demo mode): final_holdout_predictions.csv.
    In production, replace with actual production batch scoring data.
    """
    import json
    import numpy as np
    import pandas as pd

    if project_root is None:
        project_root = PROJECT_ROOT

    print("\n" + "=" * 70)
    print("  Production Monitoring")
    print("=" * 70)

    model_bank = project_root / "model_bank"
    if not model_bank.exists():
        print("  WARNING: model_bank not found; run training pipeline first.")
        return None

    run_dirs = sorted(
        [d for d in model_bank.iterdir()
         if d.is_dir() and (d.name.startswith("four_phase_") or d.name.startswith("rolling_training_v2_"))],
        reverse=True,
    )
    if not run_dirs:
        print("  WARNING: No training result directories found (four_phase_* or rolling_training_v2_*).")
        return None

    latest_dir = run_dirs[0]
    print(f"  Using latest training results: {latest_dir.name}")

    # Threshold resolution
    resolved_lower = lower_threshold
    resolved_upper = upper_threshold
    threshold_source = "default"

    zone_policy_path = latest_dir / "zone_policy_summary.json"
    champion_path    = latest_dir / "champion_summary.json"

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

    if lower_threshold is not None:
        resolved_lower   = lower_threshold
        threshold_source = "CLI argument"
    if upper_threshold is not None:
        resolved_upper   = upper_threshold
        threshold_source = "CLI argument"

    if resolved_lower is None:
        resolved_lower = ProductionMonitoringConfig().default_lower_threshold
    if resolved_upper is None:
        resolved_upper = ProductionMonitoringConfig().default_upper_threshold

    print(f"\n  Monitoring threshold:")
    print(f"    lower : {resolved_lower}")
    print(f"    upper : {resolved_upper}")
    print(f"    source: {threshold_source}")

    pv_pred_path    = latest_dir / "policy_validation_predictions.csv"
    holdout_pred_path = latest_dir / "final_holdout_predictions.csv"

    if not pv_pred_path.exists():
        print("  WARNING: policy_validation_predictions.csv not found.")
        return None

    print(f"\n  Baseline source: {pv_pred_path}")
    pv_df = pd.read_csv(pv_pred_path)
    baseline_scores = pv_df["pred_prob"].values
    print(f"    Rows: {len(baseline_scores)}  Mean score: {baseline_scores.mean():.4f}")

    config  = ProductionMonitoringConfig(
        default_lower_threshold=resolved_lower,
        default_upper_threshold=resolved_upper,
    )
    monitor = ProductionMonitor(config=config)
    monitor.set_baseline(baseline_scores, resolved_lower, resolved_upper)

    if champion_path.exists():
        champion_info = json.loads(champion_path.read_text(encoding="utf-8"))
        monitor.set_last_training_date(champion_info.get("created_at", "")[:10])

    if holdout_pred_path.exists():
        print(f"\n  Monitoring target (demo): {holdout_pred_path}")
        print("  NOTE: Replace with production batch scoring data in live deployment.")

        holdout_df  = pd.read_csv(holdout_pred_path)
        predictions = holdout_df["pred_prob"].values
        labels      = holdout_df["actual_label"].values if "actual_label" in holdout_df.columns else None

        result = monitor.run_production_monitoring(
            predictions=predictions,
            labels=labels,
            model_version=latest_dir.name,
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
            # 產生 dynamic retraining window
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
    else:
        print("  WARNING: final_holdout_predictions.csv not found; monitoring skipped.")
        return None


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
    include_monitoring: bool = False,
):
    """
    Run the full pipeline: data processing + four-phase training + (optional) production monitoring.

    Args:
        lower_threshold: Fallback threshold. Phase 3 recommended value takes priority.
        upper_threshold: Fallback threshold. Phase 3 recommended value takes priority.
        include_monitoring: If True, run monitoring after training.
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    print("\n" + "=" * 80)
    print("  ML PIPELINE - Full Run")
    print("  Four Phase Training Architecture")
    if include_monitoring:
        print("  (includes Production Monitoring)")
    print("=" * 80)

    # Step 1: Data processing
    run_data_pipeline(project_root)

    # Step 2: Four-phase training
    results = run_training_pipeline(
        project_root=project_root,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        imbalance_strategy=imbalance_strategy,
        use_calibration=use_calibration,
        model_names=model_names,
    )

    # Step 3: Production monitoring (optional)
    # Use Phase 3 selected thresholds from training results when available.
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
    if include_monitoring:
        print("  Steps completed: Data Pipeline -> Training -> Monitoring")
    print("=" * 80)

    return {"training": results, "monitoring": monitoring_result}


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
  python main.py                              # Run Data + Training (default)
  python main.py --all                        # Data + Training + Monitoring
  python main.py --data-only                  # Data processing only
  python main.py --train-only                 # Training only (requires Gold data)
  python main.py --monitor-only               # Production monitoring only
  python main.py --run-c2                     # Run C2 Feature Pruning challenger
  python main.py --run-c3                     # Run C3 Decision Tuning challenger
  python main.py --compare-baseline --challenger c2   # Generate comparison report
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
        "--monitor-only",
        action="store_true",
        help="Run production monitoring only (requires trained model)",
    )
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Run full pipeline: Data + Training + Monitoring",
    )
    mode_group.add_argument(
        "--run-c2",
        action="store_true",
        help="Run C2 Feature Pruning challenger (outputs to model_bank/experiments/c2_feature_pruning/)",
    )
    mode_group.add_argument(
        "--run-c3",
        action="store_true",
        help="Run C3 Decision Tuning challenger (outputs to model_bank/experiments/c3_decision_tuning/)",
    )
    mode_group.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Generate baseline comparison report for a challenger (requires --challenger)",
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
        from utils.challenger_manager import load_active_baseline_metrics
        challenger_id = args.challenger
        exp_dir = PROJECT_ROOT / "model_bank" / "experiments" / (
            "c2_feature_pruning" if challenger_id == "c2" else "c3_decision_tuning"
        )
        import json
        summary_path = exp_dir / "challenger_summary.json"
        if not summary_path.exists():
            print(f"ERROR: challenger_summary.json not found at {exp_dir}")
            print("       Run --run-c2 or --run-c3 first.")
            return
        challenger_metrics = json.loads(summary_path.read_text(encoding="utf-8"))
        active_baseline = load_active_baseline_metrics(PROJECT_ROOT)
        report_path = compare_against_baseline(
            challenger_id=challenger_id,
            challenger_metrics=challenger_metrics,
            baseline_metrics=active_baseline,
            output_dir=exp_dir,
        )
        print(f"\nComparison report saved: {report_path}")

    elif args.create_baseline:
        name = _create_named_baseline(PROJECT_ROOT, baseline_name=args.baseline_name)
        if name:
            print(f"\nBaseline '{name}' is now active.")
            print("Future challengers will compare against this baseline.")

    else:
        # Default: Data + Training (no monitoring)
        run_full_pipeline(
            lower_threshold=args.lower_threshold,
            upper_threshold=args.upper_threshold,
            imbalance_strategy=args.imbalance,
            use_calibration=not args.no_calibration,
            model_names=args.models,
            include_monitoring=False,
        )


if __name__ == "__main__":
    main()
