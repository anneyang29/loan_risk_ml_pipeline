# C3 Challenger: Decision Tuning

## Purpose

Build on C2 (19-feature set) to find calibration method + threshold policy combinations
that improve reject zone routing precision and reduce human review workload, while
keeping AUC / F1_reject comparable to baseline v1.

## Strategy

- **Feature set**: C2 pruned set (19 features, same as C2 challenger)
- **Calibration methods tested**: isotonic, sigmoid
- **Threshold candidates**: grid search over lower/upper zone boundaries
- **Scoring**: decision_score = weighted sum of (AUC, F1_reject, lzp) − overfitting_penalty − robustness_penalty
- **Hard constraints**: high_zone_precision >= 0.98, lzp >= 0.65, manual_ratio <= 0.12

## Business Targets

| KPI | Constraint |
|---|---|
| high_zone_approve_precision | >= 0.98 |
| low_zone_reject_precision (lzp) | >= 0.65 |
| manual_zone_ratio | <= 0.12 |
| human_review_workload_ratio | <= 0.15 |
| auto_decision_rate | >= 0.85 |

## Upgrade Candidate Evaluation

- **yes**: all hard constraints met AND lzp improved AND F1_reject not regressed
- **conditional**: lzp improved but F1_reject regressed
- **no**: failed hard constraints

## How to Run

```bash
python main.py --run-c3
```

or directly:

```bash
python experiments/c3_decision_tuning/run_decision_tuning.py
```

## Output Location

```
model_bank/experiments/c3_decision_tuning/
  challenger_summary.json
  c3_decision_tuning_results.csv
  final_holdout_metrics.json
  final_holdout_predictions.csv
  rolling_results.csv
  c3_vs_baseline_comparison.md
  c3_routing_report.md
  c3_decision_tuning_report.md
```
