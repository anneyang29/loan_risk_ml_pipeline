# C2 Challenger: Feature Pruning

## Purpose

Remove 6 unstable / zero-importance features from baseline v1 (25 features -> 19 features)
to improve model stability and OOT generalisation without changing model type, hyperparameters,
or threshold policy logic.

## Features Dropped (6)

| Feature | Reason |
|---|---|
| 車齡_異常旗標 | gain=0.000, mean_shift=2111.7 (extreme instability) |
| 內部往來次數_是否特殊值 | gain=0.000, corr=0.00 (zero signal) |
| 近半年同業查詢次數_是否缺失 | gain=0.000, corr=0.03 (zero importance) |
| 教育程度_是否缺失 | gain=0.013, mean_shift=1.80 (high drift) |
| 負債月所得比_scaled | gain=0.008 (lowest non-zero importance) |
| 教育_所得交互_scaled | gain=0.022, PSI=0.68 (severe drift) |

## Key Metrics vs Baseline v1

| Metric | Baseline v1 | C2 |
|---|---|---|
| AUC | 0.8953 | 0.9016 |
| F1_reject | 0.3975 | 0.3903 |
| lzp (low zone precision) | 0.5298 | 0.5478 |
| manual_zone_ratio | 0.0955 | 0.0868 |
| rolling_stability (CV std) | 0.0308 | 0.0286 |

## Upgrade Candidate

- AUC improved (+0.0063), rolling stability improved.
- F1_reject slightly regressed (−0.0072); upgrade candidate is **conditional**.
- Suitable for **routing improvement** (lower manual review burden).

## How to Run

```bash
python main.py --run-c2
```

or directly:

```bash
python experiments/c2_feature_pruning/run_c2_challenger.py
```

## Output Location

```
model_bank/experiments/c2_feature_pruning/
  challenger_summary.json
  final_holdout_metrics.json
  final_holdout_predictions.csv
  rolling_results.csv
  c2_vs_baseline_comparison.md
  c2_routing_report.md
```
