# Loan Risk ML Pipeline

An end-to-end machine learning pipeline for credit risk assessment, covering data processing, four-phase model training, baseline creation, challenger evaluation, final decision reporting, and optional production monitoring.

---

## Project Overview

The goal of this project is not simply to train a classification model, but to build a more complete **credit risk model governance workflow**.  
The overall pipeline starts from data cleaning, connects to a time-based four-phase training framework, and further incorporates baseline lifecycle management, challenger comparison, final decision reporting, and downstream monitoring.

The current end-to-end lifecycle supported by this project includes:

1. **Data Pipeline**  
   Bronze → Silver → Gold

2. **Baseline Pipeline**  
   Four-phase training + automatic baseline creation / activation

3. **Challenger Stage**  
   Run C2 / C3 and compare them against the active baseline

4. **Final Decision Stage**  
   Consolidate the results of the baseline and challengers, determine the final chosen candidate, and automatically generate the final report

5. **Optional Monitoring**  
   If `--all` is used, production-style monitoring is executed after the stages above

---

## Core Project Philosophy

### This project does not simply optimize for the best-looking aggregate metrics

This project follows a **routing-first** design philosophy rather than focusing only on maximizing aggregate classification metrics such as AUC or F1.

In a credit approval setting, the value of the model is not only about achieving a high score, but about whether it can effectively support practical case routing, for example:

- High zone: strong approve candidates with high approval probability
- Manual zone: cases that require human review
- Low zone: higher-risk cases suitable for enhanced reject screening

Therefore, this project places greater emphasis on:

- low-zone reject precision
- high-zone approve precision
- human review workload ratio
- whether the proportion of cases across zones is reasonable
- whether the overall routing structure is more useful in practice

Traditional classification metrics such as AUC, F1_reject, KS, and Brier score are still tracked, but in this project they are treated more as **risk-control guardrails** rather than the sole optimization objective.

In other words, the best model in this project is not necessarily the one with the highest aggregate metric, but the one that can deliver the most effective routing outcome under acceptable risk controls.

---

## Four-Phase Training Framework

The baseline model is trained using a four-phase framework:

### Phase 1: Model Development
Use 18 months of data for rolling training / monitoring to compare candidate models and assess stability.

### Phase 2: Champion Retraining
Retrain the champion strategy selected in Phase 1 using the full development dataset to produce the formal champion model.

### Phase 3: Policy Validation
Use the following 4 months of data for threshold / zone policy tuning.

### Phase 4: Final Blind Holdout
Reserve the last 2 months as a fully untouched final evaluation set, with no further model tuning or threshold adjustment.

This means the key final results presented by the project, such as:
- confusion matrix
- final classifier metrics
- zone routing outcomes

should be based primarily on the **Final Blind Holdout**, rather than the training set or policy validation set.

---

## Challenger Design

This project currently includes two challengers:

### C2: Feature Pruning
Removes unstable or low-importance features to test whether model stability and routing performance can be improved.

### C3: Decision Tuning
Focuses on decision-oriented and anti-overfitting adjustments to test whether routing performance can outperform the baseline.

All challengers are not deployed directly. Instead, they are first compared against the current **active baseline**, and then evaluated under the routing-first decision logic to determine whether they are worth promoting.

---

## Final Decision Logic

The current final decisions in the project are mainly divided into three categories:

### 1. RETAIN_BASELINE
No challenger truly outperforms the baseline, so the baseline remains the official version.

### 2. FULL_UPGRADE
The challenger is strong enough in both routing performance and classifier guardrails, and can be considered a full upgrade candidate.

### 3. ROUTING_ONLY_UPGRADE
The challenger improves routing performance, but the overall classifier metrics may not be strong enough to justify a full replacement, so it is only considered a routing strategy upgrade candidate.

The final chosen candidate drives:
- the content of the final decision report
- confusion matrix
- zone charts
- SHAP explainability plots
- the final decision summary JSON

---

## Project Structure

```text
loan_risk_ml_pipeline/
├── config/                     # pipeline configuration
├── experiments/               # challenger / experiment-related outputs
├── utils/                     # core modules
├── .dockerignore
├── .gitignore
├── Dockerfile
├── main.py                    # main entry point
├── requirements.txt
└── train.py                   # legacy / alternative training entry
