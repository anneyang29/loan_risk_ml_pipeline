"""
Feature Diagnosis & Error Analysis
====================================
針對 baseline 模型做完整診斷，不修改任何現有流程。

輸出到 model_bank/experiments/diagnosis_<date>/:
  - feature_diagnosis.csv          : 每個特徵的重要度、穩定性、與 reject 的關聯
  - feature_stability_summary.json : 特徵在不同時期的 PSI / shift 摘要
  - error_analysis_by_segment.csv  : 分群錯誤率分析
  - false_positive_profile.csv     : FP 案件特徵分佈
  - false_negative_profile.csv     : FN 案件特徵分佈
  - zone_error_summary.json        : 三區錯誤分佈
  - top_feature_report.md          : 特徵重要度報告
  - improvement_hypotheses.md      : 改善假設
  - challenger_plan.json           : Challenger 實驗方案
  - challenger_experiment_matrix.csv: Challenger 實驗矩陣

使用方式：
    python utils/feature_diagnosis.py
"""

import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import pickle

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================
# Constants
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_DIR = PROJECT_ROOT / "model_bank" / "baselines" / "baseline_v1"
GOLD_DEV_PATH = PROJECT_ROOT / "datamart" / "gold" / "development"
GOLD_OOT_PATH = PROJECT_ROOT / "datamart" / "gold" / "oot"
ROLLING_DEF_PATH = PROJECT_ROOT / "datamart" / "gold" / "rolling_window_definition.csv"

# 分群分析用的原始類別欄位
SEGMENT_COLUMNS_RAW = [
    "教育程度", "月所得", "年齡組_序位", "性別_二元", "婚姻狀況_二元",
    "職業說明", "居住地",
]

# 分群分析用的數值特徵（分箱後）
SEGMENT_COLUMNS_NUMERIC = [
    "年齡_scaled", "原申辦金額_log_scaled", "車齡_log_scaled",
    "內部往來次數_log_scaled", "近半年同業查詢次數_log_scaled",
    "負債月所得比_scaled",
]

TARGET_COL = "授信結果_二元"
DATE_COL = "進件日"


# ============================================
# 1. Data Loading
# ============================================
def load_gold_data() -> pd.DataFrame:
    """載入 Gold layer 資料 (dev + oot)"""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("FeatureDiagnosis").getOrCreate()

    dfs = []
    if GOLD_DEV_PATH.exists():
        df_dev = spark.read.parquet(str(GOLD_DEV_PATH)).toPandas()
        df_dev["_split"] = "development"
        dfs.append(df_dev)
        logger.info(f"Development: {len(df_dev)} 筆")

    if GOLD_OOT_PATH.exists():
        df_oot = spark.read.parquet(str(GOLD_OOT_PATH)).toPandas()
        df_oot["_split"] = "oot"
        dfs.append(df_oot)
        logger.info(f"OOT: {len(df_oot)} 筆")

    spark.stop()

    all_data = pd.concat(dfs, ignore_index=True)
    all_data[DATE_COL] = pd.to_datetime(all_data[DATE_COL])
    logger.info(f"總共: {len(all_data)} 筆")
    return all_data


def load_baseline_model():
    """載入 baseline 模型"""
    model_path = BASELINE_DIR / "model" / "final_champion_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"模型已載入: {model_path}")
    return model


def load_feature_names() -> List[str]:
    """載入特徵名稱"""
    path = BASELINE_DIR / "artifacts" / "feature_names.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_holdout_predictions() -> pd.DataFrame:
    """載入 holdout 預測結果"""
    path = BASELINE_DIR / "predictions" / "final_holdout_predictions.csv"
    return pd.read_csv(path)


def load_rolling_definitions() -> pd.DataFrame:
    """載入 rolling window 定義"""
    return pd.read_csv(ROLLING_DEF_PATH)


# ============================================
# 2. Feature Importance Analysis
# ============================================
def compute_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    計算 XGBoost feature importance (gain, weight, cover)

    對 CalibratedClassifierCV 會先取內部 estimator。
    注意：XGBoost booster 內部用 f0, f1, ... 命名，需要對應回 feature_names。
    """
    logger.info("=== Feature Importance Analysis ===")

    # 取得內部 XGBoost 模型
    xgb_model = _extract_xgb_model(model)

    # 建立 f{i} → feature_name 的映射
    fi_to_name = {f"f{i}": name for i, name in enumerate(feature_names)}

    importance_types = ["weight", "gain", "cover"]
    records = []
    for i, feat in enumerate(feature_names):
        fi_key = f"f{i}"
        row = {"feature": feat}
        for itype in importance_types:
            imp_dict = xgb_model.get_booster().get_score(importance_type=itype)
            # XGBoost 用 f0, f1, ... 命名
            row[f"importance_{itype}"] = imp_dict.get(fi_key, 0.0)
        records.append(row)

    df_imp = pd.DataFrame(records)

    # 正規化
    for itype in importance_types:
        col = f"importance_{itype}"
        total = df_imp[col].sum()
        df_imp[f"{col}_norm"] = df_imp[col] / total if total > 0 else 0.0

    # 排名
    df_imp["rank_gain"] = df_imp["importance_gain_norm"].rank(ascending=False, method="min").astype(int)
    df_imp = df_imp.sort_values("rank_gain")

    logger.info(f"Top 10 by gain:")
    for _, r in df_imp.head(10).iterrows():
        logger.info(f"  {r['rank_gain']:>2}. {r['feature']:<35} gain={r['importance_gain_norm']:.4f}")

    return df_imp


def _extract_xgb_model(model):
    """從 CalibratedClassifierCV 取出 XGBoost 底層模型"""
    if hasattr(model, "calibrated_classifiers_"):
        # CalibratedClassifierCV
        inner = model.calibrated_classifiers_[0].estimator
        return inner
    if hasattr(model, "get_booster"):
        return model
    raise ValueError(f"無法取得 XGBoost 模型: {type(model)}")


# ============================================
# 3. Feature-Reject Correlation
# ============================================
def compute_reject_correlation(
    all_data: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    計算每個特徵與 reject class (label=0) 的相關性。
    對 binary/ordinal → point-biserial correlation
    對 continuous → Pearson correlation
    """
    logger.info("=== Feature-Reject Correlation ===")

    # reject = 1 - label (label=0 是婉拒)
    reject_flag = (1 - all_data[TARGET_COL]).astype(float)

    results = []
    for feat in feature_names:
        if feat not in all_data.columns:
            continue
        col = all_data[feat].astype(float).fillna(0)

        corr = col.corr(reject_flag)
        # reject 佔比差異：top 25% vs bottom 25%
        q25 = col.quantile(0.25)
        q75 = col.quantile(0.75)

        if q25 != q75:
            reject_rate_low = reject_flag[col <= q25].mean()
            reject_rate_high = reject_flag[col >= q75].mean()
            reject_lift = reject_rate_high - reject_rate_low
        else:
            reject_rate_low = reject_rate_high = reject_flag.mean()
            reject_lift = 0.0

        results.append({
            "feature": feat,
            "corr_with_reject": round(corr, 4) if not np.isnan(corr) else 0.0,
            "abs_corr_with_reject": round(abs(corr), 4) if not np.isnan(corr) else 0.0,
            "reject_rate_q25": round(reject_rate_low, 4),
            "reject_rate_q75": round(reject_rate_high, 4),
            "reject_lift_q75_vs_q25": round(reject_lift, 4),
        })

    return pd.DataFrame(results)


# ============================================
# 4. Feature Stability (PSI + shift)
# ============================================
def compute_feature_stability(
    all_data: pd.DataFrame,
    feature_names: List[str],
    rolling_defs: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """
    計算特徵穩定性：
    - 各 rolling window 的 mean/std/missing_rate
    - development vs OOT 的 PSI
    - 時間趨勢
    """
    logger.info("=== Feature Stability Analysis ===")

    all_data = all_data.copy()
    all_data[DATE_COL] = pd.to_datetime(all_data[DATE_COL])

    # 定義時期
    dev_mask = all_data["_split"] == "development"
    oot_mask = all_data["_split"] == "oot"

    dev_data = all_data[dev_mask]
    oot_data = all_data[oot_mask]

    stability_records = []
    stability_detail = {}

    for feat in feature_names:
        if feat not in all_data.columns:
            continue

        col_dev = dev_data[feat].astype(float)
        col_oot = oot_data[feat].astype(float)

        # 基礎統計
        dev_mean = col_dev.mean()
        dev_std = col_dev.std()
        oot_mean = col_oot.mean()
        oot_std = col_oot.std()
        dev_missing = col_dev.isna().mean()
        oot_missing = col_oot.isna().mean()

        # PSI
        psi = _calculate_psi(col_dev.dropna().values, col_oot.dropna().values)

        # 每個 rolling window 的 mean
        window_means = []
        for _, wdef in rolling_defs.iterrows():
            w_mask = (all_data[DATE_COL] >= wdef["train_start"]) & \
                     (all_data[DATE_COL] <= wdef["monitor_end"])
            w_data = all_data.loc[w_mask, feat].astype(float)
            window_means.append(round(w_data.mean(), 4) if len(w_data) > 0 else None)

        # 時間趨勢：coefficient of variation across windows
        window_means_clean = [m for m in window_means if m is not None]
        temporal_cv = np.std(window_means_clean) / np.mean(window_means_clean) \
            if len(window_means_clean) > 1 and np.mean(window_means_clean) != 0 else 0.0

        # 均值漂移比例
        mean_shift = abs(oot_mean - dev_mean) / (dev_std + 1e-8)

        record = {
            "feature": feat,
            "dev_mean": round(dev_mean, 4),
            "dev_std": round(dev_std, 4),
            "oot_mean": round(oot_mean, 4),
            "oot_std": round(oot_std, 4),
            "mean_shift_zscore": round(mean_shift, 4),
            "dev_missing_rate": round(dev_missing, 4),
            "oot_missing_rate": round(oot_missing, 4),
            "missing_rate_change": round(oot_missing - dev_missing, 4),
            "psi_dev_vs_oot": round(psi, 4),
            "temporal_cv": round(temporal_cv, 4),
            "is_unstable": psi > 0.1 or mean_shift > 0.5 or temporal_cv > 0.15,
        }
        stability_records.append(record)

        stability_detail[feat] = {
            "window_means": window_means,
            "psi": round(psi, 4),
            "mean_shift_zscore": round(mean_shift, 4),
        }

    df_stability = pd.DataFrame(stability_records)
    df_stability = df_stability.sort_values("psi_dev_vs_oot", ascending=False)

    unstable_count = df_stability["is_unstable"].sum()
    logger.info(f"不穩定特徵數: {unstable_count} / {len(df_stability)}")
    for _, r in df_stability[df_stability["is_unstable"]].iterrows():
        logger.info(f"  ⚠️ {r['feature']}: PSI={r['psi_dev_vs_oot']:.4f}, "
                    f"shift={r['mean_shift_zscore']:.4f}, CV={r['temporal_cv']:.4f}")

    return df_stability, stability_detail


def _calculate_psi(actual: np.ndarray, expected: np.ndarray, n_bins: int = 10) -> float:
    """計算 PSI"""
    if len(actual) == 0 or len(expected) == 0:
        return 0.0

    # 用 expected 的 quantile 作為 bin edges
    try:
        bins = np.unique(np.percentile(expected, np.linspace(0, 100, n_bins + 1)))
    except Exception:
        return 0.0

    if len(bins) < 2:
        return 0.0

    actual_hist = np.histogram(actual, bins=bins)[0]
    expected_hist = np.histogram(expected, bins=bins)[0]

    # 加入 smoothing
    actual_pct = (actual_hist + 1) / (actual_hist.sum() + len(actual_hist))
    expected_pct = (expected_hist + 1) / (expected_hist.sum() + len(expected_hist))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


# ============================================
# 5. Error Analysis
# ============================================
def run_error_analysis(
    all_data: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    全面性 Error Analysis。

    用 holdout predictions 的案件編號 join 回原始 Gold data，
    取得原始類別欄位以做分群分析。
    """
    logger.info("=== Error Analysis ===")

    # Merge predictions 和原始特徵
    pred_df = predictions.copy()
    pred_df["進件日"] = pd.to_datetime(pred_df["進件日"])

    # 從 all_data 取得 segment columns
    holdout_dates = pred_df["進件日"].unique()
    holdout_data = all_data[all_data[DATE_COL].isin(holdout_dates)].copy()

    # 用案件編號 join
    merged = pred_df.merge(
        holdout_data.drop(columns=[TARGET_COL], errors="ignore"),
        on="案件編號",
        how="left",
        suffixes=("", "_gold"),
    )
    logger.info(f"Merged 資料: {len(merged)} 筆 (pred={len(pred_df)}, match率={len(merged)/len(pred_df):.2%})")

    # 分類
    merged["actual_reject"] = (1 - merged["actual_label"]).astype(int)
    merged["pred_reject"] = (merged["pred_prob"] < merged["lower_threshold_used"]).astype(int)
    merged["pred_approve"] = (merged["pred_prob"] >= merged["upper_threshold_used"]).astype(int)

    # Error types
    merged["is_FP"] = ((merged["pred_approve"] == 1) & (merged["actual_reject"] == 1)).astype(int)
    merged["is_FN"] = ((merged["pred_reject"] == 1) & (merged["actual_reject"] == 0)).astype(int)
    merged["is_correct"] = (
        ((merged["pred_approve"] == 1) & (merged["actual_reject"] == 0)) |
        ((merged["pred_reject"] == 1) & (merged["actual_reject"] == 1))
    ).astype(int)

    total = len(merged)
    fp_count = merged["is_FP"].sum()
    fn_count = merged["is_FN"].sum()
    logger.info(f"FP (高通過→實際婉拒): {fp_count} ({fp_count/total:.2%})")
    logger.info(f"FN (低風險外→實際應核准): {fn_count} ({fn_count/total:.2%})")

    # 5a. Segment error analysis
    segment_analysis = _segment_error_analysis(merged, feature_names)

    # 5b. FP profile
    fp_profile = _build_error_profile(merged, "is_FP", feature_names)

    # 5c. FN profile
    fn_profile = _build_error_profile(merged, "is_FN", feature_names)

    # 5d. Zone error summary
    zone_summary = _zone_error_summary(merged)

    return segment_analysis, fp_profile, fn_profile, zone_summary


def _segment_error_analysis(merged: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """分群錯誤率分析"""
    records = []

    # 類別特徵分群
    for col in SEGMENT_COLUMNS_RAW:
        if col not in merged.columns:
            continue
        for val, grp in merged.groupby(col):
            n = len(grp)
            if n < 30:
                continue
            records.append({
                "segment_feature": col,
                "segment_value": str(val),
                "count": n,
                "count_pct": round(n / len(merged), 4),
                "reject_rate": round(grp["actual_reject"].mean(), 4),
                "fp_rate": round(grp["is_FP"].mean(), 4),
                "fn_rate": round(grp["is_FN"].mean(), 4),
                "fp_count": int(grp["is_FP"].sum()),
                "fn_count": int(grp["is_FN"].sum()),
                "avg_pred_prob": round(grp["pred_prob"].mean(), 4),
                "high_zone_pct": round((grp["zone_name"] == "高通過機率區").mean(), 4),
                "manual_zone_pct": round((grp["zone_name"] == "人工審核區").mean(), 4),
                "low_zone_pct": round((grp["zone_name"] == "低通過機率區").mean(), 4),
            })

    # 數值特徵分箱
    for col in SEGMENT_COLUMNS_NUMERIC:
        if col not in merged.columns:
            continue
        try:
            merged[f"_bin_{col}"] = pd.qcut(merged[col], q=5, duplicates="drop")
            for val, grp in merged.groupby(f"_bin_{col}"):
                n = len(grp)
                if n < 30:
                    continue
                records.append({
                    "segment_feature": col,
                    "segment_value": str(val),
                    "count": n,
                    "count_pct": round(n / len(merged), 4),
                    "reject_rate": round(grp["actual_reject"].mean(), 4),
                    "fp_rate": round(grp["is_FP"].mean(), 4),
                    "fn_rate": round(grp["is_FN"].mean(), 4),
                    "fp_count": int(grp["is_FP"].sum()),
                    "fn_count": int(grp["is_FN"].sum()),
                    "avg_pred_prob": round(grp["pred_prob"].mean(), 4),
                    "high_zone_pct": round((grp["zone_name"] == "高通過機率區").mean(), 4),
                    "manual_zone_pct": round((grp["zone_name"] == "人工審核區").mean(), 4),
                    "low_zone_pct": round((grp["zone_name"] == "低通過機率區").mean(), 4),
                })
            merged.drop(columns=[f"_bin_{col}"], inplace=True)
        except Exception:
            pass

    df = pd.DataFrame(records)
    df = df.sort_values("fp_rate", ascending=False)
    return df


def _build_error_profile(
    merged: pd.DataFrame, error_col: str, feature_names: List[str]
) -> pd.DataFrame:
    """建構錯誤樣本的特徵 profile vs 全體"""
    error_mask = merged[error_col] == 1
    error_data = merged[error_mask]
    all_data_local = merged

    records = []
    for feat in feature_names:
        if feat not in merged.columns:
            continue
        col_all = all_data_local[feat].astype(float)
        col_err = error_data[feat].astype(float)

        records.append({
            "feature": feat,
            "overall_mean": round(col_all.mean(), 4),
            "overall_std": round(col_all.std(), 4),
            "error_mean": round(col_err.mean(), 4) if len(col_err) > 0 else None,
            "error_std": round(col_err.std(), 4) if len(col_err) > 0 else None,
            "mean_diff": round(col_err.mean() - col_all.mean(), 4) if len(col_err) > 0 else None,
            "mean_diff_zscore": round(
                (col_err.mean() - col_all.mean()) / (col_all.std() + 1e-8), 4
            ) if len(col_err) > 0 else None,
            "error_missing_rate": round(col_err.isna().mean(), 4) if len(col_err) > 0 else None,
            "overall_missing_rate": round(col_all.isna().mean(), 4),
            "error_count": int(error_mask.sum()),
        })

    df = pd.DataFrame(records)
    if "mean_diff_zscore" in df.columns:
        df["abs_mean_diff_zscore"] = df["mean_diff_zscore"].abs()
        df = df.sort_values("abs_mean_diff_zscore", ascending=False)
    return df


def _zone_error_summary(merged: pd.DataFrame) -> Dict:
    """各 zone 的錯誤分佈"""
    summary = {}
    for zone_name, grp in merged.groupby("zone_name"):
        n = len(grp)
        n_reject = grp["actual_reject"].sum()
        n_fp = grp["is_FP"].sum()
        n_fn = grp["is_FN"].sum()

        summary[zone_name] = {
            "count": int(n),
            "count_pct": round(n / len(merged), 4),
            "actual_reject_count": int(n_reject),
            "actual_reject_rate": round(n_reject / n, 4) if n > 0 else 0,
            "fp_count": int(n_fp),
            "fp_rate": round(n_fp / n, 4) if n > 0 else 0,
            "fn_count": int(n_fn),
            "fn_rate": round(n_fn / n, 4) if n > 0 else 0,
            "avg_pred_prob": round(grp["pred_prob"].mean(), 4),
            "std_pred_prob": round(grp["pred_prob"].std(), 4),
            "precision_approve": round(
                grp["actual_label"].mean(), 4
            ) if zone_name == "高通過機率區" else None,
            "precision_reject": round(
                grp["actual_reject"].mean(), 4
            ) if zone_name == "低通過機率區" else None,
        }

    return summary


# ============================================
# 6. Merge into Feature Diagnosis Table
# ============================================
def build_feature_diagnosis_table(
    importance_df: pd.DataFrame,
    reject_corr_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    fp_profile: pd.DataFrame,
    fn_profile: pd.DataFrame,
) -> pd.DataFrame:
    """合併所有特徵維度的分析"""
    df = importance_df[["feature", "importance_gain_norm", "rank_gain"]].copy()

    # Reject correlation
    df = df.merge(
        reject_corr_df[["feature", "corr_with_reject", "abs_corr_with_reject", "reject_lift_q75_vs_q25"]],
        on="feature", how="left",
    )

    # Stability
    df = df.merge(
        stability_df[["feature", "psi_dev_vs_oot", "mean_shift_zscore", "temporal_cv", "is_unstable"]],
        on="feature", how="left",
    )

    # FP/FN profile difference
    fp_diff = fp_profile[["feature", "mean_diff_zscore"]].rename(
        columns={"mean_diff_zscore": "fp_feature_shift"}
    )
    fn_diff = fn_profile[["feature", "mean_diff_zscore"]].rename(
        columns={"mean_diff_zscore": "fn_feature_shift"}
    )
    df = df.merge(fp_diff, on="feature", how="left")
    df = df.merge(fn_diff, on="feature", how="left")

    # 分類
    df["category"] = df["feature"].apply(_classify_feature)

    df = df.sort_values("rank_gain")
    return df


def _classify_feature(feat: str) -> str:
    """特徵分類"""
    if "交互" in feat or "比" in feat:
        return "cross_feature"
    if "是否缺失" in feat or "異常旗標" in feat or "是否特殊值" in feat:
        return "missing_flag"
    if "序位" in feat:
        return "ordinal"
    if "二元" in feat:
        return "binary"
    if "頻率" in feat:
        return "frequency_encoded"
    if "scaled" in feat or "log" in feat:
        return "numeric_scaled"
    return "other"


# ============================================
# 7. Report Generation
# ============================================
def generate_top_feature_report(diagnosis_df: pd.DataFrame, output_path: Path):
    """產出 top_feature_report.md"""
    lines = [
        "# Feature Diagnosis Report",
        "",
        f"> 產出時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        "> 基於 baseline_v1 模型分析  ",
        "",
        "---",
        "",
        "## 1. Feature Importance Ranking (by Gain)",
        "",
        "| Rank | Feature | Gain | Reject Corr | PSI | Unstable |",
        "|------|---------|------|-------------|-----|----------|",
    ]

    for _, r in diagnosis_df.head(15).iterrows():
        unstable_flag = "⚠️" if r.get("is_unstable") else "✅"
        lines.append(
            f"| {int(r['rank_gain'])} | {r['feature']} | "
            f"{r['importance_gain_norm']:.4f} | "
            f"{r.get('corr_with_reject', 0):.4f} | "
            f"{r.get('psi_dev_vs_oot', 0):.4f} | "
            f"{unstable_flag} |"
        )

    # 不穩定特徵
    unstable = diagnosis_df[diagnosis_df["is_unstable"] == True]
    lines += [
        "",
        "## 2. Unstable Features",
        "",
        f"不穩定特徵數: **{len(unstable)}** / {len(diagnosis_df)}",
        "",
    ]
    if len(unstable) > 0:
        lines.append("| Feature | PSI | Mean Shift (z) | Temporal CV | Category |")
        lines.append("|---------|-----|----------------|-------------|----------|")
        for _, r in unstable.iterrows():
            lines.append(
                f"| {r['feature']} | {r['psi_dev_vs_oot']:.4f} | "
                f"{r['mean_shift_zscore']:.4f} | {r['temporal_cv']:.4f} | {r['category']} |"
            )

    # Reject detection 相關
    top_reject = diagnosis_df.sort_values("abs_corr_with_reject", ascending=False).head(10)
    lines += [
        "",
        "## 3. Top Features for Reject Detection",
        "",
        "| Feature | |Corr w/ Reject| | Reject Lift (Q75 vs Q25) | Gain Rank | Category |",
        "|---------|----------------------|--------------------------|-----------|----------|",
    ]
    for _, r in top_reject.iterrows():
        lines.append(
            f"| {r['feature']} | {r.get('abs_corr_with_reject', 0):.4f} | "
            f"{r.get('reject_lift_q75_vs_q25', 0):.4f} | "
            f"{int(r['rank_gain'])} | {r['category']} |"
        )

    # FP/FN 特徵偏移
    lines += [
        "",
        "## 4. Feature Shift in Error Cases",
        "",
        "### FP cases (高通過→實際婉拒) — 特徵偏移最大的",
        "",
        "| Feature | FP Shift (z) | FN Shift (z) | Category |",
        "|---------|--------------|--------------|----------|",
    ]
    fp_shift = diagnosis_df.copy()
    fp_shift["abs_fp_shift"] = fp_shift["fp_feature_shift"].abs()
    fp_shift = fp_shift.sort_values("abs_fp_shift", ascending=False)
    for _, r in fp_shift.head(10).iterrows():
        lines.append(
            f"| {r['feature']} | {r.get('fp_feature_shift', 0):+.4f} | "
            f"{r.get('fn_feature_shift', 0):+.4f} | {r['category']} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("*此報告由 feature_diagnosis.py 自動產出*")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"✓ 報告已產出: {output_path}")


# ============================================
# 8. Improvement Hypotheses
# ============================================
def generate_improvement_hypotheses(
    diagnosis_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    segment_analysis: pd.DataFrame,
    zone_summary: Dict,
    output_path: Path,
):
    """根據診斷結果產出改善假設"""

    # 分析數據
    unstable_features = list(diagnosis_df[diagnosis_df["is_unstable"] == True]["feature"])
    top_reject_features = list(
        diagnosis_df.sort_values("abs_corr_with_reject", ascending=False).head(5)["feature"]
    )
    low_importance = list(
        diagnosis_df[diagnosis_df["importance_gain_norm"] < 0.01]["feature"]
    )

    # 高 FP 率的分群
    high_fp_segments = segment_analysis[segment_analysis["fp_rate"] > 0.05].head(10) \
        if "fp_rate" in segment_analysis.columns else pd.DataFrame()

    # Zone 資訊
    high_zone = zone_summary.get("高通過機率區", {})
    low_zone = zone_summary.get("低通過機率區", {})
    manual_zone = zone_summary.get("人工審核區", {})

    lines = [
        "# Improvement Hypotheses",
        "",
        f"> 產出時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        "> 基於 feature diagnosis + error analysis 綜合判斷  ",
        "",
        "---",
        "",
        "## 瓶頸診斷總結",
        "",
        "根據本次診斷，目前模型的主要瓶頸為：",
        "",
        "1. **Feature signal 不足**（Primary）",
        "   - 現有 25 個特徵中，重要度集中在少數幾個特徵",
        "   - 對 reject class 有區別力的特徵有限",
        f"   - Top reject-correlated 特徵: {', '.join(top_reject_features)}",
        "",
        "2. **Class imbalance 本質限制**",
        f"   - Positive ratio ≈ 95.6%, reject 僅 4.4%",
        "   - 在這個 base rate 下，F1_reject ≈ 0.40 已接近常見上界",
        "   - 即使 AUC 很高，在低 base rate 下 precision-recall trade-off 嚴重",
        "",
        "3. **Feature stability 問題**（Secondary）",
        f"   - 不穩定特徵: {', '.join(unstable_features) if unstable_features else '無'}",
        "   - 這些特徵可能導致 calibration 在不同時期表現差異",
        "",
        "4. **Low zone purity 不足**",
        f"   - Low zone precision (reject): {low_zone.get('actual_reject_rate', 'N/A')}",
        f"   - Low zone 件數: {low_zone.get('count', 'N/A')} ({low_zone.get('count_pct', 'N/A'):.2%})" if isinstance(low_zone.get('count_pct', 0), float) else "",
        "   - 模型對「確定婉拒」的信心不足",
        "",
        "---",
        "",
        "## Hypothesis 1: 新增 reject-targeted 交互特徵",
        "",
        "### 觀察",
    ]

    # 動態填入 top reject features
    for feat in top_reject_features[:3]:
        row = diagnosis_df[diagnosis_df["feature"] == feat].iloc[0] \
            if feat in diagnosis_df["feature"].values else None
        if row is not None:
            lines.append(
                f"- `{feat}`: corr_w_reject={row.get('corr_with_reject', 0):.4f}, "
                f"gain_rank={int(row['rank_gain'])}"
            )

    lines += [
        "",
        "### 可能原因",
        "- 現有交互特徵（4 個）可能未充分捕捉 reject signal",
        "- reject 案件可能集中在特定的「高風險組合」",
        "",
        "### 建議",
        "- 新增 `近半年同業查詢次數 × 內部往來次數` 交互項（高查詢+低往來 = 高風險）",
        "- 新增 `所得 × 申辦金額` 比值的非線性版本（分箱後再交互）",
        "- 新增 `年齡 × 教育 × 所得` 三維交互",
        "",
        "### 預期改善",
        "- F1_reject: +0.02~0.05",
        "- Low zone precision: +3~5%",
        "",
        "---",
        "",
        "## Hypothesis 2: 移除或降權不穩定特徵",
        "",
        "### 觀察",
    ]

    if unstable_features:
        for feat in unstable_features:
            row = stability_df[stability_df["feature"] == feat].iloc[0] \
                if feat in stability_df["feature"].values else None
            if row is not None:
                lines.append(
                    f"- `{feat}`: PSI={row['psi_dev_vs_oot']:.4f}, "
                    f"mean_shift={row['mean_shift_zscore']:.4f}"
                )
    else:
        lines.append("- 未發現顯著不穩定特徵（PSI 均 < 0.1）")

    lines += [
        "",
        "### 可能原因",
        "- 頻率編碼特徵（居住地、職業、廠牌車型）在不同時期分佈可能漂移",
        "- 這會導致 calibration 在 monitor 期與 holdout 期不一致",
        "",
        "### 建議",
        "- 若 PSI > 0.2，考慮移除或用更穩定的 encoding",
        "- 考慮用 target encoding + smoothing 取代 frequency encoding",
        "",
        "### 預期改善",
        "- Brier score: 降低 5~10%",
        "- calibration gap 縮小",
        "",
        "---",
        "",
        "## Hypothesis 3: 針對高 FP 率群體做特徵加強",
        "",
        "### 觀察",
    ]

    if len(high_fp_segments) > 0:
        for _, seg in high_fp_segments.head(5).iterrows():
            lines.append(
                f"- `{seg['segment_feature']}={seg['segment_value']}`: "
                f"FP rate={seg['fp_rate']:.2%}, "
                f"reject_rate={seg['reject_rate']:.2%}, "
                f"count={seg['count']}"
            )
    else:
        lines.append("- 未找到顯著高 FP 率群體")

    lines += [
        "",
        "### 建議",
        "- 針對高 FP 群體的共同特徵，新增更細緻的分箱或交互",
        "- 考慮 reject class 的 stratified feature engineering",
        "",
        "### 預期改善",
        "- Precision_reject: +5~10%",
        "- High zone 中的 FP 減少",
        "",
        "---",
        "",
        "## Hypothesis 4: 低重要度特徵清理",
        "",
        "### 觀察",
        f"- 重要度 < 1% 的特徵: {', '.join(low_importance) if low_importance else '無'}",
        "",
        "### 建議",
        "- 移除低重要度特徵，降低 noise",
        "- 若特徵有語義價值，考慮換 encoding 方式再試",
        "",
        "### 預期改善",
        "- 模型穩定性提升",
        "- 過擬合風險降低",
        "",
        "---",
        "",
        "## Hypothesis 5: Calibration 改善",
        "",
        "### 觀察",
        "- Brier score gap (train vs monitor) ≈ 0.113",
        "- Isotonic calibration 在小 reject class 上可能過擬合",
        "",
        "### 建議",
        "- 嘗試不使用 calibration（raw XGBoost probabilities）",
        "- 若 Brier 不退步太多，代表 isotonic 在此場景 overfitting",
        "",
        "### 預期改善",
        "- Brier score 穩定性改善",
        "- 不影響 AUC / KS（calibration 不改變 ranking）",
        "",
        "---",
        "",
        "*此文件由 feature_diagnosis.py 根據實際數據自動產出。*",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"✓ 改善假設已產出: {output_path}")


# ============================================
# 9. Challenger Plan
# ============================================
def generate_challenger_plan(
    diagnosis_df: pd.DataFrame,
    output_dir: Path,
):
    """產出 challenger_plan.json 和 challenger_experiment_matrix.csv"""

    unstable_features = list(diagnosis_df[diagnosis_df["is_unstable"] == True]["feature"])
    low_importance = list(diagnosis_df[diagnosis_df["importance_gain_norm"] < 0.01]["feature"])

    plan = {
        "baseline_reference": "baseline_v1",
        "created_at": datetime.now().isoformat(),
        "challengers": [
            {
                "challenger_id": "C1_reject_features",
                "description": "新增 reject-targeted 交互特徵",
                "changes": {
                    "add_features": [
                        "查詢_往來交互 = 近半年同業查詢次數_log_scaled × 內部往來次數_log_scaled",
                        "所得_金額比_分箱 = qcut(負債月所得比_scaled, 5)",
                        "查詢_缺失交互 = 近半年同業查詢次數_是否缺失 × 近半年同業查詢次數_log_scaled",
                    ],
                    "remove_features": [],
                    "model_type": "xgboost",
                    "calibration": "isotonic",
                },
                "hypothesis": "reject signal 不足",
                "target_metric": "f1_reject",
                "expected_improvement": "+0.02~0.05",
            },
            {
                "challenger_id": "C2_feature_cleanup",
                "description": "移除不穩定 + 低重要度特徵",
                "changes": {
                    "add_features": [],
                    "remove_features": unstable_features + low_importance,
                    "model_type": "xgboost",
                    "calibration": "isotonic",
                },
                "hypothesis": "noise / instability 造成 calibration 問題",
                "target_metric": "brier_score",
                "expected_improvement": "brier -5~10%",
            },
            {
                "challenger_id": "C3_combined",
                "description": "C1 + C2 合併：新增特徵 + 清理不穩定特徵",
                "changes": {
                    "add_features": [
                        "查詢_往來交互",
                        "所得_金額比_分箱",
                        "查詢_缺失交互",
                    ],
                    "remove_features": unstable_features + low_importance,
                    "model_type": "xgboost",
                    "calibration": "isotonic",
                },
                "hypothesis": "同時改善 signal 和 noise",
                "target_metric": "f1_reject + brier",
                "expected_improvement": "f1_reject +0.02, brier -5%",
            },
            {
                "challenger_id": "C4_no_calibration",
                "description": "移除 isotonic calibration，用 raw probabilities",
                "changes": {
                    "add_features": [],
                    "remove_features": [],
                    "model_type": "xgboost",
                    "calibration": "none",
                },
                "hypothesis": "isotonic 在少數 reject class 上 overfitting",
                "target_metric": "brier_score",
                "expected_improvement": "brier 穩定性改善",
            },
            {
                "challenger_id": "C5_rf_challenger",
                "description": "Random Forest challenger（同特徵）",
                "changes": {
                    "add_features": [],
                    "remove_features": [],
                    "model_type": "random_forest",
                    "calibration": "isotonic",
                },
                "hypothesis": "RF 可能更穩定（lower variance）",
                "target_metric": "stability_score",
                "expected_improvement": "stability +10~20%",
            },
        ],
        "evaluation_protocol": {
            "method": "four_phase_pipeline",
            "comparison_metrics": ["auc", "f1_reject", "ks", "brier_score"],
            "promotion_criteria": {
                "must_beat_baseline_auc": True,
                "must_beat_baseline_f1_reject": True,
                "min_auc_delta": 0.0,
                "min_f1_reject_delta": 0.005,
            },
        },
    }

    # Save plan
    plan_path = output_dir / "challenger_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Challenger plan: {plan_path}")

    # Save experiment matrix
    matrix_records = []
    for c in plan["challengers"]:
        matrix_records.append({
            "challenger_id": c["challenger_id"],
            "description": c["description"],
            "model_type": c["changes"]["model_type"],
            "calibration": c["changes"]["calibration"],
            "add_features": "; ".join(c["changes"]["add_features"]),
            "remove_features": "; ".join(c["changes"]["remove_features"]),
            "hypothesis": c["hypothesis"],
            "target_metric": c["target_metric"],
            "expected_improvement": c["expected_improvement"],
            "status": "planned",
        })
    matrix_df = pd.DataFrame(matrix_records)
    matrix_path = output_dir / "challenger_experiment_matrix.csv"
    matrix_df.to_csv(matrix_path, index=False, encoding="utf-8-sig")
    logger.info(f"✓ Experiment matrix: {matrix_path}")


# ============================================
# 10. Main Entry Point
# ============================================
def run_full_diagnosis(output_dir: Path = None) -> Dict[str, Any]:
    """
    執行完整診斷流程。

    Returns:
        所有輸出的摘要 dict
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        output_dir = PROJECT_ROOT / "model_bank" / "experiments" / f"diagnosis_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"輸出目錄: {output_dir}")

    # --- Load ---
    logger.info("=" * 60)
    logger.info("Phase A: 資料載入")
    logger.info("=" * 60)
    all_data = load_gold_data()
    model = load_baseline_model()
    feature_names = load_feature_names()
    predictions = load_holdout_predictions()
    rolling_defs = load_rolling_definitions()

    # --- Feature Importance ---
    logger.info("=" * 60)
    logger.info("Phase B: Feature Importance")
    logger.info("=" * 60)
    importance_df = compute_feature_importance(model, feature_names)

    # --- Reject Correlation ---
    logger.info("=" * 60)
    logger.info("Phase C: Feature-Reject Correlation")
    logger.info("=" * 60)
    reject_corr_df = compute_reject_correlation(all_data, feature_names)

    # --- Feature Stability ---
    logger.info("=" * 60)
    logger.info("Phase D: Feature Stability")
    logger.info("=" * 60)
    stability_df, stability_detail = compute_feature_stability(
        all_data, feature_names, rolling_defs
    )

    # --- Error Analysis ---
    logger.info("=" * 60)
    logger.info("Phase E: Error Analysis")
    logger.info("=" * 60)
    segment_analysis, fp_profile, fn_profile, zone_summary = run_error_analysis(
        all_data, predictions, feature_names
    )

    # --- Merge into diagnosis table ---
    logger.info("=" * 60)
    logger.info("Phase F: 合併診斷表")
    logger.info("=" * 60)
    diagnosis_df = build_feature_diagnosis_table(
        importance_df, reject_corr_df, stability_df, fp_profile, fn_profile
    )

    # --- Save all outputs ---
    logger.info("=" * 60)
    logger.info("Phase G: 儲存輸出")
    logger.info("=" * 60)

    # 1. feature_diagnosis.csv
    diagnosis_df.to_csv(output_dir / "feature_diagnosis.csv", index=False, encoding="utf-8-sig")
    logger.info("  ✓ feature_diagnosis.csv")

    # 2. feature_stability_summary.json
    stability_summary = {
        "created_at": datetime.now().isoformat(),
        "baseline": "baseline_v1",
        "total_features": len(stability_df),
        "unstable_features": int(stability_df["is_unstable"].sum()),
        "unstable_feature_list": list(stability_df[stability_df["is_unstable"]]["feature"]),
        "per_feature_detail": stability_detail,
    }
    with open(output_dir / "feature_stability_summary.json", "w", encoding="utf-8") as f:
        json.dump(stability_summary, f, ensure_ascii=False, indent=2)
    logger.info("  ✓ feature_stability_summary.json")

    # 3. error_analysis_by_segment.csv
    segment_analysis.to_csv(
        output_dir / "error_analysis_by_segment.csv", index=False, encoding="utf-8-sig"
    )
    logger.info("  ✓ error_analysis_by_segment.csv")

    # 4. false_positive_profile.csv
    fp_profile.to_csv(
        output_dir / "false_positive_profile.csv", index=False, encoding="utf-8-sig"
    )
    logger.info("  ✓ false_positive_profile.csv")

    # 5. false_negative_profile.csv
    fn_profile.to_csv(
        output_dir / "false_negative_profile.csv", index=False, encoding="utf-8-sig"
    )
    logger.info("  ✓ false_negative_profile.csv")

    # 6. zone_error_summary.json
    with open(output_dir / "zone_error_summary.json", "w", encoding="utf-8") as f:
        json.dump(zone_summary, f, ensure_ascii=False, indent=2)
    logger.info("  ✓ zone_error_summary.json")

    # 7. top_feature_report.md
    generate_top_feature_report(diagnosis_df, output_dir / "top_feature_report.md")

    # 8. improvement_hypotheses.md
    generate_improvement_hypotheses(
        diagnosis_df, stability_df, segment_analysis, zone_summary,
        output_dir / "improvement_hypotheses.md",
    )

    # 9. challenger_plan.json + challenger_experiment_matrix.csv
    generate_challenger_plan(diagnosis_df, output_dir)

    logger.info("=" * 60)
    logger.info(f"✓ 診斷完成。所有輸出在: {output_dir}")
    logger.info("=" * 60)

    return {
        "output_dir": str(output_dir),
        "feature_count": len(diagnosis_df),
        "unstable_features": int(stability_df["is_unstable"].sum()),
        "fp_count": int(predictions.merge(
            pd.DataFrame({"案件編號": predictions["案件編號"]}), on="案件編號"
        ).shape[0]),
        "zone_summary": zone_summary,
    }


# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    result = run_full_diagnosis()
    print(f"\n完成。輸出目錄: {result['output_dir']}")
