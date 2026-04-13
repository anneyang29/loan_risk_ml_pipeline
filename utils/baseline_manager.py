"""
Baseline Manager — 基線模型管理工具

功能：
1. 從指定 run 打包成正式 baseline
2. 查詢/比較 baseline
3. 切換 active baseline
4. 產出 baseline 差異報告

使用方式：
    # 打包新 baseline
    python utils/baseline_manager.py --create --run-id 20260412_235140 --name baseline_v2

    # 列出所有 baselines
    python utils/baseline_manager.py --list

    # 比較兩個 baselines
    python utils/baseline_manager.py --compare baseline_v1 baseline_v2

    # 設定 active baseline
    python utils/baseline_manager.py --activate baseline_v2
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ============================================
# 檔案分類規則
# ============================================
FILE_ROUTING = {
    "model": ["final_champion_model.pkl"],
    "artifacts": ["feature_names.json"],
    "metrics": [
        "final_holdout_metrics.json",
        "diagnostics_summary.json",
        "rolling_results.csv",
        "tuning_comparison.csv",
    ],
    "policy": [
        "threshold_policy_comparison.csv",
        "zone_policy_summary.json",
        "zone_summary.csv",
        "policy_validation_predictions.csv",
    ],
    "predictions": ["final_holdout_predictions.csv"],
    "metadata": ["champion_summary.json"],
}


@dataclass
class BaselineRecord:
    """Baseline 紀錄"""
    baseline_name: str
    baseline_version: str
    created_at: str
    status: str  # active | archived | deprecated
    source_run_id: str
    source_run_folder: str

    # 核心指標（快速比較用）
    holdout_auc: float = 0.0
    holdout_f1_reject: float = 0.0
    holdout_ks: float = 0.0
    holdout_brier: float = 0.0
    lower_threshold: float = 0.0
    upper_threshold: float = 0.0
    feature_count: int = 0

    # 診斷
    overfitting_severity: str = ""
    passes_hard_constraints: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "BaselineRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class BaselineManager:
    """
    Baseline 管理器

    目錄結構：
        model_bank/
        ├── baselines/
        │   ├── active_baseline.json    <- 目前使用的 baseline 指標
        │   ├── baseline_v1/
        │   │   ├── baseline_metadata.json
        │   │   ├── model/
        │   │   ├── artifacts/
        │   │   ├── metrics/
        │   │   ├── policy/
        │   │   ├── predictions/
        │   │   └── metadata/
        │   └── baseline_v2/
        │       └── ...
        └── four_phase_XXXXXXXX_XXXXXX/  <- 原始 run
    """

    SUBDIRS = ["model", "artifacts", "metrics", "policy", "predictions", "metadata"]

    def __init__(self, model_bank_path: str = "model_bank"):
        self.model_bank_path = Path(model_bank_path)
        self.baselines_path = self.model_bank_path / "baselines"
        self.active_file = self.baselines_path / "active_baseline.json"
        self.baselines_path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 建立 Baseline
    # --------------------------------------------------
    def create_baseline(
        self,
        run_id: str,
        baseline_name: str,
        description: str = "",
        auto_activate: bool = True,
    ) -> Path:
        """
        從指定 run 打包成正式 baseline。

        Args:
            run_id: run 資料夾名稱 (e.g., "20260412_235140")
            baseline_name: baseline 名稱 (e.g., "baseline_v1")
            description: 自由文字說明
            auto_activate: 是否自動設為 active

        Returns:
            baseline 資料夾 Path
        """
        # 找到 run 資料夾
        run_folder = self._find_run_folder(run_id)
        if run_folder is None:
            raise FileNotFoundError(
                f"找不到 run: {run_id}，"
                f"搜尋路徑: {self.model_bank_path}"
            )

        baseline_dir = self.baselines_path / baseline_name
        if baseline_dir.exists():
            raise FileExistsError(f"Baseline 已存在: {baseline_name}")

        # 建立子目錄
        for subdir in self.SUBDIRS:
            (baseline_dir / subdir).mkdir(parents=True, exist_ok=True)

        # 複製檔案
        copied = self._copy_run_files(run_folder, baseline_dir)
        logger.info(f"已複製 {len(copied)} 個檔案到 {baseline_dir}")

        # 讀取 run metadata 產出 baseline_metadata.json
        metadata = self._build_metadata(
            run_folder=run_folder,
            baseline_name=baseline_name,
            description=description,
        )
        metadata_path = baseline_dir / "baseline_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # 產出 baseline_summary.md
        self._generate_summary_md(metadata, baseline_dir)

        logger.info(f"OK: Baseline 建立完成: {baseline_name}")

        if auto_activate:
            self.activate_baseline(baseline_name)

        return baseline_dir

    # --------------------------------------------------
    # 啟用 / 列出 / 查詢
    # --------------------------------------------------
    def activate_baseline(self, baseline_name: str):
        """設定 active baseline"""
        baseline_dir = self.baselines_path / baseline_name
        if not baseline_dir.exists():
            raise FileNotFoundError(f"Baseline 不存在: {baseline_name}")

        metadata_path = baseline_dir / "baseline_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"缺少 metadata: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        holdout = metadata.get("holdout_metrics", {})
        thresholds = metadata.get("threshold_policy", {})

        active = {
            "active_baseline": baseline_name,
            "baseline_path": f"baselines/{baseline_name}",
            "model_version": f"credit_model_{metadata['source_run']['run_id']}",
            "source_run_id": metadata["source_run"]["run_id"],
            "activated_at": datetime.now().isoformat(),
            "holdout_auc": holdout.get("auc", 0),
            "holdout_f1_reject": holdout.get("f1_reject", 0),
            "holdout_ks": holdout.get("ks", 0),
            "thresholds": {
                "lower": thresholds.get("lower_threshold", 0),
                "upper": thresholds.get("upper_threshold", 0),
            },
        }

        with open(self.active_file, "w", encoding="utf-8") as f:
            json.dump(active, f, ensure_ascii=False, indent=2)

        logger.info(f"OK: Active baseline 已切換: {baseline_name}")

    def get_active_baseline(self) -> Optional[Dict]:
        """取得目前 active baseline"""
        if not self.active_file.exists():
            return None
        with open(self.active_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_baselines(self) -> List[str]:
        """列出所有 baseline 名稱"""
        baselines = []
        for d in sorted(self.baselines_path.iterdir()):
            if d.is_dir() and (d / "baseline_metadata.json").exists():
                baselines.append(d.name)
        return baselines

    def get_baseline_metadata(self, baseline_name: str) -> Dict:
        """取得 baseline metadata"""
        path = self.baselines_path / baseline_name / "baseline_metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"Baseline 不存在: {baseline_name}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # --------------------------------------------------
    # 比較
    # --------------------------------------------------
    def compare_baselines(
        self, name_a: str, name_b: str
    ) -> Dict[str, Any]:
        """
        比較兩個 baseline 的指標差異。

        Returns:
            Dict 含 summary table 和 per-metric delta
        """
        meta_a = self.get_baseline_metadata(name_a)
        meta_b = self.get_baseline_metadata(name_b)

        holdout_a = meta_a.get("holdout_metrics", {})
        holdout_b = meta_b.get("holdout_metrics", {})

        compare_keys = ["auc", "f1_reject", "ks", "brier_score", "precision_reject", "recall_reject"]
        deltas = {}
        for k in compare_keys:
            va = holdout_a.get(k, 0)
            vb = holdout_b.get(k, 0)
            deltas[k] = {
                name_a: va,
                name_b: vb,
                "delta": round(vb - va, 6),
                "improved": vb > va if k != "brier_score" else vb < va,
            }

        # Threshold 比較
        thresh_a = meta_a.get("threshold_policy", {})
        thresh_b = meta_b.get("threshold_policy", {})
        threshold_change = {
            "lower": {name_a: thresh_a.get("lower_threshold"), name_b: thresh_b.get("lower_threshold")},
            "upper": {name_a: thresh_a.get("upper_threshold"), name_b: thresh_b.get("upper_threshold")},
        }

        result = {
            "baselines": [name_a, name_b],
            "holdout_deltas": deltas,
            "threshold_change": threshold_change,
            "diagnostics_a": meta_a.get("diagnostics", {}),
            "diagnostics_b": meta_b.get("diagnostics", {}),
        }

        return result

    def print_comparison(self, name_a: str, name_b: str):
        """印出格式化的比較結果"""
        comp = self.compare_baselines(name_a, name_b)

        lines = [
            "=" * 80,
            f"Baseline Comparison: {name_a} vs {name_b}",
            "=" * 80,
            f"{'Metric':<20} {name_a:>15} {name_b:>15} {'Delta':>10} {'Status':>10}",
            "-" * 80,
        ]

        for metric, info in comp["holdout_deltas"].items():
            va = info[name_a]
            vb = info[name_b]
            delta = info["delta"]
            status = "OK: 改善" if info["improved"] else "FAIL: 退步"
            if abs(delta) < 1e-6:
                status = "— 相同"
            lines.append(
                f"{metric:<20} {va:>15.4f} {vb:>15.4f} {delta:>+10.4f} {status:>10}"
            )

        lines.append("-" * 80)

        tc = comp["threshold_change"]
        lines.append(
            f"{'Lower Threshold':<20} {tc['lower'][name_a]:>15} {tc['lower'][name_b]:>15}"
        )
        lines.append(
            f"{'Upper Threshold':<20} {tc['upper'][name_a]:>15} {tc['upper'][name_b]:>15}"
        )
        lines.append("=" * 80)

        print("\n".join(lines))

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _find_run_folder(self, run_id: str) -> Optional[Path]:
        """搜尋 model_bank 下符合 run_id 的資料夾"""
        # 嘗試直接匹配
        for d in self.model_bank_path.iterdir():
            if d.is_dir() and run_id in d.name:
                return d
        return None

    def _copy_run_files(self, run_folder: Path, baseline_dir: Path) -> List[str]:
        """依據 FILE_ROUTING 複製檔案"""
        copied = []
        for subdir, filenames in FILE_ROUTING.items():
            dest_dir = baseline_dir / subdir
            for fname in filenames:
                src = run_folder / fname
                if src.exists():
                    shutil.copy2(src, dest_dir / fname)
                    copied.append(f"{subdir}/{fname}")
                else:
                    logger.warning(f"來源檔案不存在，跳過: {src}")
        return copied

    def _build_metadata(
        self,
        run_folder: Path,
        baseline_name: str,
        description: str,
    ) -> Dict:
        """從 run 檔案建構 baseline_metadata.json"""

        # 讀取各來源檔
        champion = self._read_json(run_folder / "champion_summary.json") or {}
        holdout = self._read_json(run_folder / "final_holdout_metrics.json") or {}
        diagnostics = self._read_json(run_folder / "diagnostics_summary.json") or {}
        zone_policy = self._read_json(run_folder / "zone_policy_summary.json") or {}
        features = self._read_json(run_folder / "feature_names.json") or []

        strategy = champion.get("champion_strategy", {})
        tuning = champion.get("tuning_config", {})
        threshold_cfg = champion.get("threshold_config", {})

        metadata = {
            "baseline_name": baseline_name,
            "baseline_version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "description": description or f"Baseline from run {run_folder.name}",

            "source_run": {
                "run_id": champion.get("run_id", run_folder.name),
                "run_folder": run_folder.name,
                "pipeline": "FourPhaseTrainer",
                "created_at": champion.get("created_at", ""),
            },

            "champion_model": {
                "model_type": strategy.get("model_name", "unknown"),
                "imbalance_strategy": strategy.get("imbalance_strategy", ""),
                "calibration_method": tuning.get("calibration_method", "isotonic"),
                "feature_count": len(features),
                "model_file": "model/final_champion_model.pkl",
                "feature_file": "artifacts/feature_names.json",
            },

            "tuning": {
                "config_id": tuning.get("config_id", ""),
                "tuning_score": round(tuning.get("tuning_score", 0), 4),
                "hyperparameters": tuning.get("params", {}),
                "why_selected": tuning.get("why_selected", ""),
            },

            "threshold_policy": {
                "lower_threshold": threshold_cfg.get("lower_threshold", zone_policy.get("selected_lower_threshold", 0)),
                "upper_threshold": threshold_cfg.get("upper_threshold", zone_policy.get("selected_upper_threshold", 0)),
                "source": threshold_cfg.get("source", zone_policy.get("threshold_source", "")),
                "threshold_score": round(zone_policy.get("threshold_score", 0), 4),
                "passes_hard_constraints": bool(zone_policy.get("passes_hard_constraints", False)),
            },

            "holdout_metrics": {
                "auc": round(holdout.get("auc", 0), 4),
                "auc_pr": round(holdout.get("auc_pr", 0), 4),
                "f1": round(holdout.get("f1", 0), 4),
                "precision": round(holdout.get("precision", 0), 4),
                "recall": round(holdout.get("recall", 0), 4),
                "f1_reject": round(holdout.get("f1_reject", 0), 4),
                "precision_reject": round(holdout.get("precision_reject", 0), 4),
                "recall_reject": round(holdout.get("recall_reject", 0), 4),
                "ks": round(holdout.get("ks", 0), 4),
                "brier_score": round(holdout.get("brier_score", 0), 4),
                "positive_ratio": round(holdout.get("positive_ratio", 0), 4),
            },

            "rolling_cv_metrics": self._extract_rolling_metrics(champion),

            "business_constraints": {
                "auto_decision_rate": round(zone_policy.get("auto_decision_rate", 0), 4),
                "manual_review_load": round(zone_policy.get("manual_review_load", 0), 4),
                "high_zone_precision": round(zone_policy.get("high_zone_precision", 0), 4),
                "low_zone_precision": round(zone_policy.get("low_zone_precision", 0), 4),
                "constraint_status": "all_passed" if zone_policy.get("passes_hard_constraints") else "failed",
            },

            "diagnostics": {
                "is_overfitting": bool(diagnostics.get("is_overfitting", False)),
                "overfitting_severity": diagnostics.get("overfitting_severity", ""),
                "has_calibration_issue": bool(diagnostics.get("has_calibration_issue", False)),
                "has_reject_detection_issue": bool(diagnostics.get("has_reject_detection_issue", False)),
                "gap_train_vs_holdout_auc": round(diagnostics.get("gap_train_vs_holdout_auc", 0), 4),
                "gap_train_vs_monitor_brier": round(diagnostics.get("gap_train_vs_monitor_brier", 0), 4),
            },

            "champion_selection_weights": champion.get("champion_selection_config", {}).get("weights", {}),

            "files": {
                subdir: fnames for subdir, fnames in FILE_ROUTING.items()
            },
        }

        return metadata

    def _extract_rolling_metrics(self, champion: Dict) -> Dict:
        """從 champion_summary 提取 champion strategy 的 rolling metrics"""
        strategy_name = champion.get("champion_strategy", {}).get("model_name", "")
        results = champion.get("strategy_results", {}).get(strategy_name, {})
        if not results:
            return {}
        return {
            "avg_cv_auc": round(results.get("avg_cv_auc", 0), 4),
            "avg_monitor_auc": round(results.get("avg_monitor_auc", 0), 4),
            "avg_monitor_f1_reject": round(results.get("avg_monitor_f1_reject", 0), 4),
            "avg_monitor_ks": round(results.get("avg_monitor_ks", 0), 4),
            "stability_score": round(results.get("stability_score", 0), 4),
            "overall_score": round(results.get("overall_score", 0), 4),
        }

    def _generate_summary_md(self, metadata: Dict, baseline_dir: Path):
        """產出 baseline_summary.md"""
        m = metadata
        h = m.get("holdout_metrics", {})
        t = m.get("threshold_policy", {})
        d = m.get("diagnostics", {})
        bc = m.get("business_constraints", {})
        tuning = m.get("tuning", {})
        params = tuning.get("hyperparameters", {})

        lines = [
            f"# {m['baseline_name']} — 信用風險模型基線報告",
            "",
            f"> **建立日期**: {m['created_at'][:10]}  ",
            f"> **來源 Run**: `{m['source_run']['run_folder']}`  ",
            f"> **狀態**: OK Active Baseline  ",
            "",
            "---",
            "",
            "## 1. 模型概要",
            "",
            "| 項目 | 值 |",
            "|------|-----|",
            f"| 模型類型 | {m['champion_model']['model_type']} + {m['champion_model']['calibration_method']} |",
            f"| 不平衡策略 | {m['champion_model']['imbalance_strategy']} |",
            f"| 特徵數量 | {m['champion_model']['feature_count']} |",
            f"| Tuning Config | `{tuning.get('config_id', '')}` |",
            f"| Tuning Score | {tuning.get('tuning_score', 0)} |",
            "",
            "## 2. 超參數",
            "",
            "| 參數 | 值 |",
            "|------|-----|",
        ]
        for k, v in params.items():
            lines.append(f"| {k} | {v} |")

        lines += [
            "",
            "## 3. Holdout 測試指標",
            "",
            "| 指標 | 值 |",
            "|------|-----|",
            f"| AUC | **{h.get('auc', 0)}** |",
            f"| F1 (Reject) | **{h.get('f1_reject', 0)}** |",
            f"| KS | **{h.get('ks', 0)}** |",
            f"| Brier Score | **{h.get('brier_score', 0)}** |",
            f"| Precision (Reject) | {h.get('precision_reject', 0)} |",
            f"| Recall (Reject) | {h.get('recall_reject', 0)} |",
            "",
            "## 4. Threshold 政策",
            "",
            "| 項目 | 值 |",
            "|------|-----|",
            f"| Lower Threshold | **{t.get('lower_threshold', 0)}** |",
            f"| Upper Threshold | **{t.get('upper_threshold', 0)}** |",
            f"| 通過硬約束 | {'OK' if t.get('passes_hard_constraints') else 'FAIL'} |",
            "",
            "## 5. 營運指標",
            "",
            "| 項目 | 值 | 狀態 |",
            "|------|-----|------|",
            f"| 自動決策率 | {bc.get('auto_decision_rate', 0):.2%} | {'OK' if bc.get('auto_decision_rate', 0) >= 0.85 else 'FAIL'} |",
            f"| 人工審核率 | {bc.get('manual_review_load', 0):.2%} | {'OK' if bc.get('manual_review_load', 0) <= 0.15 else 'FAIL'} |",
            "",
            "## 6. 診斷",
            "",
            f"- 過擬合: {'WARNING: ' + d.get('overfitting_severity', '') if d.get('is_overfitting') else 'OK 無'}",
            f"- 校準問題: {'WARNING: 有' if d.get('has_calibration_issue') else 'OK 無'}",
            f"- 拒絕偵測問題: {'WARNING: 有' if d.get('has_reject_detection_issue') else 'OK 無'}",
            "",
            "---",
            "",
            "*此文件由 BaselineManager 自動產生。*",
        ]

        md_path = baseline_dir / "metadata" / "baseline_summary.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _read_json(path: Path) -> Any:
        """安全讀取 JSON"""
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    import argparse
    import sys

    # 抑制第三方套件的 INFO 日誌（如 NumExpr）
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for _logger_name in ["numexpr", "numexpr.utils"]:
        logging.getLogger(_logger_name).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Baseline Manager CLI")
    parser.add_argument("--model-bank", type=str, default="model_bank", help="Model bank 路徑")

    # 動作
    parser.add_argument("--create", action="store_true", help="建立新 baseline")
    parser.add_argument("--run-id", type=str, help="來源 run ID (用於 --create)")
    parser.add_argument("--name", type=str, help="Baseline 名稱 (用於 --create)")
    parser.add_argument("--description", type=str, default="", help="Baseline 說明")

    parser.add_argument("--list", action="store_true", help="列出所有 baselines")
    parser.add_argument("--info", type=str, help="顯示指定 baseline 的資訊")
    parser.add_argument("--activate", type=str, help="設定 active baseline")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="比較兩個 baselines")
    parser.add_argument("--active", action="store_true", help="顯示目前 active baseline")

    args = parser.parse_args()
    mgr = BaselineManager(args.model_bank)

    if args.create:
        if not args.run_id or not args.name:
            parser.error("--create 需要 --run-id 和 --name")
        baseline_dir = mgr.create_baseline(
            run_id=args.run_id,
            baseline_name=args.name,
            description=args.description,
        )
        print(f"OK: Baseline 已建立: {baseline_dir}")

    if args.list:
        active_info = mgr.get_active_baseline()
        active_name = active_info["active_baseline"] if active_info else None
        print("Baselines:")
        for name in mgr.list_baselines():
            marker = " <- ACTIVE" if name == active_name else ""
            meta = mgr.get_baseline_metadata(name)
            auc = meta.get("holdout_metrics", {}).get("auc", 0)
            run = meta.get("source_run", {}).get("run_id", "")
            print(f"  {name}  (AUC={auc:.4f}, run={run}){marker}")

    if args.info:
        meta = mgr.get_baseline_metadata(args.info)
        print(json.dumps(meta, ensure_ascii=False, indent=2))

    if args.activate:
        mgr.activate_baseline(args.activate)
        print(f"OK: Active baseline: {args.activate}")

    if args.compare:
        mgr.print_comparison(args.compare[0], args.compare[1])

    if args.active:
        active = mgr.get_active_baseline()
        if active:
            print(json.dumps(active, ensure_ascii=False, indent=2))
        else:
            print("尚未設定 active baseline")
