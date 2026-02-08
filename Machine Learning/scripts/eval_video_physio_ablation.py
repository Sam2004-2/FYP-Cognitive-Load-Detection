#!/usr/bin/env python3
"""
Evaluate how well video window features predict a physio-derived target.

This script runs a small ablation:
  1) base window features
  2) base + baseline-centered
  3) base + centered + delta

It reports:
  - window-level Spearman (OOF) vs chosen target
  - session-level Spearman (OOF, mean over windows per user/task) vs chosen target
  - secondary binary-stress AUC/accuracy vs labels.csv (session-level)
  - per-task and per-user breakdowns (session/window Spearman)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Ensure project root is on sys.path so `import src.*` works when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cle.data.alignment import overlap_weighted_label_video_windows  # noqa: E402
from src.cle.train.feature_engineering import (  # noqa: E402
    DEFAULT_BASELINE_TASKS,
    add_centered_and_delta,
    compute_user_baseline,
)


BASE_VIDEO_FEATURES: List[str] = [
    "blink_rate",
    "blink_count",
    "mean_blink_duration",
    "ear_std",
    "perclos",
    "mouth_open_mean",
    "mouth_open_std",
    "roll_std",
    "pitch_std",
    "yaw_std",
    "motion_mean",
    "motion_std",
]


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 3:
        return None
    r, _ = spearmanr(y_true, y_pred)
    if r is None or np.isnan(r):
        return None
    return float(r)


def load_binary_stress_labels(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    df[["user_id", "task"]] = df["subject/task"].str.split("_", n=1, expand=True)
    return df[["user_id", "task", "binary-stress"]].rename(columns={"binary-stress": "binary_stress"})


def _tuned_threshold_accuracy_cv(
    session_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    group_col: str = "user_id",
    y_col: str = "y_pred",
    label_col: str = "binary_stress",
    n_splits: int = 5,
) -> float | None:
    """
    Compute fold-mean session accuracy with threshold tuned on train sessions.

    Group split is by user_id (same as the main model CV).
    """
    merged = session_df.merge(labels_df, on=["user_id", "task"], how="inner").dropna(
        subset=[label_col, y_col]
    )
    if len(merged) == 0:
        return None

    groups = merged[group_col].astype(str).values
    y_pred = merged[y_col].values.astype(float)
    y_true = merged[label_col].values.astype(int)

    cv = GroupKFold(n_splits=n_splits)
    accs: List[float] = []
    for train_idx, test_idx in cv.split(y_pred.reshape(-1, 1), y_true, groups):
        train_pred = y_pred[train_idx]
        train_true = y_true[train_idx]
        if len(np.unique(train_true)) < 2:
            continue

        # Candidate thresholds: unique predicted values
        cand = np.unique(train_pred)
        best_th = 0.5
        best_acc = -1.0
        for th in cand:
            acc = accuracy_score(train_true, (train_pred >= th).astype(int))
            if acc > best_acc:
                best_acc = float(acc)
                best_th = float(th)

        test_pred = y_pred[test_idx]
        test_true = y_true[test_idx]
        accs.append(float(accuracy_score(test_true, (test_pred >= best_th).astype(int))))

    return float(np.mean(accs)) if accs else None


@dataclass(frozen=True)
class EvalResult:
    feature_set: str
    model_name: str
    feature_cols: List[str]
    folds: List[Dict[str, Any]]
    overall: Dict[str, Any]
    per_task: List[Dict[str, Any]]
    per_user: List[Dict[str, Any]]


def evaluate_cv(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    labels_df: pd.DataFrame,
    *,
    model_name: str,
    model_factory: Callable[[int], Any],
    n_splits: int,
    seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    X = df[list(feature_cols)].values.astype(float)
    y = df["y_true"].values.astype(float)
    groups = df["user_id"].astype(str).values

    cv = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y), dtype=float)
    fold_rows: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X[train_idx])
        X_test = imputer.transform(X[test_idx])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = model_factory(seed)
        model.fit(X_train, y[train_idx])

        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0.0, 1.0)
        oof[test_idx] = y_pred

        fold_rows.append(
            {
                "fold": fold_idx + 1,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "window_spearman": compute_spearman(y[test_idx], y_pred),
            }
        )

    window_pred_df = df[["user_id", "task", "t_start_s", "physio_stress_score", "y_true"]].copy()
    window_pred_df["y_pred"] = oof

    session_df = (
        window_pred_df.groupby(["user_id", "task"], as_index=False)
        .agg(
            y_true=("y_true", "mean"),
            y_pred=("y_pred", "mean"),
            n_windows=("y_pred", "count"),
        )
        .copy()
    )

    window_spearman = compute_spearman(window_pred_df["y_true"].values, window_pred_df["y_pred"].values)
    session_spearman = compute_spearman(session_df["y_true"].values, session_df["y_pred"].values)

    secondary_auc = None
    secondary_acc_05 = None
    merged = session_df.merge(labels_df, on=["user_id", "task"], how="inner").dropna(
        subset=["binary_stress", "y_pred"]
    )
    if len(merged) > 0 and merged["binary_stress"].nunique() > 1:
        secondary_auc = float(roc_auc_score(merged["binary_stress"].values, merged["y_pred"].values))
        secondary_acc_05 = float(
            accuracy_score(merged["binary_stress"].values, (merged["y_pred"].values >= 0.5).astype(int))
        )

    tuned_acc = _tuned_threshold_accuracy_cv(session_df, labels_df, n_splits=n_splits)

    overall = {
        "window_spearman": window_spearman,
        "session_spearman": session_spearman,
        "n_windows": int(len(window_pred_df)),
        "n_sessions": int(len(session_df)),
        "secondary_binary_stress_auc": secondary_auc,
        "secondary_binary_stress_accuracy@0.5": secondary_acc_05,
        "secondary_binary_stress_accuracy_tuned": tuned_acc,
    }

    return overall, window_pred_df, session_df, fold_rows


def per_task_breakdown(session_df: pd.DataFrame, labels_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task, g in session_df.groupby("task", sort=True):
        r = compute_spearman(g["y_true"].values, g["y_pred"].values)
        merged = g.merge(labels_df, on=["user_id", "task"], how="inner").dropna(subset=["binary_stress", "y_pred"])
        auc = None
        if len(merged) > 0 and merged["binary_stress"].nunique() > 1:
            auc = float(roc_auc_score(merged["binary_stress"].values, merged["y_pred"].values))
        rows.append({"task": task, "n_sessions": int(len(g)), "session_spearman": r, "binary_auc": auc})
    return rows


def per_user_breakdown(window_pred_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for user_id, g in window_pred_df.groupby("user_id", sort=True):
        r = compute_spearman(g["y_true"].values, g["y_pred"].values)
        rows.append({"user_id": user_id, "n_windows": int(len(g)), "window_spearman": r})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablation eval: video -> physio target")
    parser.add_argument("--video-features", type=str, default="data/processed/stress_features_10s.csv")
    parser.add_argument("--physio-labels", type=str, default="data/processed/physio_stress_labels.csv")
    parser.add_argument("--binary-labels", type=str, default="../labels.csv")
    parser.add_argument("--report", type=str, default="reports/video_physio_ablation_eval.json")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-n-windows", type=int, default=4)
    parser.add_argument(
        "--target",
        type=str,
        default="z01",
        choices=["abs", "delta", "z01"],
        help="Target transform: abs=raw physio_stress_score, delta=baseline-subtracted mapped to 0-1, z01=per-user z-score mapped via sigmoid (recommended).",
    )
    args = parser.parse_args()

    video_path = Path(args.video_features)
    physio_path = Path(args.physio_labels)
    labels_path = Path(args.binary_labels)

    video_df = pd.read_csv(video_path)
    physio_df = pd.read_csv(physio_path)
    labels_df = load_binary_stress_labels(labels_path)
    labels_df["user_id"] = labels_df["user_id"].astype(str)
    labels_df["task"] = labels_df["task"].astype(str)

    labeled = overlap_weighted_label_video_windows(video_df, physio_df)
    labeled = labeled.dropna(subset=["physio_stress_score"]).copy()
    if len(labeled) == 0:
        raise ValueError("No labeled rows after overlap labeling.")

    # Target transform
    score = labeled["physio_stress_score"].values.astype(float)
    if args.target == "abs":
        labeled["y_true"] = score
        target_name = "physio_stress_score"
    elif args.target == "delta":
        score_baseline_df = compute_user_baseline(
            labeled,
            feature_cols=["physio_stress_score"],
            baseline_tasks=DEFAULT_BASELINE_TASKS,
            n_windows=args.baseline_n_windows,
            time_col="t_start_s",
        )
        labeled = labeled.merge(
            score_baseline_df[["user_id", "baseline_physio_stress_score"]],
            on="user_id",
            how="left",
        )
        delta = score - labeled["baseline_physio_stress_score"].values.astype(float)
        labeled["y_true"] = np.clip(0.5 + 0.5 * delta, 0.0, 1.0)
        target_name = "physio_stress_delta01"
    else:
        mu = labeled.groupby("user_id")["physio_stress_score"].transform("mean").values.astype(float)
        sigma = (
            labeled.groupby("user_id")["physio_stress_score"]
            .transform(lambda s: s.std(ddof=0))
            .values.astype(float)
        )
        z = np.zeros_like(score, dtype=float)
        mask = sigma >= 1e-6
        z[mask] = (score[mask] - mu[mask]) / sigma[mask]
        labeled["y_true"] = 1.0 / (1.0 + np.exp(-z))
        target_name = "physio_stress_z01"

    missing_feats = [c for c in BASE_VIDEO_FEATURES if c not in labeled.columns]
    if missing_feats:
        raise ValueError(f"Missing base features in video features CSV: {missing_feats}")

    baseline_df = compute_user_baseline(
        labeled,
        feature_cols=BASE_VIDEO_FEATURES,
        baseline_tasks=DEFAULT_BASELINE_TASKS,
        n_windows=args.baseline_n_windows,
        time_col="t_start_s",
    )
    engineered, _ = add_centered_and_delta(
        labeled,
        feature_cols=BASE_VIDEO_FEATURES,
        baseline_df=baseline_df,
        session_cols=("user_id", "task"),
        time_col="t_start_s",
    )

    feature_sets: Dict[str, List[str]] = {
        "base_14": list(BASE_VIDEO_FEATURES),
        "base_plus_centered_28": [*BASE_VIDEO_FEATURES, *[f"{c}_centered" for c in BASE_VIDEO_FEATURES]],
        "base_centered_delta_42": [
            *BASE_VIDEO_FEATURES,
            *[f"{c}_centered" for c in BASE_VIDEO_FEATURES],
            *[f"{c}_delta" for c in BASE_VIDEO_FEATURES],
        ],
    }

    models: Dict[str, Callable[[int], Any]] = {
        "Ridge": lambda seed: Ridge(alpha=1.0),
        "GradientBoosting": lambda seed: GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=10,
            random_state=seed,
        ),
    }

    results: List[EvalResult] = []
    for fs_name, fs_cols in feature_sets.items():
        for model_name, factory in models.items():
            overall, window_pred_df, session_df, folds = evaluate_cv(
                engineered,
                fs_cols,
                labels_df,
                model_name=model_name,
                model_factory=factory,
                n_splits=args.n_splits,
                seed=args.seed,
            )
            results.append(
                EvalResult(
                    feature_set=fs_name,
                    model_name=model_name,
                    feature_cols=list(fs_cols),
                    folds=folds,
                    overall=overall,
                    per_task=per_task_breakdown(session_df, labels_df),
                    per_user=per_user_breakdown(window_pred_df),
                )
            )

    report = {
        "timestamp": datetime.now().isoformat(),
        "target": args.target,
        "target_name": target_name,
        "video_features": str(video_path),
        "physio_labels": str(physio_path),
        "binary_labels": str(labels_path),
        "baseline_tasks": sorted(DEFAULT_BASELINE_TASKS),
        "baseline_n_windows": args.baseline_n_windows,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "results": [
            {
                "feature_set": r.feature_set,
                "model": r.model_name,
                "n_features": len(r.feature_cols),
                "overall": r.overall,
                "folds": r.folds,
                "per_task": r.per_task,
                "per_user": r.per_user,
            }
            for r in results
        ],
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    # Print a compact summary to stdout
    print("Ablation summary (OOF):")
    for r in results:
        o = r.overall
        print(
            f"- {r.feature_set} / {r.model_name}: "
            f"session_spearman={o['session_spearman']}, "
            f"window_spearman={o['window_spearman']}, "
            f"auc={o['secondary_binary_stress_auc']}, "
            f"acc_tuned={o['secondary_binary_stress_accuracy_tuned']}"
        )
    print(f"\nSaved full report to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
