#!/usr/bin/env python3
"""
Train a video regression model to predict a physio-derived target.

Pipeline:
  1) Load video window features (should be 10s/2.5s for alignment)
  2) Load physio_stress_labels.csv (10s/2.5s)
  3) Label each video window with physio_stress_score (exact join or overlap-weighted)
  4) Optionally transform the target per-user (abs / delta01 / z01)
  5) Add baseline-centered and delta features (3N total)
  5) Train/evaluate with GroupKFold (subject-wise)
  6) Save best model artifacts for backend real-time inference
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Ensure project root is on sys.path so `import src.*` works when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cle.train.feature_engineering import (
    DEFAULT_BASELINE_TASKS,
    add_centered_and_delta,
    compute_user_baseline,
)
from src.cle.utils.io import save_json, save_model_artifact


BASE_VIDEO_FEATURES = [
    "blink_rate",
    "blink_count",
    "mean_blink_duration",
    "ear_std",
    "mean_brightness",
    "std_brightness",
    "perclos",
    "mean_quality",
    "valid_frame_ratio",
    "mouth_open_mean",
    "mouth_open_std",
    "roll_std",
    "motion_mean",
    "motion_std",
]


def _task_from_video_path(video_path: str) -> str | None:
    if not isinstance(video_path, str) or not video_path:
        return None
    stem = Path(video_path).stem
    parts = stem.split("_")
    return parts[-1] if parts else None


def load_binary_stress_labels(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    df[["user_id", "task"]] = df["subject/task"].str.split("_", n=1, expand=True)
    return df[["user_id", "task", "binary-stress"]].rename(columns={"binary-stress": "binary_stress"})


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if r is not None else float("nan")


def aggregate_session(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["user_id", "task"], as_index=False)
        .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
    )


def evaluate_model_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta: pd.DataFrame,
    model_factory,
    n_splits: int,
    seed: int,
) -> Dict:
    cv = GroupKFold(n_splits=n_splits)
    imputer = SimpleImputer(strategy="median")

    all_pred = np.zeros(len(y), dtype=float)
    fold_rows: List[Dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = model_factory(seed)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0.0, 1.0)
        all_pred[test_idx] = y_pred

        fold_rows.append(
            {
                "fold": fold_idx + 1,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "window_spearman": compute_spearman(y_test, y_pred),
            }
        )

    pred_df = meta.copy()
    pred_df["y_pred"] = all_pred

    window_spearman = compute_spearman(pred_df["y_true"].values, pred_df["y_pred"].values)
    session_df = aggregate_session(pred_df)
    session_spearman = compute_spearman(session_df["y_true"].values, session_df["y_pred"].values)

    return {
        "folds": fold_rows,
        "overall": {
            "window_spearman": window_spearman,
            "session_spearman": session_spearman,
            "n_windows": int(len(pred_df)),
            "n_sessions": int(len(session_df)),
        },
        "predictions": pred_df,
        "session_predictions": session_df,
    }


def _exact_join_video_physio(
    video_df: pd.DataFrame,
    physio_df: pd.DataFrame,
    *,
    round_dp: int,
) -> pd.DataFrame:
    v = video_df.copy()
    p = physio_df.copy()
    for df in (v, p):
        df["t_start_s_rounded"] = df["t_start_s"].round(round_dp)

    physio_keep = p[["user_id", "task", "t_start_s_rounded", "physio_stress_score"]].copy()
    merged = v.merge(
        physio_keep,
        on=["user_id", "task", "t_start_s_rounded"],
        how="inner",
    )
    return merged


def _overlap_weighted_label_video_windows(
    video_df: pd.DataFrame,
    physio_df: pd.DataFrame,
    *,
    user_col: str = "user_id",
    task_col: str = "task",
    v_start_col: str = "t_start_s",
    v_end_col: str = "t_end_s",
    p_start_col: str = "t_start_s",
    p_end_col: str = "t_end_s",
    score_col: str = "physio_stress_score",
) -> pd.DataFrame:
    """
    Assign a physio label to each video window by overlap-weighted averaging.

    For each (user_id, task):
      - For each video window [vs, ve], consider physio windows overlapping it.
      - Compute overlap durations and return weighted mean of physio_stress_score.
      - If there is no overlap, fall back to nearest physio window by center time.
    """
    required_video = {user_col, task_col, v_start_col, v_end_col}
    missing_video = required_video - set(video_df.columns)
    if missing_video:
        raise ValueError(f"video_df missing required columns: {sorted(missing_video)}")

    required_physio = {user_col, task_col, p_start_col, p_end_col, score_col}
    missing_physio = required_physio - set(physio_df.columns)
    if missing_physio:
        raise ValueError(f"physio_df missing required columns: {sorted(missing_physio)}")

    v = video_df.copy()
    p = physio_df.copy()
    v[user_col] = v[user_col].astype(str)
    v[task_col] = v[task_col].astype(str)
    p[user_col] = p[user_col].astype(str)
    p[task_col] = p[task_col].astype(str)

    # Pre-group physio for efficient lookup.
    physio_groups = {k: g for k, g in p.groupby([user_col, task_col], sort=False)}

    labeled_groups: List[pd.DataFrame] = []
    for (user_id, task), vg in v.groupby([user_col, task_col], sort=False):
        pg = physio_groups.get((user_id, task))
        vg_out = vg.copy()

        if pg is None or len(pg) == 0:
            vg_out[score_col] = np.nan
            labeled_groups.append(vg_out)
            continue

        pg = pg[[p_start_col, p_end_col, score_col]].dropna(subset=[score_col]).copy()
        if len(pg) == 0:
            vg_out[score_col] = np.nan
            labeled_groups.append(vg_out)
            continue

        # Sort physio windows by start time (also keeps end times monotonic).
        pg = pg.sort_values(p_start_col, ascending=True)
        p_start = pg[p_start_col].values.astype(float)
        p_end = pg[p_end_col].values.astype(float)
        p_score = pg[score_col].values.astype(float)
        p_center = (p_start + p_end) / 2.0

        labels: List[float] = []
        for vs, ve in vg_out[[v_start_col, v_end_col]].values.astype(float):
            # Overlap condition: p_start < ve and p_end > vs
            left = int(np.searchsorted(p_end, vs, side="right"))  # first p_end > vs
            right = int(np.searchsorted(p_start, ve, side="left"))  # first p_start >= ve

            if left < right:
                ps = p_start[left:right]
                pe = p_end[left:right]
                sc = p_score[left:right]
                overlap = np.maximum(0.0, np.minimum(ve, pe) - np.maximum(vs, ps))
                total = float(overlap.sum())
                if total > 0:
                    labels.append(float((overlap * sc).sum() / total))
                    continue

            # Fallback: nearest physio window by center time
            v_center = (vs + ve) / 2.0
            idx = int(np.argmin(np.abs(p_center - v_center)))
            labels.append(float(p_score[idx]))

        vg_out[score_col] = labels
        labeled_groups.append(vg_out)

    return pd.concat(labeled_groups, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train video regression model on physio_stress_score")
    parser.add_argument(
        "--video-features",
        type=str,
        default="data/processed/stress_features_10s.csv",
        help="Video window features CSV (10s/2.5s recommended).",
    )
    parser.add_argument(
        "--physio-labels",
        type=str,
        default="data/processed/physio_stress_labels.csv",
        help="Physio stress label CSV (10s/2.5s).",
    )
    parser.add_argument(
        "--binary-labels",
        type=str,
        default="../labels.csv",
        help="labels.csv path for secondary binary-stress AUC.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models/video_physio_regression",
        help="Output directory for model artifacts.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="reports/video_physio_regression_eval.json",
        help="Output report JSON path.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round-dp", type=int, default=3, help="Rounding decimals for t_start_s join.")
    parser.add_argument(
        "--merge-mode",
        type=str,
        default="auto",
        choices=["auto", "exact", "overlap"],
        help="How to label video windows with physio: exact (rounded t_start join), overlap (overlap-weighted), auto (try exact then fallback).",
    )
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
    out_dir = Path(args.out)
    report_path = Path(args.report)

    video_df = pd.read_csv(video_path)
    if "task" not in video_df.columns:
        video_df["task"] = video_df["video"].apply(_task_from_video_path)

    physio_df = pd.read_csv(physio_path)

    # Ensure join keys are compatible
    for df in (video_df, physio_df):
        df["user_id"] = df["user_id"].astype(str)
        df["task"] = df["task"].astype(str)

    merged: pd.DataFrame
    if args.merge_mode in ("exact", "auto"):
        merged_exact = _exact_join_video_physio(video_df, physio_df, round_dp=args.round_dp)
    else:
        merged_exact = pd.DataFrame()

    if args.merge_mode == "exact":
        merged = merged_exact
    elif args.merge_mode == "overlap":
        merged = _overlap_weighted_label_video_windows(video_df, physio_df)
    else:
        # auto: prefer exact join if it captures most windows; otherwise use overlap labeling
        exact_ratio = (len(merged_exact) / max(len(video_df), 1)) if len(video_df) else 0.0
        if len(merged_exact) > 0 and exact_ratio >= 0.8:
            merged = merged_exact
        else:
            merged = _overlap_weighted_label_video_windows(video_df, physio_df)

    merged = merged.dropna(subset=["physio_stress_score"]).copy()
    if len(merged) == 0:
        raise ValueError("No merged rows. Check windowing alignment and join keys.")

    # Ensure feature columns exist
    missing_feats = [c for c in BASE_VIDEO_FEATURES if c not in merged.columns]
    if missing_feats:
        raise ValueError(f"Missing video features in merged data: {missing_feats}")

    baseline_df = compute_user_baseline(
        merged,
        feature_cols=BASE_VIDEO_FEATURES,
        baseline_tasks=DEFAULT_BASELINE_TASKS,
        n_windows=args.baseline_n_windows,
        time_col="t_start_s",
    )

    engineered, spec = add_centered_and_delta(
        merged,
        feature_cols=BASE_VIDEO_FEATURES,
        baseline_df=baseline_df,
        session_cols=("user_id", "task"),
        time_col="t_start_s",
    )

    X = engineered[spec.all_features].values
    score = engineered["physio_stress_score"].values.astype(float)
    target_name = "physio_stress_score"

    if args.target == "abs":
        y = score
        target_name = "physio_stress_score"
    elif args.target == "delta":
        # Per-user baseline on physio score using the same baseline task definition as features.
        score_baseline_df = compute_user_baseline(
            merged,
            feature_cols=["physio_stress_score"],
            baseline_tasks=DEFAULT_BASELINE_TASKS,
            n_windows=args.baseline_n_windows,
            time_col="t_start_s",
        )
        engineered = engineered.merge(
            score_baseline_df[["user_id", "baseline_physio_stress_score"]],
            on="user_id",
            how="left",
        )
        delta = score - engineered["baseline_physio_stress_score"].values.astype(float)
        y = np.clip(0.5 + 0.5 * delta, 0.0, 1.0)
        target_name = "physio_stress_delta01"
    else:
        # Per-user z-score mapped to 0-1 via sigmoid.
        mu = engineered.groupby("user_id")["physio_stress_score"].transform("mean").values.astype(float)
        sigma = (
            engineered.groupby("user_id")["physio_stress_score"]
            .transform(lambda s: s.std(ddof=0))
            .values.astype(float)
        )
        z = np.zeros_like(score, dtype=float)
        mask = sigma >= 1e-6
        z[mask] = (score[mask] - mu[mask]) / sigma[mask]
        y = 1.0 / (1.0 + np.exp(-z))
        target_name = "physio_stress_z01"

    groups = engineered["user_id"].astype(str).values
    meta = engineered[["user_id", "task", "t_start_s", "physio_stress_score"]].copy()
    meta["y_true"] = y

    models = {
        "Ridge": lambda seed: Ridge(alpha=1.0),
        "GradientBoosting": lambda seed: GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=10,
            random_state=seed,
        ),
    }

    results: Dict[str, Dict] = {}
    for name, factory in models.items():
        results[name] = evaluate_model_cv(
            X=X,
            y=y,
            groups=groups,
            meta=meta,
            model_factory=factory,
            n_splits=args.n_splits,
            seed=args.seed,
        )

    best_name = max(results.keys(), key=lambda n: results[n]["overall"]["session_spearman"])

    # Train final artifacts on all data with best model
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    final_model = models[best_name](args.seed)
    final_model.fit(X_scaled, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_model_artifact(final_model, out_dir / "model_regression.bin")
    save_model_artifact(scaler, out_dir / "scaler.bin")
    save_model_artifact(imputer, out_dir / "imputer.bin")

    feature_spec = {
        "features": spec.all_features,
        "n_features": len(spec.all_features),
        "task_mode": "regression",
        "target": target_name,
        "base_features": list(spec.base_features),
    }
    save_json(feature_spec, out_dir / "feature_spec.json")

    model_meta = {
        "training_date": datetime.now().isoformat(),
        "model_type": best_name,
        "seed": args.seed,
        "video_features": str(video_path),
        "physio_labels": str(physio_path),
        "target": args.target,
        "target_name": target_name,
        "baseline_tasks": sorted(DEFAULT_BASELINE_TASKS),
        "baseline_n_windows": args.baseline_n_windows,
        "merge_mode": args.merge_mode,
        "round_dp": args.round_dp,
    }
    save_json(model_meta, out_dir / "model_meta.json")

    # Secondary metric: session AUC vs binary-stress
    binary_auc = None
    try:
        labels_df = load_binary_stress_labels(Path(args.binary_labels))
        session_best = results[best_name]["session_predictions"].merge(labels_df, on=["user_id", "task"], how="inner")
        session_best = session_best.dropna(subset=["binary_stress", "y_pred"])
        if len(session_best["binary_stress"].unique()) > 1:
            binary_auc = float(roc_auc_score(session_best["binary_stress"].values, session_best["y_pred"].values))
    except Exception:
        binary_auc = None

    report = {
        "timestamp": datetime.now().isoformat(),
        "target": args.target,
        "target_name": target_name,
        "video_features": str(video_path),
        "physio_labels": str(physio_path),
        "feature_spec": feature_spec,
        "baseline_tasks": sorted(DEFAULT_BASELINE_TASKS),
        "baseline_n_windows": args.baseline_n_windows,
        "models": {
            name: {
                "overall": results[name]["overall"],
                "folds": results[name]["folds"],
            }
            for name in results.keys()
        },
        "best_model": best_name,
        "best_overall": results[best_name]["overall"],
        "secondary_binary_stress_auc": binary_auc,
        "artifacts_dir": str(out_dir),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
