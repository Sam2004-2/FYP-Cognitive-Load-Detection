#!/usr/bin/env python3
"""
Empirical analysis of physiological stress patterns.

This script discovers what stress looks like in ECG, EDA, and respiratory data
by comparing high-stress vs low-stress tasks. It validates that physiological
features discriminate stress states before using them for model training.

Key outputs:
- Statistical comparison (Cohen's d, t-tests) between stress conditions
- Feature importance ranking
- Validation against physiological literature expectations
- Composite stress score recommendations
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime

# Task categorization based on StressID protocol
HIGH_STRESS_TASKS = {"Math", "Speaking", "Stroop", "Counting1", "Counting2", "Counting3"}
LOW_STRESS_TASKS = {"Relax", "Baseline", "Breathing"}
MID_STRESS_TASKS = {"Reading", "Video1", "Video2"}  # Excluded from analysis

# Expected physiological stress responses (from literature)
# Positive = increases with stress, Negative = decreases with stress
EXPECTED_DIRECTION = {
    # ECG/HRV features
    "hr": "positive",           # Heart rate increases under stress
    "rmssd": "negative",        # HRV (RMSSD) decreases under stress (vagal withdrawal)
    "sdnn": "negative",         # HRV (SDNN) decreases under stress
    
    # EDA features  
    "scl": "positive",          # Skin conductance level increases with arousal
    "scr_count": "positive",    # More skin conductance responses under stress
    "scr_amplitude_mean": "positive",  # Larger SCRs under stress
    
    # Respiratory features
    "resp_rate": "positive",    # Breathing rate increases under stress
    "resp_amplitude_mean": "variable",  # Can go either way
    "resp_variability": "negative",     # More regular breathing under stress (controlled)
}


def load_physio_features(features_path: Path) -> pd.DataFrame:
    """Load extracted physiological features."""
    print(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    print(f"  Loaded {len(df)} windows")
    print(f"  Participants: {df['user_id'].nunique()}")
    print(f"  Tasks: {df['task'].unique().tolist()}")
    return df


def categorize_stress_level(df: pd.DataFrame) -> pd.DataFrame:
    """Add stress_category column based on task type."""
    df = df.copy()
    
    def get_category(task):
        if task in HIGH_STRESS_TASKS:
            return "high"
        elif task in LOW_STRESS_TASKS:
            return "low"
        else:
            return "mid"
    
    df["stress_category"] = df["task"].apply(get_category)
    
    # Print distribution
    print("\nStress category distribution:")
    for cat in ["low", "mid", "high"]:
        count = (df["stress_category"] == cat).sum()
        tasks = df[df["stress_category"] == cat]["task"].unique().tolist()
        print(f"  {cat}: {count} windows ({tasks})")
    
    return df


def aggregate_to_task_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window-level features to task level (mean per participant-task)."""
    feature_cols = [
        "hr", "rmssd", "sdnn",
        "scl", "scr_count", "scr_amplitude_mean",
        "resp_rate", "resp_amplitude_mean", "resp_variability"
    ]
    
    # Only keep rows with valid features
    valid_mask = df["hr"].notna() & (df["hr"] > 0)
    df_valid = df[valid_mask].copy()
    
    # Aggregate to task level
    agg_dict = {col: "mean" for col in feature_cols if col in df_valid.columns}
    agg_dict["stress_category"] = "first"
    
    df_agg = df_valid.groupby(["user_id", "task"]).agg(agg_dict).reset_index()
    
    print(f"\nAggregated to {len(df_agg)} participant-task combinations")
    return df_agg


def compute_effect_sizes(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Compute Cohen's d effect sizes comparing high vs low stress.
    
    Cohen's d interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    results = {}
    
    high = df[df["stress_category"] == "high"]
    low = df[df["stress_category"] == "low"]
    
    print(f"\nComparing {len(high)} high-stress vs {len(low)} low-stress samples")
    
    for col in feature_cols:
        if col not in df.columns:
            continue
            
        high_vals = high[col].dropna()
        low_vals = low[col].dropna()
        
        if len(high_vals) < 10 or len(low_vals) < 10:
            print(f"  {col}: insufficient data")
            continue
        
        # Cohen's d
        pooled_std = np.sqrt(
            ((len(high_vals) - 1) * high_vals.std()**2 + 
             (len(low_vals) - 1) * low_vals.std()**2) / 
            (len(high_vals) + len(low_vals) - 2)
        )
        
        if pooled_std > 0:
            cohens_d = (high_vals.mean() - low_vals.mean()) / pooled_std
        else:
            cohens_d = 0
        
        # t-test
        t_stat, p_value = stats.ttest_ind(high_vals, low_vals)
        
        # Direction check
        observed_direction = "positive" if high_vals.mean() > low_vals.mean() else "negative"
        expected = EXPECTED_DIRECTION.get(col, "unknown")
        matches_literature = (
            expected == "variable" or 
            expected == "unknown" or 
            observed_direction == expected
        )
        
        results[col] = {
            "high_mean": float(high_vals.mean()),
            "high_std": float(high_vals.std()),
            "low_mean": float(low_vals.mean()),
            "low_std": float(low_vals.std()),
            "cohens_d": float(cohens_d),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "observed_direction": observed_direction,
            "expected_direction": expected,
            "matches_literature": matches_literature,
            "n_high": len(high_vals),
            "n_low": len(low_vals),
        }
    
    return results


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def print_results_table(results: dict):
    """Print a formatted results table."""
    print("\n" + "=" * 100)
    print("PHYSIOLOGICAL STRESS PATTERN ANALYSIS")
    print("=" * 100)
    
    # Sort by absolute effect size
    sorted_features = sorted(results.keys(), key=lambda x: abs(results[x]["cohens_d"]), reverse=True)
    
    print(f"\n{'Feature':<25} {'Cohen d':>10} {'Magnitude':>12} {'Direction':>12} {'Expected':>12} {'Match':>8} {'p-value':>12}")
    print("-" * 100)
    
    for feat in sorted_features:
        r = results[feat]
        magnitude = interpret_effect_size(r["cohens_d"])
        match_str = "✓" if r["matches_literature"] else "✗"
        p_str = f"{r['p_value']:.2e}" if r['p_value'] < 0.001 else f"{r['p_value']:.4f}"
        
        print(f"{feat:<25} {r['cohens_d']:>10.3f} {magnitude:>12} {r['observed_direction']:>12} "
              f"{r['expected_direction']:>12} {match_str:>8} {p_str:>12}")
    
    print("-" * 100)
    
    # Summary
    valid_features = [f for f in sorted_features if results[f]["matches_literature"] and results[f]["p_value"] < 0.05]
    print(f"\nValidated features (p<0.05 & matches literature): {valid_features}")
    
    strong_features = [f for f in sorted_features if abs(results[f]["cohens_d"]) >= 0.5 and results[f]["p_value"] < 0.05]
    print(f"Strong discriminators (|d|>=0.5 & p<0.05): {strong_features}")


def compute_composite_score(df: pd.DataFrame, validated_features: list, effect_sizes: dict) -> pd.DataFrame:
    """
    Compute a composite physiological stress score using validated features.
    
    Each feature is z-scored and weighted by its effect size, then combined.
    Direction is aligned so positive = more stress.
    """
    df = df.copy()
    
    weighted_sum = np.zeros(len(df))
    total_weight = 0
    
    for feat in validated_features:
        if feat not in df.columns or feat not in effect_sizes:
            continue
        
        # Z-score the feature
        mean_val = df[feat].mean()
        std_val = df[feat].std()
        if std_val == 0:
            continue
        
        z_scored = (df[feat] - mean_val) / std_val
        
        # Flip direction if high stress = lower values
        if effect_sizes[feat]["observed_direction"] == "negative":
            z_scored = -z_scored
        
        # Weight by absolute effect size
        weight = abs(effect_sizes[feat]["cohens_d"])
        weighted_sum += z_scored.fillna(0) * weight
        total_weight += weight
    
    if total_weight > 0:
        df["composite_stress_score"] = weighted_sum / total_weight
    else:
        df["composite_stress_score"] = 0
    
    return df


def validate_composite_score(df: pd.DataFrame):
    """Validate that composite score discriminates stress conditions."""
    print("\n" + "=" * 60)
    print("COMPOSITE STRESS SCORE VALIDATION")
    print("=" * 60)
    
    high = df[df["stress_category"] == "high"]["composite_stress_score"].dropna()
    low = df[df["stress_category"] == "low"]["composite_stress_score"].dropna()
    
    print(f"\nHigh-stress tasks: mean={high.mean():.3f}, std={high.std():.3f}, n={len(high)}")
    print(f"Low-stress tasks:  mean={low.mean():.3f}, std={low.std():.3f}, n={len(low)}")
    
    # Effect size
    pooled_std = np.sqrt(((len(high)-1)*high.std()**2 + (len(low)-1)*low.std()**2) / (len(high)+len(low)-2))
    cohens_d = (high.mean() - low.mean()) / pooled_std
    
    t_stat, p_value = stats.ttest_ind(high, low)
    
    print(f"\nComposite score Cohen's d: {cohens_d:.3f} ({interpret_effect_size(cohens_d)})")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.2e}")
    
    # AUC-like metric
    from scipy.stats import mannwhitneyu
    u_stat, mw_p = mannwhitneyu(high, low, alternative='greater')
    auc = u_stat / (len(high) * len(low))
    print(f"Mann-Whitney U AUC: {auc:.3f}")
    
    return {
        "cohens_d": float(cohens_d),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "auc": float(auc),
        "high_mean": float(high.mean()),
        "low_mean": float(low.mean()),
    }


def main():
    """Run full physiological stress pattern analysis."""
    # Paths
    project_root = Path(__file__).parent.parent
    features_path = project_root / "data" / "processed" / "physio_features.csv"
    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)
    
    # Feature columns to analyze
    feature_cols = [
        "hr", "rmssd", "sdnn",
        "scl", "scr_count", "scr_amplitude_mean", 
        "resp_rate", "resp_amplitude_mean", "resp_variability"
    ]
    
    # Load and prepare data
    df = load_physio_features(features_path)
    df = categorize_stress_level(df)
    
    # Aggregate to task level for cleaner analysis
    df_agg = aggregate_to_task_level(df)
    
    # Compute effect sizes for all features
    effect_sizes = compute_effect_sizes(df_agg, feature_cols)
    
    # Print results table
    print_results_table(effect_sizes)
    
    # Identify validated features
    validated_features = [
        f for f in effect_sizes 
        if effect_sizes[f]["matches_literature"] 
        and effect_sizes[f]["p_value"] < 0.05
        and abs(effect_sizes[f]["cohens_d"]) >= 0.2  # At least small effect
    ]
    
    print(f"\n\nVALIDATED FEATURES FOR TEACHER MODEL:")
    print(f"  {validated_features}")
    
    # Compute and validate composite score
    df_agg = compute_composite_score(df_agg, validated_features, effect_sizes)
    composite_results = validate_composite_score(df_agg)
    
    # Save detailed results
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_windows": len(df),
        "n_task_aggregated": len(df_agg),
        "high_stress_tasks": list(HIGH_STRESS_TASKS),
        "low_stress_tasks": list(LOW_STRESS_TASKS),
        "feature_analysis": effect_sizes,
        "validated_features": validated_features,
        "composite_score_validation": composite_results,
        "expected_directions": EXPECTED_DIRECTION,
    }
    
    output_path = output_dir / "physio_stress_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")
    
    # Save recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if len(validated_features) >= 3:
        print("✓ Sufficient validated features for teacher model")
        print(f"  Use these features: {validated_features}")
    else:
        print("⚠ Few validated features - consider:")
        print("  - Using all features but with lower confidence")
        print("  - Investigating data quality issues")
    
    if composite_results["auc"] >= 0.65:
        print(f"✓ Good composite discrimination (AUC={composite_results['auc']:.3f})")
        print("  Proceed with teacher training using composite score as target")
    elif composite_results["auc"] >= 0.55:
        print(f"⚠ Moderate composite discrimination (AUC={composite_results['auc']:.3f})")
        print("  Teacher may provide modest improvement")
    else:
        print(f"✗ Weak composite discrimination (AUC={composite_results['auc']:.3f})")
        print("  Physiological data may not reliably indicate stress in this dataset")
    
    return validated_features, effect_sizes, composite_results


if __name__ == "__main__":
    validated_features, effect_sizes, composite_results = main()
