"""
Validate AVCAffe labeled dataset.

Generates comprehensive validation report including coverage statistics,
label distribution, feature quality metrics, and data consistency checks.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cle.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def validate_dataset(df: pd.DataFrame) -> Dict:
    """
    Run comprehensive validation on AVCAffe dataset.

    Args:
        df: DataFrame with features and cognitive_load labels

    Returns:
        Dictionary with validation results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_windows": len(df),
    }

    logger.info(f"Total windows: {len(df):,}")

    # 1. Feature coverage
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE COVERAGE")
    logger.info("=" * 60)

    unique_participants = df["participant_id"].nunique()
    unique_tasks = df["task"].nunique()

    # Count unique participant-task combinations
    participant_task_pairs = (
        df.groupby(["participant_id", "task"])
        .size()
        .reset_index(name="window_count")
    )
    unique_pairs = len(participant_task_pairs)

    results["feature_coverage"] = {
        "unique_participants": int(unique_participants),
        "unique_tasks": int(unique_tasks),
        "unique_participant_tasks": int(unique_pairs),
        "windows_per_participant_task_mean": float(
            participant_task_pairs["window_count"].mean()
        ),
        "windows_per_participant_task_std": float(
            participant_task_pairs["window_count"].std()
        ),
    }

    logger.info(f"Unique participants: {unique_participants}")
    logger.info(f"Unique tasks: {unique_tasks}")
    logger.info(f"Unique participant-task pairs: {unique_pairs}")
    logger.info(
        f"Windows per participant-task: "
        f"{participant_task_pairs['window_count'].mean():.1f} ± "
        f"{participant_task_pairs['window_count'].std():.1f}"
    )

    # 2. Label coverage
    logger.info("\n" + "=" * 60)
    logger.info("LABEL COVERAGE")
    logger.info("=" * 60)

    labeled_windows = (~df["cognitive_load"].isna()).sum()
    unlabeled_windows = df["cognitive_load"].isna().sum()
    coverage_pct = (labeled_windows / len(df)) * 100 if len(df) > 0 else 0

    results["label_coverage"] = {
        "labeled_windows": int(labeled_windows),
        "unlabeled_windows": int(unlabeled_windows),
        "coverage_percentage": float(coverage_pct),
    }

    logger.info(f"Labeled windows: {labeled_windows:,} ({coverage_pct:.1f}%)")
    logger.info(f"Unlabeled windows: {unlabeled_windows:,}")

    # 3. Label distribution
    if labeled_windows > 0:
        logger.info("\n" + "=" * 60)
        logger.info("LABEL DISTRIBUTION")
        logger.info("=" * 60)

        labeled_df = df[~df["cognitive_load"].isna()]

        results["label_distribution"] = {
            "min": float(labeled_df["cognitive_load"].min()),
            "max": float(labeled_df["cognitive_load"].max()),
            "mean": float(labeled_df["cognitive_load"].mean()),
            "std": float(labeled_df["cognitive_load"].std()),
            "quartiles": [
                float(labeled_df["cognitive_load"].quantile(0.25)),
                float(labeled_df["cognitive_load"].quantile(0.50)),
                float(labeled_df["cognitive_load"].quantile(0.75)),
            ],
        }

        logger.info(f"Min: {results['label_distribution']['min']:.3f}")
        logger.info(f"Max: {results['label_distribution']['max']:.3f}")
        logger.info(f"Mean: {results['label_distribution']['mean']:.3f}")
        logger.info(f"Std: {results['label_distribution']['std']:.3f}")
        logger.info(
            f"Quartiles: "
            f"[{results['label_distribution']['quartiles'][0]:.3f}, "
            f"{results['label_distribution']['quartiles'][1]:.3f}, "
            f"{results['label_distribution']['quartiles'][2]:.3f}]"
        )

    # 4. Feature quality
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE QUALITY")
    logger.info("=" * 60)

    quality_metrics = {}
    if "mean_quality" in df.columns:
        quality_metrics["mean_quality_avg"] = float(df["mean_quality"].mean())
        quality_metrics["mean_quality_std"] = float(df["mean_quality"].std())
        logger.info(
            f"mean_quality: {quality_metrics['mean_quality_avg']:.3f} ± "
            f"{quality_metrics['mean_quality_std']:.3f}"
        )

    if "valid_frame_ratio" in df.columns:
        quality_metrics["valid_frame_ratio_avg"] = float(
            df["valid_frame_ratio"].mean()
        )
        quality_metrics["valid_frame_ratio_std"] = float(
            df["valid_frame_ratio"].std()
        )
        logger.info(
            f"valid_frame_ratio: {quality_metrics['valid_frame_ratio_avg']:.3f} ± "
            f"{quality_metrics['valid_frame_ratio_std']:.3f}"
        )

    # Check for NaN in features
    feature_cols = [
        col
        for col in df.columns
        if col
        not in [
            "participant_id",
            "task",
            "window_idx",
            "video_file",
            "window_start_s",
            "window_end_s",
            "window_start_frame",
            "window_end_frame",
            "cognitive_load",
        ]
    ]

    nan_counts = df[feature_cols].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0].to_dict()

    quality_metrics["features_with_nan"] = {
        str(k): int(v) for k, v in cols_with_nan.items()
    }

    if cols_with_nan:
        logger.warning("Features with NaN values:")
        for col, count in cols_with_nan.items():
            pct = (count / len(df)) * 100
            logger.warning(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        logger.info("No NaN values in features ✓")

    results["feature_quality"] = quality_metrics

    # 5. Data consistency
    logger.info("\n" + "=" * 60)
    logger.info("DATA CONSISTENCY")
    logger.info("=" * 60)

    consistency = {}

    # Check for duplicates
    duplicates = df.duplicated(subset=["participant_id", "task", "window_idx"]).sum()
    consistency["duplicates"] = int(duplicates)

    if duplicates > 0:
        logger.error(f"Found {duplicates} duplicate windows! ✗")
    else:
        logger.info("No duplicate windows ✓")

    # Check time ranges
    if "window_start_s" in df.columns and "window_end_s" in df.columns:
        invalid_time = (df["window_end_s"] <= df["window_start_s"]).sum()
        consistency["time_range_errors"] = int(invalid_time)

        if invalid_time > 0:
            logger.error(f"Found {invalid_time} windows with invalid time ranges! ✗")
        else:
            logger.info("All time ranges valid ✓")

    # Check frame ranges
    if "window_start_frame" in df.columns and "window_end_frame" in df.columns:
        invalid_frames = (df["window_end_frame"] <= df["window_start_frame"]).sum()
        consistency["frame_range_errors"] = int(invalid_frames)

        if invalid_frames > 0:
            logger.error(
                f"Found {invalid_frames} windows with invalid frame ranges! ✗"
            )
        else:
            logger.info("All frame ranges valid ✓")

    results["consistency"] = consistency

    # 6. Overall status
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    # Determine overall status
    issues = []

    if coverage_pct < 95.0:
        issues.append(f"Low label coverage: {coverage_pct:.1f}%")

    if duplicates > 0:
        issues.append(f"{duplicates} duplicate windows")

    if consistency.get("time_range_errors", 0) > 0:
        issues.append(f"{consistency['time_range_errors']} time range errors")

    if consistency.get("frame_range_errors", 0) > 0:
        issues.append(f"{consistency['frame_range_errors']} frame range errors")

    if len(cols_with_nan) > 0:
        issues.append(f"{len(cols_with_nan)} features with NaN values")

    if issues:
        results["status"] = "FAIL"
        results["issues"] = issues
        logger.error(f"Validation FAILED with {len(issues)} issue(s):")
        for issue in issues:
            logger.error(f"  - {issue}")
    else:
        results["status"] = "PASS"
        results["issues"] = []
        logger.info("Validation PASSED ✓")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate AVCAffe labeled dataset"
    )
    parser.add_argument(
        "--input",
        default="data/processed/avcaffe_labeled_features.csv",
        help="Input CSV file to validate",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for validation report",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    setup_logging(level=args.log_level)

    logger.info("=" * 60)
    logger.info("AVCAffe Dataset Validation")
    logger.info("=" * 60)

    try:
        # Load dataset
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1

        logger.info(f"Loading dataset from: {input_path}")
        df = pd.read_csv(input_path)

        # Run validation
        results = validate_dataset(df)

        # Save report
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"avcaffe_validation_{timestamp}.json"

        logger.info(f"\nSaving validation report to: {report_path}")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("=" * 60)

        if results["status"] == "PASS":
            logger.info("Validation completed successfully!")
            return 0
        else:
            logger.error("Validation completed with issues.")
            return 1

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
