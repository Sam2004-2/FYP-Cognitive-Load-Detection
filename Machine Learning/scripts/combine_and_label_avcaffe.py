"""
Combine AVCAffe feature parts and add cognitive load labels.

Takes part1-4.csv files from parallel extraction, combines them,
and joins with ground truth mental demand labels to create a single
labeled dataset for regression training.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cle.data.load_avcaffe_labels import load_mental_demand_labels
from src.cle.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def combine_part_files(part_files: List[str]) -> pd.DataFrame:
    """
    Combine multiple part CSV files into single DataFrame.

    Args:
        part_files: List of paths to part*.csv files

    Returns:
        Combined DataFrame

    Raises:
        FileNotFoundError: If any part file doesn't exist
        ValueError: If part files have inconsistent schemas
    """
    logger.info(f"Loading {len(part_files)} part files...")

    dfs = []
    total_rows = 0

    for part_file in part_files:
        if not Path(part_file).exists():
            raise FileNotFoundError(f"Part file not found: {part_file}")

        df = pd.read_csv(part_file)
        rows = len(df)
        total_rows += rows

        logger.info(f"  {Path(part_file).name}: {rows:,} rows")
        dfs.append(df)

    # Check schema consistency
    base_cols = set(dfs[0].columns)
    for i, df in enumerate(dfs[1:], start=2):
        if set(df.columns) != base_cols:
            missing = base_cols - set(df.columns)
            extra = set(df.columns) - base_cols
            raise ValueError(
                f"Part {i} has inconsistent schema. "
                f"Missing: {missing}, Extra: {extra}"
            )

    # Combine
    logger.info(f"Combining {len(dfs)} DataFrames...")
    combined = pd.concat(dfs, ignore_index=True)

    logger.info(f"Combined: {len(combined):,} rows ({total_rows:,} total from parts)")

    # Check for and remove duplicates
    before_dedup = len(combined)
    combined = combined.drop_duplicates(
        subset=["participant_id", "task", "window_idx"],
        keep="first"
    )
    after_dedup = len(combined)

    if before_dedup != after_dedup:
        removed = before_dedup - after_dedup
        logger.warning(
            f"Removed {removed:,} duplicate windows from overlapping parts"
        )
        logger.info(f"Unique windows after deduplication: {after_dedup:,}")

    return combined


def join_features_with_labels(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join feature windows with cognitive load labels.

    Args:
        features_df: DataFrame with columns participant_id, task, and features
        labels_df: DataFrame from load_mental_demand_labels()

    Returns:
        DataFrame with cognitive_load column added

    Raises:
        ValueError: If required columns are missing
    """
    logger.info("Joining features with labels...")

    # Check required columns
    if "participant_id" not in features_df.columns:
        raise ValueError("features_df missing 'participant_id' column")
    if "task" not in features_df.columns:
        raise ValueError("features_df missing 'task' column")

    # Create join keys
    features_df = features_df.copy()
    labels_df = labels_df.copy()

    features_df["join_key"] = (
        features_df["participant_id"] + "_" + features_df["task"]
    )
    labels_df["join_key"] = labels_df["participant_id"] + "_" + labels_df["task"]

    logger.info(f"Features: {len(features_df):,} windows")
    logger.info(f"Labels: {len(labels_df)} participant-task pairs")

    # Left join to preserve all windows
    combined = features_df.merge(
        labels_df[["join_key", "cognitive_load"]],
        on="join_key",
        how="left",
    )

    # Drop the temporary join key
    combined = combined.drop(columns=["join_key"])

    # Check label coverage
    labeled = (~combined["cognitive_load"].isna()).sum()
    unlabeled = combined["cognitive_load"].isna().sum()
    coverage_pct = (labeled / len(combined)) * 100 if len(combined) > 0 else 0

    logger.info(f"Join results:")
    logger.info(f"  Labeled windows: {labeled:,} ({coverage_pct:.1f}%)")
    logger.info(f"  Unlabeled windows: {unlabeled:,}")

    if unlabeled > 0:
        # Identify which participant-task combinations are missing labels
        unlabeled_tasks = (
            combined[combined["cognitive_load"].isna()]
            .groupby(["participant_id", "task"])
            .size()
            .reset_index(name="window_count")
        )

        logger.warning(
            f"Missing labels for {len(unlabeled_tasks)} participant-task combinations:"
        )
        for _, row in unlabeled_tasks.head(10).iterrows():
            logger.warning(
                f"  {row['participant_id']}_{row['task']}: "
                f"{row['window_count']} windows"
            )

        if len(unlabeled_tasks) > 10:
            logger.warning(f"  ... and {len(unlabeled_tasks) - 10} more")

        if coverage_pct < 95.0:
            logger.error(
                f"Label coverage is only {coverage_pct:.1f}%! "
                f"Expected >95%. Check if label file is complete."
            )

    return combined


def validate_combined_data(df: pd.DataFrame) -> None:
    """
    Validate the combined dataset.

    Args:
        df: Combined DataFrame with features and labels

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating combined dataset...")

    # Check for duplicates
    duplicates = df.duplicated(subset=["participant_id", "task", "window_idx"]).sum()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate windows!")

    # Check cognitive_load range for labeled windows
    labeled_df = df[~df["cognitive_load"].isna()]
    if len(labeled_df) > 0:
        min_load = labeled_df["cognitive_load"].min()
        max_load = labeled_df["cognitive_load"].max()

        if min_load < 0 or max_load > 1:
            raise ValueError(
                f"cognitive_load out of range [0,1]: [{min_load}, {max_load}]"
            )

        logger.info(f"Cognitive load range: [{min_load:.3f}, {max_load:.3f}]")
        logger.info(f"Cognitive load mean: {labeled_df['cognitive_load'].mean():.3f}")
        logger.info(f"Cognitive load std: {labeled_df['cognitive_load'].std():.3f}")

    # Check for NaN in features
    feature_cols = [
        col for col in df.columns
        if col not in ["participant_id", "task", "window_idx", "video_file",
                       "window_start_s", "window_end_s", "window_start_frame",
                       "window_end_frame", "cognitive_load"]
    ]

    nan_counts = df[feature_cols].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]

    if len(cols_with_nan) > 0:
        logger.warning("Features with NaN values:")
        for col, count in cols_with_nan.items():
            pct = (count / len(df)) * 100
            logger.warning(f"  {col}: {count:,} ({pct:.2f}%)")

    logger.info("Validation passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Combine AVCAffe feature parts and add cognitive load labels"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input part CSV files (e.g., part1.csv part2.csv ...)",
    )
    parser.add_argument(
        "--labels",
        default="E:/FYP/Dataset/AVCAffe/codes/downloader/data/ground_truths/mental_demand.txt",
        help="Path to mental_demand.txt label file",
    )
    parser.add_argument(
        "--output",
        default="data/processed/avcaffe_labeled_features.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after combining",
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
    logger.info("AVCAffe Feature Combination & Labeling")
    logger.info("=" * 60)

    try:
        # Step 1: Combine part files
        features_df = combine_part_files(args.input)

        # Step 2: Load labels
        labels_df = load_mental_demand_labels(args.labels)

        # Step 3: Join features with labels
        combined_df = join_features_with_labels(features_df, labels_df)

        # Step 4: Validate if requested
        if args.validate:
            validate_combined_data(combined_df)

        # Step 5: Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving to: {output_path}")
        combined_df.to_csv(output_path, index=False)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {len(combined_df):,} rows ({file_size_mb:.1f} MB)")

        logger.info("=" * 60)
        logger.info("SUCCESS! Combined dataset created.")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
