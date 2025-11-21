"""
Combine feature CSV files from multiple participants into a single file.

This script merges individual participant feature files into one consolidated
CSV file suitable for model training.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cle.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def find_participant_files(input_pattern: str) -> list:
    """
    Find all participant feature files matching pattern.

    Args:
        input_pattern: Glob pattern for finding files (e.g., "participant_*_features.csv")

    Returns:
        List of Path objects for matching files
    """
    # Extract directory and pattern
    pattern_path = Path(input_pattern)
    
    if pattern_path.is_absolute():
        directory = pattern_path.parent
        pattern = pattern_path.name
    else:
        directory = Path.cwd()
        pattern = input_pattern

    # Find matching files
    files = sorted(directory.glob(pattern))
    
    if not files:
        # Try from data/processed directory
        alt_directory = Path("data/processed")
        if alt_directory.exists():
            files = sorted(alt_directory.glob(pattern))
    
    return files


def combine_features(input_files: list, output_file: str, verify: bool = True) -> pd.DataFrame:
    """
    Combine multiple feature CSV files into one.

    Args:
        input_files: List of input CSV file paths
        output_file: Output CSV file path
        verify: Whether to verify data consistency

    Returns:
        Combined DataFrame
    """
    if not input_files:
        raise ValueError("No input files provided")

    logger.info(f"Combining {len(input_files)} participant files...")

    dfs = []
    participant_stats = {}

    for file_path in input_files:
        logger.info(f"Reading: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Track stats
            if "user_id" in df.columns:
                user_ids = df["user_id"].unique()
                for user_id in user_ids:
                    user_df = df[df["user_id"] == user_id]
                    participant_stats[user_id] = {
                        "windows": len(user_df),
                        "labels": user_df["label"].value_counts().to_dict() if "label" in user_df.columns else {},
                        "source_file": file_path.name,
                    }
            
            dfs.append(df)
            logger.info(f"  Loaded {len(df)} windows")
            
        except Exception as e:
            logger.error(f"  Error reading {file_path}: {e}")
            continue

    if not dfs:
        raise ValueError("No data loaded from input files")

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined total: {len(combined_df)} windows")

    # Verify data consistency
    if verify:
        logger.info("Verifying data consistency...")
        
        # Check for required columns
        required_cols = ["user_id", "video", "label", "role"]
        missing_cols = [col for col in required_cols if col not in combined_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        
        # Check for duplicates
        if "user_id" in combined_df.columns and "t_start_s" in combined_df.columns:
            duplicates = combined_df.duplicated(subset=["user_id", "video", "t_start_s"])
            if duplicates.any():
                logger.warning(f"Found {duplicates.sum()} duplicate windows")
        
        # Check for NaN values in features
        feature_cols = [col for col in combined_df.columns 
                       if col not in ["user_id", "video", "label", "role", "t_start_s", "t_end_s"]]
        nan_counts = combined_df[feature_cols].isna().sum()
        if nan_counts.any():
            logger.warning("NaN values found in features:")
            for col, count in nan_counts[nan_counts > 0].items():
                logger.warning(f"  {col}: {count} NaN values")
        
        # Check label distribution
        if "label" in combined_df.columns:
            label_dist = combined_df["label"].value_counts()
            logger.info(f"Label distribution:\n{label_dist}")
            
            # Warn if imbalanced
            if len(label_dist) > 1:
                ratio = label_dist.max() / label_dist.min()
                if ratio > 3:
                    logger.warning(f"Class imbalance detected (ratio: {ratio:.1f}:1)")
        
        # Check role distribution
        if "role" in combined_df.columns:
            role_dist = combined_df["role"].value_counts()
            logger.info(f"Role distribution:\n{role_dist}")
    
    # Save combined file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Saved combined features to: {output_path}")
    
    # Print summary
    logger.info("=" * 80)
    logger.info("COMBINATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Input files: {len(input_files)}")
    logger.info(f"Total windows: {len(combined_df)}")
    logger.info(f"Participants: {len(participant_stats)}")
    
    if participant_stats:
        logger.info("\nPer-Participant Statistics:")
        for user_id, stats in sorted(participant_stats.items()):
            logger.info(f"  {user_id}:")
            logger.info(f"    Windows: {stats['windows']}")
            if stats['labels']:
                label_str = ", ".join([f"{k}={v}" for k, v in stats['labels'].items()])
                logger.info(f"    Labels: {label_str}")
            logger.info(f"    Source: {stats['source_file']}")
    
    logger.info("=" * 80)
    
    return combined_df


def validate_combined_data(df: pd.DataFrame) -> dict:
    """
    Validate combined data and return quality metrics.

    Args:
        df: Combined DataFrame

    Returns:
        Dictionary with validation results
    """
    validation = {
        "status": "PASS",
        "warnings": [],
        "metrics": {},
    }

    # Check minimum sample size
    if len(df) < 30:
        validation["warnings"].append(f"Small sample size: {len(df)} windows (recommend >50)")
        validation["status"] = "WARNING"
    
    # Check participant count
    if "user_id" in df.columns:
        n_participants = df["user_id"].nunique()
        validation["metrics"]["n_participants"] = n_participants
        
        if n_participants < 3:
            validation["warnings"].append(f"Very few participants: {n_participants} (recommend ≥3)")
            validation["status"] = "WARNING"
    
    # Check label balance
    if "label" in df.columns:
        label_counts = df["label"].value_counts()
        validation["metrics"]["label_distribution"] = label_counts.to_dict()
        
        if len(label_counts) < 2:
            validation["warnings"].append("Only one label present - cannot train classifier")
            validation["status"] = "FAIL"
        else:
            min_count = label_counts.min()
            if min_count < 5:
                validation["warnings"].append(f"Minimum class has only {min_count} samples (recommend ≥10)")
                validation["status"] = "WARNING"
    
    # Check for missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    nan_ratio = df[numeric_cols].isna().sum() / len(df)
    high_nan_cols = nan_ratio[nan_ratio > 0.1].index.tolist()
    
    if high_nan_cols:
        validation["warnings"].append(f"High NaN ratio in: {high_nan_cols}")
        validation["status"] = "WARNING"
    
    # Check quality metrics if available
    if "valid_frame_ratio" in df.columns:
        mean_vfr = df["valid_frame_ratio"].mean()
        validation["metrics"]["mean_valid_frame_ratio"] = mean_vfr
        
        if mean_vfr < 0.80:
            validation["warnings"].append(f"Low mean valid_frame_ratio: {mean_vfr:.2f} (target >0.80)")
            validation["status"] = "WARNING"
    
    if "mean_quality" in df.columns:
        mean_qual = df["mean_quality"].mean()
        validation["metrics"]["mean_quality"] = mean_qual
        
        if mean_qual < 0.85:
            validation["warnings"].append(f"Low mean quality: {mean_qual:.2f} (target >0.85)")
            validation["status"] = "WARNING"
    
    return validation


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Combine participant feature CSV files"
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Input CSV files or glob pattern (e.g., 'participant_*_features.csv')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/all_participants_features.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip data verification",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run additional validation checks",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_dir="logs", log_file="combine_features.log")
    
    logger.info("=" * 80)
    logger.info("Combining participant feature files")
    logger.info("=" * 80)

    # Find input files
    input_files = []
    for pattern in args.input:
        if "*" in pattern or "?" in pattern:
            # Glob pattern
            matched_files = find_participant_files(pattern)
            input_files.extend(matched_files)
        else:
            # Direct file path
            file_path = Path(pattern)
            if file_path.exists():
                input_files.append(file_path)
            else:
                logger.warning(f"File not found: {file_path}")

    if not input_files:
        logger.error("No input files found")
        sys.exit(1)

    logger.info(f"Found {len(input_files)} input files")

    try:
        # Combine files
        combined_df = combine_features(
            input_files=input_files,
            output_file=args.output,
            verify=not args.no_verify,
        )

        # Run validation if requested
        if args.validate:
            logger.info("=" * 80)
            logger.info("Running validation checks...")
            logger.info("=" * 80)
            
            validation = validate_combined_data(combined_df)
            
            logger.info(f"Validation status: {validation['status']}")
            
            if validation["metrics"]:
                logger.info("\nMetrics:")
                for key, value in validation["metrics"].items():
                    logger.info(f"  {key}: {value}")
            
            if validation["warnings"]:
                logger.info("\nWarnings:")
                for warning in validation["warnings"]:
                    logger.warning(f"  ⚠️  {warning}")
            
            logger.info("=" * 80)
            
            # Exit with appropriate code
            if validation["status"] == "FAIL":
                sys.exit(2)
            elif validation["status"] == "WARNING":
                sys.exit(1)

        logger.info("Successfully combined participant features!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error combining features: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

