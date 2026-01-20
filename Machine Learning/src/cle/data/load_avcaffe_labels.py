"""
AVCAffe dataset label loading utilities.

Parses AVCAffe ground truth labels (mental demand scores) and normalizes
them to continuous [0,1] scale for regression training.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.cle.logging_setup import get_logger

logger = get_logger(__name__)


DEFAULT_LABELS_PATH = "E:/FYP/Dataset/AVCAffe/codes/downloader/data/ground_truths/mental_demand.txt"


def load_mental_demand_labels(
    labels_path: Optional[str] = None,
    max_score: float = 21.0,
) -> pd.DataFrame:
    """
    Load and normalize AVCAffe mental demand labels.

    Parses the mental_demand.txt file which contains NASA-TLX mental demand
    subscale scores (0-21) and normalizes them to [0,1] for continuous regression.

    File format example:
        aiim001_task_1, 1.0
        aiim001_task_2, 1.0
        aiim001_task_3, 12.0

    Args:
        labels_path: Path to mental_demand.txt. If None, uses default path.
        max_score: Maximum score for normalization (default: 21.0 for NASA-TLX)

    Returns:
        DataFrame with columns:
            - participant_id: str (e.g., "aiim001")
            - task: str (e.g., "task_1")
            - raw_score: float (0-21)
            - cognitive_load: float (normalized to 0-1)

    Raises:
        FileNotFoundError: If labels file doesn't exist
        ValueError: If parsing fails or invalid scores found
    """
    if labels_path is None:
        labels_path = DEFAULT_LABELS_PATH

    labels_file = Path(labels_path)
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    logger.info(f"Loading mental demand labels from: {labels_path}")

    # Parse the file
    data = []
    errors = []

    with open(labels_file, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                # Parse format: "aiim001_task_1, 1.0"
                parts = line.split(",")
                if len(parts) != 2:
                    errors.append(f"Line {line_num}: Invalid format (expected 'id_task, score')")
                    continue

                participant_task = parts[0].strip()
                score_str = parts[1].strip()

                # Split participant_task into participant_id and task
                if "_task_" not in participant_task:
                    errors.append(f"Line {line_num}: Invalid format (expected 'participantid_task_N')")
                    continue

                participant_id, task_part = participant_task.split("_task_")
                task = f"task_{task_part}"

                # Parse score
                try:
                    raw_score = float(score_str)
                except ValueError:
                    errors.append(f"Line {line_num}: Invalid score '{score_str}'")
                    continue

                # Validate score range
                if not (0 <= raw_score <= max_score):
                    errors.append(
                        f"Line {line_num}: Score {raw_score} out of range [0, {max_score}]"
                    )
                    continue

                # Normalize to [0, 1]
                cognitive_load = raw_score / max_score

                data.append({
                    "participant_id": participant_id,
                    "task": task,
                    "raw_score": raw_score,
                    "cognitive_load": cognitive_load,
                })

            except Exception as e:
                errors.append(f"Line {line_num}: Unexpected error: {e}")
                continue

    # Log parsing errors
    if errors:
        logger.warning(f"Found {len(errors)} parsing errors:")
        for error in errors[:10]:  # Show first 10 errors
            logger.warning(f"  {error}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more errors")

    if not data:
        raise ValueError("No valid labels parsed from file")

    # Create DataFrame
    df = pd.DataFrame(data)

    # Validation checks
    logger.info(f"Loaded {len(df)} labels")
    logger.info(f"Unique participants: {df['participant_id'].nunique()}")
    logger.info(f"Unique tasks: {df['task'].nunique()}")

    # Check for duplicates
    duplicates = df.duplicated(subset=["participant_id", "task"]).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate participant-task combinations")
        df = df.drop_duplicates(subset=["participant_id", "task"], keep="first")
        logger.info(f"Kept first occurrence, remaining: {len(df)} labels")

    # Validate normalization
    assert df["cognitive_load"].min() >= 0.0, "cognitive_load minimum < 0"
    assert df["cognitive_load"].max() <= 1.0, "cognitive_load maximum > 1"

    logger.info("Label statistics:")
    logger.info(f"  Raw scores - min: {df['raw_score'].min():.1f}, "
                f"max: {df['raw_score'].max():.1f}, "
                f"mean: {df['raw_score'].mean():.2f}")
    logger.info(f"  Cognitive load - min: {df['cognitive_load'].min():.3f}, "
                f"max: {df['cognitive_load'].max():.3f}, "
                f"mean: {df['cognitive_load'].mean():.3f}")

    # Validate participant ID format
    invalid_ids = df[~df["participant_id"].str.match(r"^aiim\d+$")]
    if len(invalid_ids) > 0:
        logger.warning(f"Found {len(invalid_ids)} participants with unexpected ID format:")
        logger.warning(f"  {invalid_ids['participant_id'].unique().tolist()}")

    return df


def get_label_for_participant_task(
    labels_df: pd.DataFrame,
    participant_id: str,
    task: str,
) -> Optional[float]:
    """
    Get cognitive load label for specific participant and task.

    Args:
        labels_df: DataFrame from load_mental_demand_labels()
        participant_id: Participant ID (e.g., "aiim001")
        task: Task name (e.g., "task_1")

    Returns:
        Cognitive load value [0,1] or None if not found
    """
    mask = (labels_df["participant_id"] == participant_id) & (labels_df["task"] == task)
    matches = labels_df.loc[mask, "cognitive_load"]

    if len(matches) == 0:
        return None
    elif len(matches) == 1:
        return matches.iloc[0]
    else:
        logger.warning(
            f"Multiple labels found for {participant_id}/{task}, using first"
        )
        return matches.iloc[0]


if __name__ == "__main__":
    # Test the label loader
    import sys
    from src.cle.logging_setup import setup_logging

    setup_logging(level="INFO")

    labels_path = None
    if len(sys.argv) > 1:
        labels_path = sys.argv[1]

    df = load_mental_demand_labels(labels_path)

    print(f"\nLoaded {len(df)} labels")
    print(f"\nFirst 10 rows:")
    print(df.head(10))

    print(f"\nLabel distribution:")
    print(df["cognitive_load"].describe())

    print(f"\nParticipant-task coverage:")
    print(f"  Participants: {df['participant_id'].nunique()}")
    print(f"  Tasks per participant: {df.groupby('participant_id').size().describe()}")
