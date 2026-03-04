"""
FastAPI server for cognitive load prediction.

Provides REST API endpoints for real-time cognitive load estimation.
"""

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import os
import re
import secrets
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.cle.api import load_model, predict_window
from src.cle.config import load_config
from src.cle.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

# Global state for model artifacts
artifacts: Optional[Dict] = None
config: Optional[Dict] = None

SAFE_SEGMENT_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def parse_iso8601(value: str) -> datetime:
    """Parse an ISO timestamp into a timezone-aware UTC datetime."""
    cleaned = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_storage_segment(raw: str, label: str) -> str:
    """Normalise a user-provided segment for safe filesystem use."""
    safe = SAFE_SEGMENT_PATTERN.sub("-", raw.strip())
    safe = safe.strip("-")
    if not safe:
        raise HTTPException(status_code=400, detail=f"Invalid {label}")
    return safe[:128]


def get_reports_root() -> Path:
    """Return persistent reports root, creating it on demand."""
    root = Path(os.environ.get("CLE_REPORTS_DIR", "data/reports"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_activity_root() -> Path:
    """Return persistent activity-events root, creating it on demand."""
    root = get_reports_root() / "activity"
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON payload to disk with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json_file(path: Path) -> Dict[str, Any]:
    """Read a JSON file from disk."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Expected JSON object")
    return loaded


def configured_allowed_origins() -> List[str]:
    """Resolve allowed CORS origins from env."""
    raw = os.environ.get("CLE_ALLOWED_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def require_admin_token(authorization: Optional[str]) -> None:
    """Validate admin bearer token for protected endpoints."""
    configured_token = os.environ.get("CLE_ADMIN_TOKEN", "").strip()
    if not configured_token:
        raise HTTPException(status_code=503, detail="Admin token is not configured")

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Authorization must be a Bearer token")

    provided = authorization[len(prefix):].strip()
    if not provided or not secrets.compare_digest(provided, configured_token):
        raise HTTPException(status_code=401, detail="Invalid admin token")


def record_event_time(record: Dict[str, Any]) -> Optional[datetime]:
    """Pick the most relevant event timestamp for filtering/indexing."""
    for key in ("completedAtIso", "startedAtIso", "dueAtIso"):
        value = record.get(key)
        if isinstance(value, str) and value:
            try:
                return parse_iso8601(value)
            except ValueError:
                continue
    return None


def derive_learning_items(record: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Derive easy/hard learning items from a stored session record."""
    trials = record.get("trials")
    if not isinstance(trials, list):
        return {"easy": [], "hard": []}

    unique_items: Dict[str, Dict[str, Any]] = {}
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        if trial.get("kind") != "learning":
            continue

        item_id = trial.get("itemId")
        if not isinstance(item_id, str) or not item_id:
            continue
        if item_id in unique_items:
            continue

        interference_group = "-".join(item_id.split("-")[:2]) if "-" in item_id else item_id
        unique_items[item_id] = {
            "id": item_id,
            "cue": trial.get("cue", ""),
            "target": trial.get("target", ""),
            "difficulty": trial.get("difficulty", ""),
            "interferenceGroup": interference_group,
        }

    items = list(unique_items.values())
    easy = [item for item in items if item.get("difficulty") == "easy"]
    hard = [item for item in items if item.get("difficulty") == "hard"]
    return {"easy": easy, "hard": hard}


def ensure_required_fields(record: Dict[str, Any], required: List[str], label: str) -> None:
    """Validate record has required fields."""
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise HTTPException(status_code=400, detail=f"{label} missing required fields: {', '.join(missing)}")


def iter_record_files(kind: Literal["sessions", "delayed"], participant_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load stored report records from disk with metadata."""
    root = get_reports_root() / kind
    if not root.exists():
        return []

    participants: List[Path]
    if participant_id:
        participant_safe = normalize_storage_segment(participant_id, "participant_id")
        participant_dir = root / participant_safe
        participants = [participant_dir] if participant_dir.exists() else []
    else:
        participants = [path for path in root.iterdir() if path.is_dir()]

    loaded: List[Dict[str, Any]] = []
    for participant_dir in participants:
        for json_file in sorted(participant_dir.glob("*.json")):
            try:
                payload = read_json_file(json_file)
            except Exception as err:
                logger.warning(f"Skipping invalid JSON file {json_file}: {err}")
                continue

            event_time = record_event_time(payload)
            stored_at_iso = datetime.fromtimestamp(json_file.stat().st_mtime, tz=timezone.utc).isoformat()
            loaded.append(
                {
                    "kind": kind,
                    "participant_id": participant_dir.name,
                    "record_id": json_file.stem,
                    "path": json_file,
                    "event_time": event_time,
                    "event_time_iso": event_time.isoformat() if event_time else None,
                    "stored_at_iso": stored_at_iso,
                    "payload": payload,
                }
            )
    return loaded


def filter_records(
    records: List[Dict[str, Any]],
    from_dt: Optional[datetime],
    to_dt: Optional[datetime],
) -> List[Dict[str, Any]]:
    """Apply optional event-time filters to report records."""
    if not from_dt and not to_dt:
        return records

    filtered: List[Dict[str, Any]] = []
    for record in records:
        event_time = record.get("event_time")
        if event_time is None:
            continue
        if from_dt and event_time < from_dt:
            continue
        if to_dt and event_time > to_dt:
            continue
        filtered.append(record)
    return filtered


def parse_time_filter(label: str, value: Optional[str]) -> Optional[datetime]:
    """Parse optional query-time filter."""
    if not value:
        return None
    try:
        return parse_iso8601(value)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=f"Invalid {label}: {err}") from err


def append_activity_event(payload: Dict[str, Any]) -> None:
    """Append an activity event to a newline-delimited JSON log file."""
    filename = f"{datetime.now(timezone.utc):%Y-%m-%d}.ndjson"
    path = get_activity_root() / filename
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _file_date_str(path: Path) -> str:
    """Extract YYYY-MM-DD from an NDJSON filename stem."""
    return path.stem


def iter_activity_events(
    from_dt: Optional[datetime] = None,
    to_dt: Optional[datetime] = None,
    max_events: int = 20000,
) -> List[tuple]:
    """Load activity events from NDJSON files with optional time filtering.

    Returns a list of ``(event_dict, occurred_at_datetime)`` tuples.
    """
    root = get_activity_root()
    if not root.exists():
        return []

    from_date_str = from_dt.strftime("%Y-%m-%d") if from_dt else None
    to_date_str = to_dt.strftime("%Y-%m-%d") if to_dt else None

    loaded: List[tuple] = []
    for event_file in sorted(root.glob("*.ndjson")):
        file_date = _file_date_str(event_file)
        if from_date_str and file_date < from_date_str:
            continue
        if to_date_str and file_date > to_date_str:
            continue

        try:
            with event_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue

                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(payload, dict):
                        continue

                    occurred_at_iso = payload.get("occurred_at_iso")
                    if not isinstance(occurred_at_iso, str) or not occurred_at_iso:
                        continue

                    try:
                        occurred_at = parse_iso8601(occurred_at_iso)
                    except ValueError:
                        continue

                    if from_dt and occurred_at < from_dt:
                        continue
                    if to_dt and occurred_at > to_dt:
                        continue

                    loaded.append((payload, occurred_at))
        except OSError as err:
            logger.warning(f"Skipping unreadable activity file {event_file}: {err}")

    if len(loaded) > max_events:
        loaded = loaded[-max_events:]
    return loaded


class PredictionRequest(BaseModel):
    """Request for cognitive load prediction."""

    features: Dict[str, float] = Field(..., description="Feature map (name -> float)")


class PredictionResponse(BaseModel):
    """Response with cognitive load prediction."""

    cli: float = Field(..., description="Cognitive Load Index (0-1)")
    success: bool = Field(..., description="Whether prediction succeeded")
    message: Optional[str] = Field(None, description="Optional message")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    feature_count: Optional[int] = None


class ModelInfoResponse(BaseModel):
    """Model information response."""

    features: List[str]
    n_features: int
    task_mode: str
    metadata: Dict[str, Any]


class TrainingSample(BaseModel):
    """A single training sample from data collection."""

    timestamp: int = Field(..., description="Collection timestamp (ms)")
    window_index: int = Field(..., description="Window index in session")
    label: str = Field(..., description="Cognitive load label (low/high)")
    difficulty: str = Field(..., description="Task difficulty")
    task_type: str = Field(..., description="Type of task")
    features: Dict[str, float]
    valid_frame_ratio: float = Field(..., description="Ratio of valid frames")


class TrainingDataRequest(BaseModel):
    """Request to save collected training data."""

    participant_id: str = Field(..., description="Participant identifier")
    session_notes: Optional[str] = Field(None, description="Optional session notes")
    samples: List[TrainingSample] = Field(..., description="Collected samples")


class TrainingDataResponse(BaseModel):
    """Response after saving training data."""

    success: bool
    filename: str
    samples_saved: int
    message: Optional[str] = None


class StudyParticipantResponse(BaseModel):
    """Generated participant identifier response."""

    participant_id: str
    created_at_iso: str


class StudySessionUploadRequest(BaseModel):
    """Session record upload request."""

    record: Dict[str, Any]


class StudyDelayedUploadRequest(BaseModel):
    """Delayed record upload request."""

    record: Dict[str, Any]


class StudyUploadResponse(BaseModel):
    """Record upload response."""

    success: bool
    record_id: str
    stored_at_iso: str


class StudyActivityRequest(BaseModel):
    """Client-side activity event payload."""

    event_type: str = Field(..., min_length=1, max_length=64)
    page: str = Field(..., min_length=1, max_length=128)
    participant_id: Optional[str] = Field(None, max_length=128)
    visitor_id: Optional[str] = Field(None, max_length=128)
    session_number: Optional[int] = None
    condition: Optional[str] = Field(None, max_length=32)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudyActivityResponse(BaseModel):
    """Activity ingest response."""

    success: bool
    stored_at_iso: str


class PendingDelayedTask(BaseModel):
    """Pending delayed test descriptor."""

    linked_session_record_id: str
    participant_id: str
    session_number: int
    condition: str
    form: str
    due_at_iso: str
    easy_items: List[Dict[str, Any]]
    hard_items: List[Dict[str, Any]]


class PendingDelayedResponse(BaseModel):
    """Pending delayed tests for a participant."""

    participant_id: str
    pending: List[PendingDelayedTask]


class ReportMetadata(BaseModel):
    """Metadata about a stored report record."""

    participant_id: str
    kind: str
    record_id: str
    event_time_iso: Optional[str]
    stored_at_iso: str
    path: str


class ReportIndexResponse(BaseModel):
    """Admin report index response."""

    generated_at_iso: str
    count: int
    records: List[ReportMetadata]


class AdminMonitoringRecentRecord(BaseModel):
    """Recent report-level activity entry for admin dashboard."""

    participant_id: str
    kind: str
    record_id: str
    condition: Optional[str] = None
    session_number: Optional[int] = None
    stored_at_iso: str


class AdminMonitoringDailyUpload(BaseModel):
    """Daily uploaded-record counts."""

    date: str
    session_records: int
    delayed_records: int
    total_records: int


class AdminMonitoringActivityEvent(BaseModel):
    """Recent client activity event."""

    occurred_at_iso: str
    event_type: str
    page: str
    participant_id: Optional[str] = None
    visitor_id: Optional[str] = None
    session_number: Optional[int] = None
    condition: Optional[str] = None


class AdminMonitoringActivitySummary(BaseModel):
    """Aggregate activity metrics for admin dashboard."""

    active_last_15m: int
    active_last_60m: int
    visitors_last_24h: int
    page_views_last_24h: int
    page_view_counts: Dict[str, int]
    recent_events: List[AdminMonitoringActivityEvent]


class AdminMonitoringSummaryResponse(BaseModel):
    """High-level monitoring summary for reports and site activity."""

    generated_at_iso: str
    totals: Dict[str, int]
    condition_counts: Dict[str, int]
    intervention_counts: Dict[str, int]
    daily_uploads: List[AdminMonitoringDailyUpload]
    recent_records: List[AdminMonitoringRecentRecord]
    activity: AdminMonitoringActivitySummary


# Create FastAPI app
app = FastAPI(
    title="Cognitive Load Estimation API",
    description="REST API for real-time cognitive load prediction from facial features",
    version="1.0.0",
)

allowed_origins = configured_allowed_origins()
allow_wildcard = allowed_origins == ["*"]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=not allow_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup."""
    global artifacts, config

    logger.info("=" * 80)
    logger.info("Starting Cognitive Load Estimation API")
    logger.info("=" * 80)

    # Load configuration (will use default if not specified)
    try:
        config = load_config(None)
        logger.info(f"Loaded configuration (hash: {config.hash()[:8]})")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Load model artifacts from models directory (env override supported)
    models_dir_str = os.environ.get("CLE_MODELS_DIR", "models/video_physio_regression_z01_geom")
    models_dir = Path(models_dir_str)
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        sys.exit(1)

    try:
        artifacts = load_model(str(models_dir))
        logger.info(f"Loaded model artifacts with {len(artifacts['feature_spec']['features'])} features")
        logger.info(f"Features: {', '.join(artifacts['feature_spec']['features'])}")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        sys.exit(1)

    logger.info("API ready to accept requests")
    logger.info("-" * 80)


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint."""
    return {
        "service": "Cognitive Load Estimation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Make cognitive load prediction",
            "GET /health": "Health check",
            "GET /model-info": "Model information (legacy)",
            "POST /save-training-data": "Save training samples (legacy, admin token required)",
            "POST /study/participants": "Generate participant identifier",
            "POST /study/session-records": "Upload completed session JSON",
            "POST /study/delayed-records": "Upload completed delayed test JSON",
            "POST /study/activity": "Log non-sensitive site usage event",
            "GET /study/pending-delayed/{participant_id}": "Fetch pending delayed tasks",
            "GET /admin/reports/index": "List stored reports (admin token required)",
            "GET /admin/reports/export": "Export stored reports (admin token required)",
            "GET /admin/monitoring/summary": "Report + usage dashboard metrics (admin token required)",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = artifacts is not None
    feature_count = len(artifacts["feature_spec"]["features"]) if model_loaded else None

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        feature_count=feature_count,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        features=artifacts["feature_spec"]["features"],
        n_features=artifacts["feature_spec"]["n_features"],
        task_mode=artifacts.get("task_mode", "classification"),
        metadata=artifacts.get("calibration") or {},
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_cognitive_load(request: PredictionRequest):
    """
    Predict cognitive load from window features.

    Args:
        request: Window features

    Returns:
        Cognitive Load Index
    """
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert features to dict
        features_dict = request.features

        # Log received features
        logger.debug(f"Received features: {features_dict}")

        # Make prediction
        cli = predict_window(features_dict, artifacts)

        logger.info(f"Prediction: CLI={cli:.3f}")

        return PredictionResponse(
            cli=cli,
            success=True,
            message=None,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/save-training-data", response_model=TrainingDataResponse)
async def save_training_data(
    request: TrainingDataRequest,
    authorization: Optional[str] = Header(None),
):
    require_admin_token(authorization)
    """
    Save collected training data to CSV file.

    This endpoint saves labeled training samples collected from the
    Data Collection mode in the frontend.

    Args:
        request: Training data with participant info and samples

    Returns:
        Confirmation with filename and sample count
    """
    if not request.samples:
        raise HTTPException(status_code=400, detail="No samples provided")

    # Create output directory
    output_dir = Path("data/collected")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"collected_{request.participant_id}_{timestamp}.csv"
    filepath = output_dir / filename

    try:
        # Define feature columns (must match model training order)
        feature_columns = [
            "blink_rate", "blink_count", "mean_blink_duration", "ear_std",
            "perclos",
            "mouth_open_mean", "mouth_open_std", "roll_std",
            "pitch_std", "yaw_std",
            "motion_mean", "motion_std",
        ]

        # Write CSV
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = [
                "user_id", "timestamp", "window_index", "label", "difficulty",
                "task_type", *feature_columns, "role"
            ]
            writer.writerow(header)

            # Data rows
            for sample in request.samples:
                features_dict = sample.features
                row = [
                    request.participant_id,
                    sample.timestamp,
                    sample.window_index,
                    1 if sample.label == "high" else 0,
                    sample.difficulty,
                    sample.task_type,
                    *[features_dict.get(col, 0.0) for col in feature_columns],
                    "train"
                ]
                writer.writerow(row)

        logger.info(
            f"Saved {len(request.samples)} training samples "
            f"for participant {request.participant_id} to {filepath}"
        )

        # Also save session metadata as JSON sidecar
        metadata_path = filepath.with_suffix(".json")
        metadata = {
            "participant_id": request.participant_id,
            "session_notes": request.session_notes,
            "collection_date": datetime.now().isoformat(),
            "total_samples": len(request.samples),
            "samples_by_label": {
                "low": sum(1 for s in request.samples if s.label == "low"),
                "high": sum(1 for s in request.samples if s.label == "high"),
            },
            "samples_by_difficulty": {
                "easy": sum(1 for s in request.samples if s.difficulty == "easy"),
                "medium": sum(1 for s in request.samples if s.difficulty == "medium"),
                "hard": sum(1 for s in request.samples if s.difficulty == "hard"),
            }
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return TrainingDataResponse(
            success=True,
            filename=filename,
            samples_saved=len(request.samples),
            message=f"Data saved to {filepath}"
        )

    except Exception as e:
        logger.error(f"Failed to save training data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")


@app.post("/study/participants", response_model=StudyParticipantResponse)
async def create_study_participant():
    """Generate a participant identifier for public study flow."""
    created_at_iso = utc_now_iso()
    suffix = secrets.token_hex(3).upper()
    participant_id = f"P-{datetime.now(timezone.utc):%y%m%d}-{suffix}"
    return StudyParticipantResponse(participant_id=participant_id, created_at_iso=created_at_iso)


@app.post("/study/session-records", response_model=StudyUploadResponse)
async def upload_session_record(request: StudySessionUploadRequest):
    """Persist a session record JSON payload."""
    record = request.record
    ensure_required_fields(
        record,
        ["recordId", "participantId", "sessionNumber", "condition", "form", "startedAtIso"],
        "Session record",
    )

    participant_id = normalize_storage_segment(str(record["participantId"]), "participant_id")
    record_id = normalize_storage_segment(str(record["recordId"]), "record_id")

    output_path = get_reports_root() / "sessions" / participant_id / f"{record_id}.json"
    stored_at_iso = utc_now_iso()

    payload = dict(record)
    payload["_serverStoredAtIso"] = stored_at_iso
    write_json_file(output_path, payload)

    logger.info(f"Stored session record: {output_path}")
    return StudyUploadResponse(success=True, record_id=record_id, stored_at_iso=stored_at_iso)


@app.post("/study/delayed-records", response_model=StudyUploadResponse)
async def upload_delayed_record(request: StudyDelayedUploadRequest):
    """Persist a delayed test record JSON payload."""
    record = request.record
    ensure_required_fields(
        record,
        [
            "recordId",
            "participantId",
            "linkedSessionRecordId",
            "sessionNumber",
            "condition",
            "form",
            "dueAtIso",
        ],
        "Delayed record",
    )

    participant_id = normalize_storage_segment(str(record["participantId"]), "participant_id")
    record_id = normalize_storage_segment(str(record["recordId"]), "record_id")

    output_path = get_reports_root() / "delayed" / participant_id / f"{record_id}.json"
    stored_at_iso = utc_now_iso()

    payload = dict(record)
    payload["_serverStoredAtIso"] = stored_at_iso
    write_json_file(output_path, payload)

    logger.info(f"Stored delayed record: {output_path}")
    return StudyUploadResponse(success=True, record_id=record_id, stored_at_iso=stored_at_iso)


@app.post("/study/activity", response_model=StudyActivityResponse)
async def track_study_activity(
    request: StudyActivityRequest,
    user_agent: Optional[str] = Header(None),
):
    """Store lightweight client activity events for monitoring dashboard."""
    stored_at_iso = utc_now_iso()
    participant_id = (
        normalize_storage_segment(request.participant_id, "participant_id")
        if request.participant_id
        else None
    )
    visitor_id = (
        normalize_storage_segment(request.visitor_id, "visitor_id")
        if request.visitor_id
        else None
    )
    event_type = normalize_storage_segment(request.event_type, "event_type")
    page = request.page.strip()[:128]
    condition = request.condition.strip()[:32] if request.condition else None
    session_number = request.session_number if isinstance(request.session_number, int) else None
    metadata = request.metadata if isinstance(request.metadata, dict) else {}

    # Keep event payload bounded to avoid large writes.
    if len(json.dumps(metadata)) > 2048:
        metadata = {"_truncated": True}

    payload = {
        "occurred_at_iso": stored_at_iso,
        "event_type": event_type,
        "page": page or "unknown",
        "participant_id": participant_id,
        "visitor_id": visitor_id,
        "session_number": session_number,
        "condition": condition,
        "metadata": metadata,
        "user_agent": (user_agent or "")[:256],
    }
    append_activity_event(payload)

    return StudyActivityResponse(success=True, stored_at_iso=stored_at_iso)


@app.get("/study/pending-delayed/{participant_id}", response_model=PendingDelayedResponse)
async def get_pending_delayed(participant_id: str):
    """Return pending delayed tasks for a participant using stored server records."""
    participant_safe = normalize_storage_segment(participant_id, "participant_id")

    session_records = iter_record_files("sessions", participant_safe)
    delayed_records = iter_record_files("delayed", participant_safe)

    completed_links = {
        str(entry["payload"].get("linkedSessionRecordId", ""))
        for entry in delayed_records
        if isinstance(entry.get("payload"), dict)
    }

    pending: List[PendingDelayedTask] = []
    for entry in session_records:
        payload = entry["payload"]
        record_id = str(payload.get("recordId", ""))
        if not record_id:
            continue

        pending_flag = bool(payload.get("pendingDelayedTest", True))
        if not pending_flag:
            continue
        if record_id in completed_links:
            continue

        items = derive_learning_items(payload)

        pending.append(
            PendingDelayedTask(
                linked_session_record_id=record_id,
                participant_id=participant_safe,
                session_number=int(payload.get("sessionNumber", 1)),
                condition=str(payload.get("condition", "baseline")),
                form=str(payload.get("form", "A")),
                due_at_iso=str(payload.get("delayedDueAtIso") or utc_now_iso()),
                easy_items=items["easy"],
                hard_items=items["hard"],
            )
        )

    pending.sort(key=lambda task: task.due_at_iso)
    return PendingDelayedResponse(participant_id=participant_safe, pending=pending)


@app.get("/admin/reports/index", response_model=ReportIndexResponse)
async def admin_report_index(
    participant_id: Optional[str] = Query(None),
    from_iso: Optional[str] = Query(None),
    to_iso: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    """List report metadata for admin access."""
    require_admin_token(authorization)

    from_dt = parse_time_filter("from_iso", from_iso)
    to_dt = parse_time_filter("to_iso", to_iso)
    participant_filter = normalize_storage_segment(participant_id, "participant_id") if participant_id else None

    records = iter_record_files("sessions", participant_filter) + iter_record_files("delayed", participant_filter)
    records = filter_records(records, from_dt, to_dt)

    metadata = [
        ReportMetadata(
            participant_id=entry["participant_id"],
            kind=entry["kind"],
            record_id=entry["record_id"],
            event_time_iso=entry["event_time_iso"],
            stored_at_iso=entry["stored_at_iso"],
            path=str(entry["path"]),
        )
        for entry in sorted(records, key=lambda item: item["stored_at_iso"])
    ]

    return ReportIndexResponse(
        generated_at_iso=utc_now_iso(),
        count=len(metadata),
        records=metadata,
    )


@app.get("/admin/monitoring/summary", response_model=AdminMonitoringSummaryResponse)
async def admin_monitoring_summary(authorization: Optional[str] = Header(None)):
    """Aggregate report and site-activity metrics for admin monitoring dashboard."""
    require_admin_token(authorization)

    now = datetime.now(timezone.utc)
    recent_24h = now - timedelta(hours=24)
    recent_60m = now - timedelta(minutes=60)
    recent_15m = now - timedelta(minutes=15)

    session_records = iter_record_files("sessions")
    delayed_records = iter_record_files("delayed")
    all_records = session_records + delayed_records

    participants = {entry["participant_id"] for entry in all_records}
    participants_with_session2 = set()
    participants_with_delayed = {entry["participant_id"] for entry in delayed_records}
    condition_counts: Counter = Counter()
    intervention_counts: Counter = Counter()
    phase_integrity_issue_records = 0

    completed_delayed_links = {
        str(entry["payload"].get("linkedSessionRecordId", ""))
        for entry in delayed_records
        if isinstance(entry.get("payload"), dict)
    }

    pending_delayed_records = 0
    for entry in session_records:
        payload = entry["payload"] if isinstance(entry.get("payload"), dict) else {}
        session_number = payload.get("sessionNumber")
        if session_number == 2:
            participants_with_session2.add(entry["participant_id"])

        condition = str(payload.get("condition", "")).strip().lower() or "unknown"
        condition_counts[condition] += 1

        diagnostics = payload.get("runtimeDiagnostics")
        if isinstance(diagnostics, dict) and diagnostics.get("phaseIntegrityOk") is False:
            phase_integrity_issue_records += 1

        interventions = payload.get("interventions")
        if isinstance(interventions, list):
            for intervention in interventions:
                if not isinstance(intervention, dict):
                    continue
                intervention_type = str(intervention.get("type", "")).strip() or "unknown"
                intervention_counts[intervention_type] += 1

        record_id = str(payload.get("recordId", "")).strip()
        pending_flag = bool(payload.get("pendingDelayedTest", True))
        if pending_flag and record_id and record_id not in completed_delayed_links:
            pending_delayed_records += 1

    uploads_last_24h = 0
    daily_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"session_records": 0, "delayed_records": 0}
    )
    for entry in all_records:
        try:
            stored_at = parse_iso8601(entry["stored_at_iso"])
        except ValueError:
            continue

        if stored_at >= recent_24h:
            uploads_last_24h += 1

        day_key = stored_at.date().isoformat()
        if entry["kind"] == "sessions":
            daily_counts[day_key]["session_records"] += 1
        else:
            daily_counts[day_key]["delayed_records"] += 1

    daily_uploads: List[AdminMonitoringDailyUpload] = []
    for days_ago in range(13, -1, -1):
        day = (now - timedelta(days=days_ago)).date().isoformat()
        session_count = daily_counts[day]["session_records"]
        delayed_count = daily_counts[day]["delayed_records"]
        daily_uploads.append(
            AdminMonitoringDailyUpload(
                date=day,
                session_records=session_count,
                delayed_records=delayed_count,
                total_records=session_count + delayed_count,
            )
        )

    sorted_records = sorted(all_records, key=lambda item: item["stored_at_iso"], reverse=True)
    recent_records = [
        AdminMonitoringRecentRecord(
            participant_id=entry["participant_id"],
            kind=entry["kind"],
            record_id=entry["record_id"],
            condition=(
                str(entry["payload"].get("condition")).strip()
                if isinstance(entry.get("payload"), dict) and entry["payload"].get("condition") is not None
                else None
            ),
            session_number=(
                int(entry["payload"].get("sessionNumber"))
                if isinstance(entry.get("payload"), dict) and isinstance(entry["payload"].get("sessionNumber"), int)
                else None
            ),
            stored_at_iso=entry["stored_at_iso"],
        )
        for entry in sorted_records[:25]
    ]

    activity_tuples = iter_activity_events(from_dt=recent_24h, max_events=20000)
    activity_tuples.sort(key=lambda pair: pair[1], reverse=True)

    active_keys_15m = set()
    active_keys_60m = set()
    visitor_keys_24h = set()
    page_views_24h = 0
    page_view_counts: Counter = Counter()

    for event, occurred_at in activity_tuples:
        identity = event.get("participant_id") or event.get("visitor_id")
        if identity:
            visitor_keys_24h.add(str(identity))
            if occurred_at >= recent_60m:
                active_keys_60m.add(str(identity))
            if occurred_at >= recent_15m:
                active_keys_15m.add(str(identity))

        page = str(event.get("page", "unknown"))
        page_view_counts[page] += 1
        page_views_24h += 1

    recent_events = [
        AdminMonitoringActivityEvent(
            occurred_at_iso=str(event.get("occurred_at_iso", "")),
            event_type=str(event.get("event_type", "unknown")),
            page=str(event.get("page", "unknown")),
            participant_id=(
                str(event["participant_id"])
                if event.get("participant_id") is not None
                else None
            ),
            visitor_id=(
                str(event["visitor_id"])
                if event.get("visitor_id") is not None
                else None
            ),
            session_number=(
                int(event["session_number"])
                if isinstance(event.get("session_number"), int)
                else None
            ),
            condition=(str(event.get("condition")) if event.get("condition") else None),
        )
        for event, _ in activity_tuples[:50]
    ]

    totals = {
        "total_records": len(all_records),
        "session_records": len(session_records),
        "delayed_records": len(delayed_records),
        "unique_participants": len(participants),
        "participants_with_session2": len(participants_with_session2),
        "participants_with_delayed": len(participants_with_delayed),
        "pending_delayed_records": pending_delayed_records,
        "uploads_last_24h": uploads_last_24h,
        "phase_integrity_issue_records": phase_integrity_issue_records,
    }

    activity_summary = AdminMonitoringActivitySummary(
        active_last_15m=len(active_keys_15m),
        active_last_60m=len(active_keys_60m),
        visitors_last_24h=len(visitor_keys_24h),
        page_views_last_24h=page_views_24h,
        page_view_counts=dict(page_view_counts),
        recent_events=recent_events,
    )

    return AdminMonitoringSummaryResponse(
        generated_at_iso=utc_now_iso(),
        totals=totals,
        condition_counts=dict(condition_counts),
        intervention_counts=dict(intervention_counts),
        daily_uploads=daily_uploads,
        recent_records=recent_records,
        activity=activity_summary,
    )


@app.get("/admin/reports/export")
async def admin_report_export(
    participant_id: Optional[str] = Query(None),
    from_iso: Optional[str] = Query(None),
    to_iso: Optional[str] = Query(None),
    format: Literal["zip", "json"] = Query("zip"),
    authorization: Optional[str] = Header(None),
):
    """Export report payloads in JSON bundle or ZIP format."""
    require_admin_token(authorization)

    from_dt = parse_time_filter("from_iso", from_iso)
    to_dt = parse_time_filter("to_iso", to_iso)
    participant_filter = normalize_storage_segment(participant_id, "participant_id") if participant_id else None

    records = iter_record_files("sessions", participant_filter) + iter_record_files("delayed", participant_filter)
    records = filter_records(records, from_dt, to_dt)
    records = sorted(records, key=lambda item: item["stored_at_iso"])

    if format == "json":
        payload = {
            "generated_at_iso": utc_now_iso(),
            "count": len(records),
            "records": [
                {
                    "participant_id": entry["participant_id"],
                    "kind": entry["kind"],
                    "record_id": entry["record_id"],
                    "event_time_iso": entry["event_time_iso"],
                    "stored_at_iso": entry["stored_at_iso"],
                    "payload": entry["payload"],
                }
                for entry in records
            ],
        }
        return JSONResponse(content=payload)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        manifest = {
            "generated_at_iso": utc_now_iso(),
            "count": len(records),
            "filters": {
                "participant_id": participant_filter,
                "from_iso": from_iso,
                "to_iso": to_iso,
            },
        }
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))

        for entry in records:
            relative_name = f"{entry['kind']}/{entry['participant_id']}/{entry['record_id']}.json"
            archive.writestr(relative_name, json.dumps(entry["payload"], indent=2))

    buffer.seek(0)
    filename = f"study_reports_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.zip"
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def main():
    """Main entry point for running the server."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Cognitive Load Estimation API Server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Path to model artifacts directory (overrides CLE_MODELS_DIR).",
    )

    args = parser.parse_args()
    if args.models_dir:
        os.environ["CLE_MODELS_DIR"] = args.models_dir

    # Setup logging
    setup_logging(level=args.log_level, log_dir="logs", log_file="server.log")

    # Run server
    uvicorn.run(
        "src.cle.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
