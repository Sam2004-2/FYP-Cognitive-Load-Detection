"""
FastAPI server for cognitive load prediction.

Provides REST API endpoints for real-time cognitive load estimation.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.cle.api import load_model, predict_window
from src.cle.config import load_config
from src.cle.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

# Global state for model artifacts
artifacts: Optional[Dict] = None
config: Optional[Dict] = None


class WindowFeatures(BaseModel):
    """Window-level features for prediction."""

    blink_rate: float = Field(..., description="Blinks per minute")
    blink_count: float = Field(..., description="Total blinks in window")
    mean_blink_duration: float = Field(..., description="Mean blink duration (ms)")
    ear_std: float = Field(..., description="EAR standard deviation")
    mean_brightness: float = Field(..., description="Mean face brightness")
    std_brightness: float = Field(..., description="Brightness standard deviation")
    perclos: float = Field(..., description="Percentage of eye closure")
    mean_quality: float = Field(..., description="Mean detection quality")
    valid_frame_ratio: float = Field(..., description="Ratio of valid frames")


class PredictionRequest(BaseModel):
    """Request for cognitive load prediction."""

    features: WindowFeatures


class PredictionResponse(BaseModel):
    """Response with cognitive load prediction."""

    cli: float = Field(..., description="Cognitive Load Index (0-1)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
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
    calibration: Dict


class TrainingSample(BaseModel):
    """A single training sample from data collection."""

    timestamp: int = Field(..., description="Collection timestamp (ms)")
    window_index: int = Field(..., description="Window index in session")
    label: str = Field(..., description="Cognitive load label (low/high)")
    difficulty: str = Field(..., description="Task difficulty")
    task_type: str = Field(..., description="Type of task")
    features: WindowFeatures
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


# Create FastAPI app
app = FastAPI(
    title="Cognitive Load Estimation API",
    description="REST API for real-time cognitive load prediction from facial features",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
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

    # Load model artifacts from default models directory
    models_dir = Path("models/stress_classifier_rf")
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
            "GET /model-info": "Model information",
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
        calibration=artifacts["calibration"],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_cognitive_load(request: PredictionRequest):
    """
    Predict cognitive load from window features.

    Args:
        request: Window features

    Returns:
        Cognitive Load Index and confidence
    """
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert features to dict
        features_dict = request.features.model_dump()

        # Log received features
        logger.debug(f"Received features: {features_dict}")

        # Make prediction
        cli, confidence = predict_window(features_dict, artifacts)

        logger.info(f"Prediction: CLI={cli:.3f}, confidence={confidence:.3f}")

        return PredictionResponse(
            cli=cli,
            confidence=confidence,
            success=True,
            message=None,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/save-training-data", response_model=TrainingDataResponse)
async def save_training_data(request: TrainingDataRequest):
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
            "mean_brightness", "std_brightness", "perclos", "mean_quality",
            "valid_frame_ratio"
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
                features_dict = sample.features.model_dump()
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
        import json
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

    args = parser.parse_args()

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

