"""
Physiological feature extraction from ECG, EDA, and respiration signals.

Uses NeuroKit2 for signal processing and feature extraction.
Extracts windowed features aligned to video feature windows (10s window, 2.5s step).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import neurokit2 as nk
import numpy as np
import pandas as pd


# StressID physiological data parameters
SAMPLING_RATE = 500  # Hz (30,000 samples / ~60s task)
WINDOW_LENGTH_S = 10.0  # seconds (match video windows)
WINDOW_STEP_S = 2.5  # seconds (match video windows)
WINDOW_SAMPLES = int(WINDOW_LENGTH_S * SAMPLING_RATE)  # 5000 samples
STEP_SAMPLES = int(WINDOW_STEP_S * SAMPLING_RATE)  # 1250 samples

# Quality threshold for filtering
QUALITY_THRESHOLD = 0.5


def load_physio_file(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Load physiological data from a StressID .txt file.
    
    Args:
        file_path: Path to the .txt file with ECG, EDA, RR columns
        
    Returns:
        DataFrame with ECG, EDA, RR columns, or None if file doesn't exist
    """
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        # Validate expected columns
        expected_cols = {"ECG", "EDA", "RR"}
        if not expected_cols.issubset(set(df.columns)):
            print(f"Warning: {file_path} missing expected columns. Found: {df.columns.tolist()}")
            return None
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def compute_ecg_features(ecg_signal: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> Dict[str, float]:
    """
    Extract heart rate and HRV features from ECG signal.
    
    Args:
        ecg_signal: Raw ECG signal array
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary with HR, RMSSD, SDNN, and quality metrics
    """
    features = {
        "hr": np.nan,
        "rmssd": np.nan,
        "sdnn": np.nan,
        "ecg_quality": np.nan,
    }
    
    try:
        # Clean and process ECG
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
        
        # Compute signal quality
        quality = nk.ecg_quality(ecg_cleaned, sampling_rate=sampling_rate, method="zhao2018")
        features["ecg_quality"] = float(np.mean(quality))
        
        # Only extract features if quality is acceptable
        if features["ecg_quality"] < QUALITY_THRESHOLD:
            return features
        
        # Find R-peaks
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        r_peak_indices = rpeaks["ECG_R_Peaks"]
        
        if len(r_peak_indices) < 3:
            return features
        
        # Compute RR intervals (in ms)
        rr_intervals = np.diff(r_peak_indices) / sampling_rate * 1000
        
        # Heart rate (bpm)
        features["hr"] = 60000 / np.mean(rr_intervals) if len(rr_intervals) > 0 else np.nan
        
        # HRV time-domain features
        if len(rr_intervals) >= 2:
            features["sdnn"] = float(np.std(rr_intervals, ddof=1))
            
            # RMSSD: root mean square of successive differences
            successive_diffs = np.diff(rr_intervals)
            features["rmssd"] = float(np.sqrt(np.mean(successive_diffs ** 2)))
        
    except Exception as e:
        # Signal processing failed - return NaN features
        pass
    
    return features


def compute_eda_features(eda_signal: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> Dict[str, float]:
    """
    Extract electrodermal activity features from EDA signal.
    
    Args:
        eda_signal: Raw EDA signal array
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary with SCL (skin conductance level) and SCR (response) metrics
    """
    features = {
        "scl": np.nan,
        "scr_count": np.nan,
        "scr_amplitude_mean": np.nan,
    }
    
    try:
        # Process EDA signal
        eda_signals, info = nk.eda_process(eda_signal, sampling_rate=sampling_rate)
        
        # Skin Conductance Level (tonic component) - mean level
        if "EDA_Tonic" in eda_signals.columns:
            features["scl"] = float(eda_signals["EDA_Tonic"].mean())
        
        # Skin Conductance Responses (phasic component)
        scr_peaks = info.get("SCR_Peaks", [])
        features["scr_count"] = float(len(scr_peaks))
        
        # Mean SCR amplitude
        if "SCR_Amplitude" in info and len(info["SCR_Amplitude"]) > 0:
            amplitudes = [a for a in info["SCR_Amplitude"] if not np.isnan(a)]
            if amplitudes:
                features["scr_amplitude_mean"] = float(np.mean(amplitudes))
        
    except Exception as e:
        # Signal processing failed - return NaN features
        pass
    
    return features


def compute_resp_features(resp_signal: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> Dict[str, float]:
    """
    Extract respiration features from RR (respiration) signal.
    
    Args:
        resp_signal: Raw respiration signal array
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary with respiratory rate and variability metrics
    """
    features = {
        "resp_rate": np.nan,
        "resp_amplitude_mean": np.nan,
        "resp_variability": np.nan,
    }
    
    try:
        # Process respiration signal
        rsp_signals, info = nk.rsp_process(resp_signal, sampling_rate=sampling_rate)
        
        # Respiratory rate (breaths per minute)
        if "RSP_Rate" in rsp_signals.columns:
            features["resp_rate"] = float(rsp_signals["RSP_Rate"].mean())
        
        # Breath amplitude
        if "RSP_Amplitude" in rsp_signals.columns:
            amplitudes = rsp_signals["RSP_Amplitude"].dropna()
            if len(amplitudes) > 0:
                features["resp_amplitude_mean"] = float(amplitudes.mean())
                features["resp_variability"] = float(amplitudes.std())
        
    except Exception as e:
        # Signal processing failed - return NaN features
        pass
    
    return features


def extract_window_features(
    physio_df: pd.DataFrame,
    start_sample: int,
    end_sample: int,
    sampling_rate: int = SAMPLING_RATE,
) -> Dict[str, float]:
    """
    Extract all physiological features for a single window.
    
    Args:
        physio_df: DataFrame with ECG, EDA, RR columns
        start_sample: Starting sample index
        end_sample: Ending sample index
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary with all extracted features
    """
    # Extract signal segments
    ecg_segment = physio_df["ECG"].values[start_sample:end_sample]
    eda_segment = physio_df["EDA"].values[start_sample:end_sample]
    rr_segment = physio_df["RR"].values[start_sample:end_sample]
    
    # Compute features from each modality
    ecg_features = compute_ecg_features(ecg_segment, sampling_rate)
    eda_features = compute_eda_features(eda_segment, sampling_rate)
    resp_features = compute_resp_features(rr_segment, sampling_rate)
    
    # Combine all features
    features = {
        **ecg_features,
        **eda_features,
        **resp_features,
    }
    
    return features


def extract_physio_features_for_task(
    physio_file: Path,
    window_length_s: float = WINDOW_LENGTH_S,
    window_step_s: float = WINDOW_STEP_S,
    sampling_rate: int = SAMPLING_RATE,
) -> List[Dict]:
    """
    Extract windowed physiological features for a single task file.
    
    Args:
        physio_file: Path to the physiological data file
        window_length_s: Window length in seconds
        window_step_s: Window step in seconds
        sampling_rate: Sampling rate in Hz
        
    Returns:
        List of dictionaries, each containing features for one window
    """
    physio_df = load_physio_file(physio_file)
    if physio_df is None:
        return []
    
    window_samples = int(window_length_s * sampling_rate)
    step_samples = int(window_step_s * sampling_rate)
    total_samples = len(physio_df)
    
    windows = []
    start_sample = 0
    
    while start_sample + window_samples <= total_samples:
        end_sample = start_sample + window_samples
        
        # Compute time boundaries
        t_start_s = start_sample / sampling_rate
        t_end_s = end_sample / sampling_rate
        
        # Extract features for this window
        features = extract_window_features(physio_df, start_sample, end_sample, sampling_rate)
        
        # Add window timing info
        features["t_start_s"] = t_start_s
        features["t_end_s"] = t_end_s
        
        windows.append(features)
        start_sample += step_samples
    
    return windows


def extract_physio_features_for_participant(
    participant_dir: Path,
    participant_id: str,
    window_length_s: float = WINDOW_LENGTH_S,
    window_step_s: float = WINDOW_STEP_S,
    sampling_rate: int = SAMPLING_RATE,
) -> pd.DataFrame:
    """
    Extract physiological features for all tasks of a participant.
    
    Args:
        participant_dir: Path to participant's physiological data directory
        participant_id: Participant ID (e.g., "2ea4")
        window_length_s: Window length in seconds
        window_step_s: Window step in seconds
        sampling_rate: Sampling rate in Hz
        
    Returns:
        DataFrame with all windowed features for the participant
    """
    all_windows = []
    
    # Find all task files for this participant
    task_files = list(participant_dir.glob(f"{participant_id}_*.txt"))
    
    for task_file in task_files:
        # Extract task name from filename (e.g., "2ea4_Math.txt" -> "Math")
        task_name = task_file.stem.replace(f"{participant_id}_", "")
        
        # Extract features for all windows in this task
        windows = extract_physio_features_for_task(
            task_file, window_length_s, window_step_s, sampling_rate
        )
        
        # Add participant and task info to each window
        for window in windows:
            window["user_id"] = participant_id
            window["task"] = task_name
            all_windows.append(window)
    
    if not all_windows:
        return pd.DataFrame()
    
    return pd.DataFrame(all_windows)


def get_physio_feature_names() -> List[str]:
    """
    Get the list of physiological feature column names.
    
    Returns:
        List of feature column names
    """
    return [
        "hr",
        "rmssd",
        "sdnn",
        "scl",
        "scr_count",
        "scr_amplitude_mean",
        "resp_rate",
        "resp_amplitude_mean",
        "resp_variability",
        "ecg_quality",
    ]
