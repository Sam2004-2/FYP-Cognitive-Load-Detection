# Cognitive Load Estimation - Machine Learning Pipeline

Machine learning pipeline for training cognitive load estimation models from eye-tracking and physiological features.

## Supported Datasets

### StressID Dataset
Training on StressID dataset with physiological features and self-assessment labels.

See existing configurations in `configs/` for details.

### AVCAffe Dataset (New!)

Training on AVCAffe dataset with continuous cognitive load regression from eye-tracking features.

- **Dataset**: 106 participants, 9 tasks each, 132,862 feature windows
- **Labels**: Continuous [0,1] normalized NASA-TLX mental demand scores
- **Features**: Eye-tracking metrics (blink rate, PERCLOS, EAR, brightness, etc.)
- **Model**: Gradient Boosting Regressor with GroupKFold CV

#### Quick Start

```bash
# 1. Data preparation (if needed - already done if using existing data)
python scripts/combine_and_label_avcaffe.py \
    --input data/processed/part1.csv data/processed/part2.csv \
            data/processed/part3.csv data/processed/part4.csv \
    --labels "E:/FYP/Dataset/AVCAffe/codes/downloader/data/ground_truths/mental_demand.txt" \
    --output data/processed/avcaffe_labeled_features.csv \
    --validate

# 2. Validate dataset
python scripts/validate_avcaffe_dataset.py \
    --input data/processed/avcaffe_labeled_features.csv \
    --output-dir reports

# 3. Train regression model
python src/cle/train/train_regression.py \
    --in data/processed/avcaffe_labeled_features.csv \
    --out models/avcaffe_baseline \
    --config configs/avcaffe_regression.yaml
```

#### Documentation

See [docs/AVCAFFE_DATASET.md](docs/AVCAFFE_DATASET.md) for comprehensive documentation including:
- Dataset statistics and label distribution
- Feature extraction methodology
- Data preparation pipeline
- Training configuration
- Expected performance metrics
- Known limitations
- Usage examples

## Project Structure

```
Machine Learning/
├── configs/                    # Training configurations
│   ├── default.yaml           # Base configuration
│   ├── regression.yaml        # StressID regression config
│   └── avcaffe_regression.yaml # AVCAffe regression config (new)
├── data/
│   └── processed/             # Processed feature files (gitignored)
│       ├── part1.csv          # AVCAffe features part 1
│       ├── part2.csv          # AVCAffe features part 2
│       ├── part3.csv          # AVCAffe features part 3
│       ├── part4.csv          # AVCAffe features part 4
│       └── avcaffe_labeled_features.csv  # Combined & labeled
├── docs/                      # Documentation
│   └── AVCAFFE_DATASET.md     # AVCAffe dataset guide
├── models/                    # Model artifacts (gitignored except configs)
│   ├── face_landmarker.task   # MediaPipe model for feature extraction
│   └── avcaffe_baseline/      # Trained AVCAffe models
├── reports/                   # Validation and analysis reports
│   └── avcaffe_validation_*.json
├── scripts/                   # Data processing and utility scripts
│   ├── extract_video_features.py        # AVCAffe feature extraction
│   ├── run_parallel_extraction.ps1      # Parallel extraction
│   ├── combine_and_label_avcaffe.py     # Combine parts & add labels
│   └── validate_avcaffe_dataset.py      # Dataset validation
└── src/cle/                   # Core library
    ├── data/                  # Data loading utilities
    │   ├── load_avcaffe_labels.py   # Load mental demand labels
    │   └── load_avcaffe_data.py     # AVCAffe data adapter
    ├── extract/               # Feature extraction
    ├── train/                 # Training pipelines
    │   ├── train.py           # Classification training
    │   └── train_regression.py # Regression training
    └── utils/                 # Common utilities
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Key Features

- **Multiple Datasets**: Support for StressID and AVCAffe datasets
- **Flexible Models**: Classification and regression tasks
- **Robust CV**: GroupKFold cross-validation to prevent data leakage
- **Quality Filtering**: Automatic filtering of low-quality windows
- **Comprehensive Validation**: Automated dataset validation and reporting
- **Reproducibility**: Fixed random seeds and detailed configuration

## Data Files Note

CSV data files are gitignored due to size. Backups maintained at:
- AVCAffe: `E:\FYP\Dataset\AVCAffe\extracted_features_backup\`

## License

See main project LICENSE file.

## Contributors

- Feature extraction and AVCAffe integration: Claude Sonnet 4.5
- Project structure and StressID integration: [Original authors]
