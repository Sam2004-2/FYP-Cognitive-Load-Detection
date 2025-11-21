# CI/CD Setup Guide

## Overview

This project uses GitHub Actions for continuous integration, automatically running tests on every push and pull request.

## Current Test Suite

The ML pipeline includes tests for:
- **Feature Extraction** (`test_features.py`, `test_per_frame.py`)
- **Windowing Logic** (`test_windowing.py`)
- **Configuration Management** (`test_config.py`)
- **Integration Tests** (`test_integration.py`)

## Running Tests Locally

### Setup Development Environment

```bash
cd "Machine Learning"

# Install the package in development mode with all dev dependencies
pip install -e ".[dev]"
```

### Run All Tests

```bash
# From the Machine Learning directory
python -m pytest tests/ -v
```

### Run Specific Test Files

```bash
# Test feature extraction
python -m pytest tests/test_features.py -v

# Test windowing
python -m pytest tests/test_windowing.py -v

# Test integration
python -m pytest tests/test_integration.py -v
```

### Run Tests with Coverage

```bash
# Generate coverage report
python -m pytest tests/ -v --cov=src --cov-report=term --cov-report=html

# View HTML coverage report
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

### Run Specific Tests

```bash
# Run a specific test function
python -m pytest tests/test_features.py::test_compute_blink_features -v

# Run tests matching a pattern
python -m pytest tests/ -k "blink" -v
```

## GitHub Actions Workflow

The CI pipeline (`.github/workflows/ci.yml`) includes three jobs:

### 1. test-ml-pipeline
- **Triggers**: Push or PR to main/master/develop
- **Python Version**: 3.9 (compatible with 3.10+ codebase)
- **Steps**:
  1. Checkout code
  2. Set up Python with pip caching
  3. Install system dependencies (OpenGL, etc.)
  4. Install Python package and dependencies
  5. Run pytest with coverage
  6. Upload coverage to Codecov

### 2. lint-ml-pipeline
- **Checks**:
  - `flake8` for code quality
  - `black` for code formatting
- **Non-blocking**: Won't fail the build

### 3. test-ui
- **Tests**: React TypeScript UI
- **Steps**:
  1. Install Node.js dependencies
  2. Build the UI
  3. Run tests (if available)

## CI Configuration Files

```
FYP/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Main CI workflow
│       └── README.md           # Workflow documentation
├── Machine Learning/
│   ├── pyproject.toml          # Python project config & dependencies
│   ├── requirements.txt        # Alternative dependency list
│   └── tests/                  # Test directory
└── ui/
    └── package.json            # Node.js dependencies
```

## Adding New Tests

1. Create test file: `tests/test_your_feature.py`
2. Follow naming conventions:
   ```python
   def test_your_function():
       # Arrange
       input_data = ...
       
       # Act
       result = your_function(input_data)
       
       # Assert
       assert result == expected
   ```
3. Run locally: `python -m pytest tests/test_your_feature.py -v`
4. Commit and push - CI will run automatically

## Troubleshooting

### Tests Pass Locally But Fail in CI

**Possible causes:**
1. Missing dependencies in `pyproject.toml`
2. Hard-coded file paths (use relative paths)
3. Environment-specific assumptions
4. Missing test data files

**Solutions:**
- Check GitHub Actions logs for specific errors
- Ensure all test data is committed to git
- Use `pathlib` for cross-platform paths
- Mock external dependencies

### Import Errors

```python
ModuleNotFoundError: No module named 'src'
```

**Solution**: The package is installed in editable mode in CI. Locally:
```bash
cd "Machine Learning"
pip install -e .
```

### OpenCV/MediaPipe Errors

```
ImportError: libGL.so.1: cannot open shared object file
```

**Solution**: Already handled in CI workflow with:
```yaml
- name: Install system dependencies
  run: |
    sudo apt-get update
    sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev
```

**Note**: Ubuntu 24.04+ uses `libgl1` instead of the deprecated `libgl1-mesa-glx`

## Code Quality Tools

### Black (Code Formatter)

```bash
# Check formatting
black --check src/ tests/

# Auto-format
black src/ tests/
```

### Flake8 (Linter)

```bash
# Check for errors
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Full check with warnings
flake8 src/ tests/ --count --max-complexity=10 --max-line-length=127 --statistics
```

### MyPy (Type Checker)

```bash
# Type check the codebase
mypy src/
```

## Coverage Goals

Current coverage targets:
- **Overall**: >80%
- **Core modules** (`extract/`, `train/`): >85%
- **Utilities**: >70%

View coverage locally:
```bash
python -m pytest tests/ --cov=src --cov-report=term-missing
```

## Pre-commit Hooks (Optional)

Set up pre-commit hooks to run checks before committing:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Status Badge

Add to your main `README.md`:

```markdown
[![CI Pipeline](https://github.com/USERNAME/FYP/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/FYP/actions/workflows/ci.yml)
```

## Next Steps

1. ✅ Set up GitHub repository
2. ✅ Push code with `.github/workflows/ci.yml`
3. ⬜ Watch the Actions tab for first run
4. ⬜ (Optional) Set up Codecov integration
5. ⬜ (Optional) Add branch protection rules
6. ⬜ (Optional) Configure pre-commit hooks

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Codecov Documentation](https://docs.codecov.com/)

