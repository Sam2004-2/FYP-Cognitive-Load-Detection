# CI/CD Pipeline Documentation

This repository uses GitHub Actions for continuous integration and testing.

## Workflows

### CI Pipeline (`ci.yml`)

The CI pipeline runs automatically on:
- Push to `main`, `master`, or `develop` branches
- Pull requests targeting these branches

#### Jobs

**1. test-ml-pipeline**
- Runs all Python tests in the Machine Learning pipeline
- Generates code coverage reports
- Uploads coverage to Codecov
- Tests include:
  - Feature extraction tests
  - Windowing and preprocessing tests
  - Model training and evaluation tests
  - Integration tests

**2. lint-ml-pipeline**
- Checks code quality using flake8
- Verifies code formatting with black
- Runs on Python codebase only
- Configured to not fail the build (continue-on-error)

**3. test-ui**
- Builds the React TypeScript UI
- Runs frontend tests
- Verifies that the UI compiles successfully

## Local Testing

Before pushing code, you can run tests locally:

### Machine Learning Tests
```bash
cd "Machine Learning"
python -m pytest tests/ -v
```

### With Coverage
```bash
cd "Machine Learning"
python -m pytest tests/ -v --cov=src --cov-report=term
```

### UI Build
```bash
cd ui
npm run build
```

## Adding New Tests

1. Create test files in the appropriate `tests/` directory
2. Follow the naming convention: `test_*.py` for Python tests
3. Ensure tests are independent and can run in any order
4. The CI pipeline will automatically pick up and run new tests

## Troubleshooting CI Failures

### Python Tests Failing
- Check test output in the GitHub Actions logs
- Verify all dependencies are listed in `requirements.txt`
- Ensure `PYTHONPATH` is set correctly (handled by workflow)

### UI Build Failing
- Check for TypeScript compilation errors
- Verify all npm dependencies are in `package.json`
- Test locally with `npm run build`

### Linting Failures
- Run `black src/ tests/ --line-length=100` to auto-format
- Run `flake8 src/ tests/` to check for issues

## Coverage Reports

Code coverage reports are uploaded to Codecov automatically. You can view detailed coverage metrics by:
1. Setting up Codecov integration (add repository to Codecov.io)
2. Adding the Codecov badge to your README
3. Viewing coverage trends in the Codecov dashboard

## Status Badge

Add this to your main README.md to show CI status:

```markdown
[![CI Pipeline](https://github.com/YOUR_USERNAME/FYP/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/FYP/actions/workflows/ci.yml)
```

Replace `YOUR_USERNAME` with your GitHub username.

