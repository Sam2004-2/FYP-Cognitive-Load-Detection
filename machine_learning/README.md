# CLE Backend

This directory contains the FastAPI backend, model-loading code, training scripts, and backend tests.

## Local Run

```bash
python -m pip install -e ".[dev]"
python -m src.cle.server --host 0.0.0.0 --port 8000 --models-dir models/video_physio_regression_z01_geom
```

## Main Runtime Routes

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /study/session-records`
- `POST /study/delayed-records`
- `GET /admin/reports/index`

## Notes

- runtime routes are defined in `src/cle/server.py`
- model loading lives in `src/cle/api.py`
- package configuration lives in `pyproject.toml`

For wider project docs, see [../docs/README.md](../docs/README.md).
