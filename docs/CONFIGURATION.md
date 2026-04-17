# Configuration

## Backend and Deployment Variables

These variables are used by the backend or deployment setup.

| Variable | Purpose | Typical local value |
| --- | --- | --- |
| `CLE_MODELS_DIR` | model artifact directory | `models/video_physio_regression_z01_geom` |
| `CLE_REPORTS_DIR` | report storage path inside the backend runtime | `/app/data/reports` |
| `CLE_ALLOWED_ORIGINS` | allowed frontend origins | `http://localhost:3000` |
| `CLE_ADMIN_TOKEN` | bearer token for admin routes | set a private token |
| `STUDY_PUBLIC_URL` | public study URL | `http://localhost` |
| `REPORTS_HOST_DIR` | host path mounted into the backend container | `./data/reports` |
| `STUDY_DOMAIN` | public hostname for deployed Caddy setup | `localhost` |

A starter file is provided at [../.env.example](../.env.example).

## Frontend Variable

The UI reads:

| Variable | Purpose | Typical local value |
| --- | --- | --- |
| `REACT_APP_API_URL` | API base URL for the React app | `http://localhost:8000` |

Use [../UI/.env.example](../UI/.env.example) as a starting point for local UI development.

## Current Source Files

- backend env usage: `machine_learning/src/cle/server.py`
- frontend env usage: `UI/src/config/featureConfig.ts`
- compose defaults: `docker-compose.yml`
