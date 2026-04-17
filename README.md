# Cognitive Load Detection Thesis Project

This repository contains the code used for the final-year project: a Python backend for cognitive-load inference and study data handling, plus a React study UI.

## Repository Layout

- `machine_learning/` - FastAPI backend, model loading, training scripts, tests
- `UI/` - React/TypeScript study application
- `docs/` - lightweight submission-facing documentation
- `deploy/` - deployment assets such as the Caddy config
- `scripts/` - deployment and Azure helper scripts

Large datasets, participant exports, and thesis-writing material are not part of this repository.

## Start Here

- [Quickstart](./QUICKSTART.md)
- [Documentation Index](./docs/README.md)
- [API Notes](./docs/API.md)
- [Configuration Notes](./docs/CONFIGURATION.md)
- [Study Protocol](./docs/STUDY_PROTOCOL.md)
- [Reproducibility Notes](./docs/REPRODUCIBILITY.md)
- [Deployment Runbook](./DEPLOY_AZURE_VM.md)
- [Report Export Notes](./REPORT_EXPORTS.md)

## Main Components

- Backend API routes live in `machine_learning/src/cle/server.py`
- Frontend routes live in `UI/src/App.tsx`
- Study timings and controller settings live in `UI/src/config/studyConfig.ts`
- Frontend feature settings live in `UI/src/config/featureConfig.ts`
