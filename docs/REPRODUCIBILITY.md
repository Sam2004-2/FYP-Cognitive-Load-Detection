# Reproducibility

## What This Repo Supports Directly

- starting the backend API
- starting the React study UI
- running backend tests
- running frontend tests
- building the frontend
- running the checked-in training scripts

Use [../QUICKSTART.md](../QUICKSTART.md) for the main commands.

## What Is Not Self-Contained Here

- raw study datasets
- external physiological training inputs
- participant export archives
- thesis-writing source outside this repo

Those inputs are referenced by scripts and docs, but they are not committed here.

## Current Reproducibility Boundary

This repository is best treated as:

1. a runnable application repository
2. a reproducible code path for the checked-in workflows
3. not a fully self-contained data-and-results archive

## Important Source Files

- backend runtime: `machine_learning/src/cle/server.py`
- model loading: `machine_learning/src/cle/api.py`
- frontend feature config: `UI/src/config/featureConfig.ts`
- study controller config: `UI/src/config/studyConfig.ts`
