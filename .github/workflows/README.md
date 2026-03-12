# CI/CD Pipeline Documentation

This repository uses GitHub Actions for continuous integration and CI-gated deployment to Azure VM.

## Workflows

### CI Pipeline (`ci.yml`)

The CI pipeline runs automatically on:
- Push to `main`, `master`, `develop`, `deploy`, `thesis/prune`
- Pull requests targeting `main`, `master`, `develop`, `deploy`, `thesis/prune`

Jobs:
- `test-ml-pipeline`: installs backend dependencies and runs `pytest` in `machine_learning/`
- `test-ui`: installs frontend dependencies, builds UI, and runs tests in `UI/`

### Deploy (`deploy.yml`)

Deploy runs only when:
- CI Pipeline completes successfully for branch `thesis/prune`
- Or manually via `workflow_dispatch`

Deployment is done over SSH by running `scripts/deploy.sh` on the VM.

## Required GitHub Secrets

Configure these in repository Settings -> Secrets and variables -> Actions:

- `SSH_HOST`: Azure VM public host or IP
- `SSH_USER`: SSH username used on VM
- `SSH_PORT`: SSH port (usually `22`)
- `SSH_KEY`: private key matching the VM authorized key
- `STUDY_DOMAIN`: public domain pointing to VM (example: `study.example.com`)
- `CLE_ADMIN_TOKEN`: long random bearer token used by `/admin/*` report endpoints

## Optional Deployment Environment Variables (`/opt/cle-app/.env.production`)

- `STUDY_DOMAIN=...`
- `CLE_ALLOWED_ORIGINS=https://<domain>`
- `STUDY_PUBLIC_URL=https://<domain>`
- `CLE_ADMIN_TOKEN=<long-random-secret>`
- `REPORTS_HOST_DIR=/opt/cle-data/reports`

## Local Validation Before Push

```bash
cd machine_learning && python -m pytest tests/ -q
cd UI && npm run build
cd UI && npm test -- --watchAll=false --passWithNoTests
```

## Deployment Branch

Production VM deployment tracks: `thesis/prune`.
