#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/opt/cle-app"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"
ENV_FILE="$PROJECT_DIR/.env.production"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-thesis/prune}"

echo "=== CLE Deploy: $(date) ==="

echo "Project directory: $PROJECT_DIR"
echo "Deploy branch: $DEPLOY_BRANCH"

cd "$PROJECT_DIR"

git fetch origin "$DEPLOY_BRANCH"
git checkout "$DEPLOY_BRANCH"
git pull --ff-only origin "$DEPLOY_BRANCH"

COMPOSE_ARGS=("-f" "$COMPOSE_FILE")
if [ -f "$ENV_FILE" ]; then
  echo "Using environment file: $ENV_FILE"
  COMPOSE_ARGS=("--env-file" "$ENV_FILE" "${COMPOSE_ARGS[@]}")
fi

docker compose "${COMPOSE_ARGS[@]}" down --remove-orphans
docker compose "${COMPOSE_ARGS[@]}" up --build -d

echo "Waiting for backend to become healthy..."

MAX_WAIT=120
INTERVAL=5
ELAPSED=0

until docker compose "${COMPOSE_ARGS[@]}" exec -T backend curl -sf http://localhost:8000/health 2>/dev/null | grep -q '"status":"healthy"'; do
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Backend did not become healthy within ${MAX_WAIT}s"
        docker compose "${COMPOSE_ARGS[@]}" logs backend
        exit 1
    fi
    echo "  Waiting... (${ELAPSED}s elapsed)"
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo "Backend is healthy."

if curl -kfsS https://localhost/ >/dev/null || curl -fsS http://localhost/ >/dev/null; then
    echo "Public frontend is up."
else
    echo "ERROR: Frontend/Caddy endpoint did not respond"
    docker compose "${COMPOSE_ARGS[@]}" logs caddy frontend
    exit 1
fi

echo "=== Deploy complete ==="
