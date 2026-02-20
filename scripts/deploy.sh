#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/opt/cle-app"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

echo "=== CLE Deploy: $(date) ==="

cd "$PROJECT_DIR"
git pull origin deploy

docker compose -f "$COMPOSE_FILE" down --remove-orphans
docker compose -f "$COMPOSE_FILE" up --build -d

echo "Waiting for backend to become healthy..."

MAX_WAIT=60
INTERVAL=5
ELAPSED=0

until curl -sf http://localhost:8000/health | grep -q '"status":"healthy"'; do
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Backend did not become healthy within ${MAX_WAIT}s"
        docker compose -f "$COMPOSE_FILE" logs backend
        exit 1
    fi
    echo "  Waiting... (${ELAPSED}s elapsed)"
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo "Backend is healthy."

if curl -sf http://localhost:80 > /dev/null; then
    echo "Frontend is up."
else
    echo "WARNING: Frontend did not respond on port 80"
    docker compose -f "$COMPOSE_FILE" logs frontend
    exit 1
fi

echo "=== Deploy complete ==="
