#!/usr/bin/env bash
set -euo pipefail

# Run this script on the VM after provisioning.
# Optional overrides:
#   DEPLOY_USER, REPO_URL, APP_DIR, DATA_DIR, APP_BRANCH

: "${DEPLOY_USER:=$USER}"
: "${REPO_URL:=git@github.com:Sam2004-2/FYP-Cognitive-Load-Detection.git}"
: "${APP_DIR:=/opt/cle-app}"
: "${DATA_DIR:=/opt/cle-data}"
: "${APP_BRANCH:=thesis/prune}"

echo "Installing base packages..."
sudo apt-get update
sudo apt-get install -y \
  ca-certificates \
  curl \
  git \
  ufw \
  unattended-upgrades \
  docker.io \
  docker-compose-v2

echo "Enabling Docker..."
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker "$DEPLOY_USER"

echo "Configuring firewall..."
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

echo "Enabling unattended upgrades..."
sudo dpkg-reconfigure --frontend=noninteractive unattended-upgrades || true

echo "Creating application/data directories..."
sudo mkdir -p "$APP_DIR"
sudo mkdir -p "$DATA_DIR/reports"
sudo mkdir -p "$DATA_DIR/backups"
sudo chown -R "$DEPLOY_USER":"$DEPLOY_USER" "$APP_DIR" "$DATA_DIR"

echo "Cloning/updating repository..."
if [[ ! -d "$APP_DIR/.git" ]]; then
  git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"
git fetch origin "$APP_BRANCH"
git checkout "$APP_BRANCH"
git pull --ff-only origin "$APP_BRANCH"

echo "Creating .env.production template if missing..."
if [[ ! -f "$APP_DIR/.env.production" ]]; then
  cat > "$APP_DIR/.env.production" <<'ENVVARS'
# Domain used by Caddy for HTTPS certificate issuance
STUDY_DOMAIN=study.example.com

# Browser origin allowed by backend CORS
CLE_ALLOWED_ORIGINS=https://study.example.com

# Public app URL shown in metadata and links
STUDY_PUBLIC_URL=https://study.example.com

# Strong secret for admin report endpoints (Bearer token)
CLE_ADMIN_TOKEN=replace-with-long-random-secret

# Host path for persisted reports
REPORTS_HOST_DIR=/opt/cle-data/reports

# Container-side reports directory (leave as default unless needed)
CLE_REPORTS_DIR=/app/data/reports
ENVVARS
fi

echo "Bootstrap complete."
echo "Log out and back in so docker group membership applies, then run:"
echo "  cd $APP_DIR && bash scripts/deploy.sh"
