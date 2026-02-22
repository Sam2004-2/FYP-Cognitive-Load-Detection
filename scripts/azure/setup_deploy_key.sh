#!/usr/bin/env bash
set -euo pipefail

# Installs a read-only GitHub deploy key on the VM user account.
# Usage:
#   DEPLOY_KEY_PATH=~/Downloads/fyp-deploy-key bash scripts/azure/setup_deploy_key.sh

: "${DEPLOY_KEY_PATH:=}"
: "${KEY_NAME:=id_ed25519_fyp_deploy}"
: "${REPO_SSH_HOST_ALIAS:=github-fyp}"
: "${REPO_SSH_PATH:=Sam2004-2/FYP-Cognitive-Load-Detection.git}"

if [[ -z "$DEPLOY_KEY_PATH" ]]; then
  echo "ERROR: DEPLOY_KEY_PATH is required." >&2
  exit 1
fi

if [[ ! -f "$DEPLOY_KEY_PATH" ]]; then
  echo "ERROR: Deploy key not found at $DEPLOY_KEY_PATH" >&2
  exit 1
fi

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

install -m 600 "$DEPLOY_KEY_PATH" "$HOME/.ssh/$KEY_NAME"
ssh-keyscan github.com >> "$HOME/.ssh/known_hosts"

if ! grep -q "Host $REPO_SSH_HOST_ALIAS" "$HOME/.ssh/config" 2>/dev/null; then
  cat >> "$HOME/.ssh/config" <<EOF
Host $REPO_SSH_HOST_ALIAS
  HostName github.com
  User git
  IdentityFile ~/.ssh/$KEY_NAME
  IdentitiesOnly yes
EOF
fi

chmod 600 "$HOME/.ssh/config"

echo "Deploy key installed."
echo "Use this remote URL: git@$REPO_SSH_HOST_ALIAS:$REPO_SSH_PATH"
