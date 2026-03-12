# Azure VM Deployment Runbook

This runbook provisions and deploys the study app on a new Azure VM with HTTPS, persistent reports, and CI-gated auto-deploy.

## 1) Credentials and Access You Need

- Azure subscription access with permission to create VM/network resources.
- Domain DNS control for an `A` record.
- GitHub repo admin/maintainer access (to add deploy key and Actions secrets).
- SSH key pair for VM admin access.
- A strong random token for `CLE_ADMIN_TOKEN`.

## 2) Provision Azure Infrastructure

From your local machine:

```bash
az login
az account set --subscription "<subscription-id-or-name>"
ADMIN_CIDR="<your-public-ip>/32" \
RESOURCE_GROUP="rg-cle-study" \
LOCATION="eastus" \
VM_NAME="cle-study-vm" \
ADMIN_USERNAME="azureuser" \
SSH_PUBLIC_KEY_PATH="$HOME/.ssh/id_ed25519.pub" \
bash scripts/azure/provision_vm.sh
```

This creates:
- resource group
- vnet/subnet
- NSG (22 restricted to your CIDR, 80/443 public)
- static public IP
- Ubuntu 22.04 VM (`Standard_B2s`, 64GB disk)

## 3) Point Domain to VM

Create/update DNS A record:
- `study.<your-domain>` -> `<vm-public-ip>`

Wait for DNS propagation.

## 4) Bootstrap the VM

SSH into the VM and run:

```bash
ssh azureuser@<vm-public-ip>
cd /tmp
# If repo is already cloned locally, copy script then run it.
# Otherwise run from cloned project checkout.
bash /opt/cle-app/scripts/azure/bootstrap_vm.sh
```

The bootstrap script installs Docker, firewall rules, unattended upgrades, creates:
- `/opt/cle-app`
- `/opt/cle-data/reports`
- `/opt/cle-data/backups`

## 5) Configure GitHub Deploy Key (VM -> Repo Pull)

1. Generate a read-only deploy key pair (on VM or local).
2. Add public key in GitHub repo: Settings -> Deploy keys -> Add key (read-only).
3. Install private key on VM and configure SSH alias:

```bash
DEPLOY_KEY_PATH="$HOME/<private-key-file>" bash /opt/cle-app/scripts/azure/setup_deploy_key.sh
```

4. Set repo remote to deploy-key alias if needed:

```bash
cd /opt/cle-app
git remote set-url origin git@github-fyp:Sam2004-2/FYP-Cognitive-Load-Detection.git
```

## 6) Configure Production Environment

Edit `/opt/cle-app/.env.production`:

```env
STUDY_DOMAIN=study.example.com
CLE_ALLOWED_ORIGINS=https://study.example.com
STUDY_PUBLIC_URL=https://study.example.com
CLE_ADMIN_TOKEN=<long-random-secret>
REPORTS_HOST_DIR=/opt/cle-data/reports
CLE_REPORTS_DIR=/app/data/reports
```

## 7) First Manual Deploy

```bash
cd /opt/cle-app
bash scripts/deploy.sh
```

Verify:
- `https://study.<your-domain>/` loads
- `https://study.<your-domain>/health` returns healthy JSON

## 8) Configure GitHub Actions Secrets

In GitHub -> Settings -> Secrets and variables -> Actions, set:
- `SSH_HOST`
- `SSH_USER`
- `SSH_PORT`
- `SSH_KEY`
- `STUDY_DOMAIN`
- `CLE_ADMIN_TOKEN`

## 9) Auto-Deploy Behavior

- CI runs on `thesis/prune` pushes.
- Deploy workflow triggers only after CI success.
- Deploy action SSHes to VM and runs `scripts/deploy.sh`.
- VM pulls latest `origin/thesis/prune`, rebuilds, and health-checks.

## 10) Smoke Test Checklist

- Open `https://study.<your-domain>/`
- Verify camera permission works in browser (HTTPS required)
- Run a study session end-to-end
- Confirm files appear under `/opt/cle-data/reports/sessions/...`
- Complete delayed test and confirm `/opt/cle-data/reports/delayed/...`
- Validate admin export endpoint using bearer token (see `REPORT_EXPORTS.md`)
