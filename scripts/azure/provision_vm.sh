#!/usr/bin/env bash
set -euo pipefail

# Required:
#   ADMIN_CIDR="<your-public-ip>/32" scripts/azure/provision_vm.sh
# Optional overrides:
#   RESOURCE_GROUP, LOCATION, VM_NAME, VM_SIZE, ADMIN_USERNAME,
#   VNET_NAME, SUBNET_NAME, NSG_NAME, NIC_NAME, PUBLIC_IP_NAME,
#   SSH_PUBLIC_KEY_PATH

: "${RESOURCE_GROUP:=rg-cle-study}"
: "${LOCATION:=eastus}"
: "${VM_NAME:=cle-study-vm}"
: "${VM_SIZE:=Standard_B2s}"
: "${ADMIN_USERNAME:=azureuser}"
: "${VNET_NAME:=cle-study-vnet}"
: "${SUBNET_NAME:=cle-study-subnet}"
: "${NSG_NAME:=cle-study-nsg}"
: "${NIC_NAME:=cle-study-nic}"
: "${PUBLIC_IP_NAME:=cle-study-ip}"

if [[ -z "${ADMIN_CIDR:-}" ]]; then
  echo "ERROR: ADMIN_CIDR is required (example: ADMIN_CIDR=203.0.113.10/32)" >&2
  exit 1
fi

SSH_ARGS=(--generate-ssh-keys)
if [[ -n "${SSH_PUBLIC_KEY_PATH:-}" ]]; then
  if [[ ! -f "$SSH_PUBLIC_KEY_PATH" ]]; then
    echo "ERROR: SSH public key not found at $SSH_PUBLIC_KEY_PATH" >&2
    exit 1
  fi
  SSH_ARGS=(--ssh-key-values "$SSH_PUBLIC_KEY_PATH")
fi

echo "Creating resource group..."
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none

echo "Creating virtual network + subnet..."
az network vnet create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VNET_NAME" \
  --location "$LOCATION" \
  --address-prefixes 10.10.0.0/16 \
  --subnet-name "$SUBNET_NAME" \
  --subnet-prefixes 10.10.1.0/24 \
  --output none

echo "Creating network security group..."
az network nsg create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$NSG_NAME" \
  --location "$LOCATION" \
  --output none

echo "Adding NSG rules..."
az network nsg rule create \
  --resource-group "$RESOURCE_GROUP" \
  --nsg-name "$NSG_NAME" \
  --name allow-ssh-admin \
  --priority 100 \
  --direction Inbound \
  --access Allow \
  --protocol Tcp \
  --source-address-prefixes "$ADMIN_CIDR" \
  --source-port-ranges '*' \
  --destination-address-prefixes '*' \
  --destination-port-ranges 22 \
  --output none

az network nsg rule create \
  --resource-group "$RESOURCE_GROUP" \
  --nsg-name "$NSG_NAME" \
  --name allow-http \
  --priority 110 \
  --direction Inbound \
  --access Allow \
  --protocol Tcp \
  --source-address-prefixes '*' \
  --source-port-ranges '*' \
  --destination-address-prefixes '*' \
  --destination-port-ranges 80 \
  --output none

az network nsg rule create \
  --resource-group "$RESOURCE_GROUP" \
  --nsg-name "$NSG_NAME" \
  --name allow-https \
  --priority 120 \
  --direction Inbound \
  --access Allow \
  --protocol Tcp \
  --source-address-prefixes '*' \
  --source-port-ranges '*' \
  --destination-address-prefixes '*' \
  --destination-port-ranges 443 \
  --output none

echo "Creating static public IP..."
az network public-ip create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$PUBLIC_IP_NAME" \
  --location "$LOCATION" \
  --sku Standard \
  --allocation-method Static \
  --version IPv4 \
  --output none

echo "Creating NIC..."
az network nic create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$NIC_NAME" \
  --vnet-name "$VNET_NAME" \
  --subnet "$SUBNET_NAME" \
  --network-security-group "$NSG_NAME" \
  --public-ip-address "$PUBLIC_IP_NAME" \
  --output none

echo "Creating VM..."
az vm create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --location "$LOCATION" \
  --nics "$NIC_NAME" \
  --image Ubuntu2204 \
  --size "$VM_SIZE" \
  --admin-username "$ADMIN_USERNAME" \
  --os-disk-size-gb 64 \
  --authentication-type ssh \
  "${SSH_ARGS[@]}" \
  --output none

PUBLIC_IP=$(az network public-ip show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$PUBLIC_IP_NAME" \
  --query ipAddress \
  --output tsv)

echo "Provisioning complete."
echo "VM:        $VM_NAME"
echo "Public IP: $PUBLIC_IP"
echo "SSH:       ssh ${ADMIN_USERNAME}@${PUBLIC_IP}"
echo "Next:      point your domain A record to $PUBLIC_IP, then run scripts/azure/bootstrap_vm.sh on the VM"
