#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
# One-time VM provisioning for MindBridge central server.
# Run this ONCE on a fresh Ubuntu 22.04+ VM (DigitalOcean droplet, EC2, etc.)
# before the first CD deploy. After this, GitHub Actions handles everything.
#
# Usage: ssh into the VM, then:
#   curl -fsSL https://raw.githubusercontent.com/<org>/<repo>/main/deploy/provision_vm.sh | bash
# ══════════════════════════════════════════════════════════════════════════
set -euo pipefail

echo "==> Installing Docker + Compose plugin"
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"

echo "==> Creating deploy directory"
sudo mkdir -p /opt/mindbridge
sudo chown "$USER":"$USER" /opt/mindbridge

echo "==> Opening firewall (ufw) for HTTP/HTTPS/SSH only"
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

echo "==> Done."
echo "Next steps:"
echo "  1. Add these GitHub Actions secrets in your repo settings:"
echo "       VM_HOST      = $(curl -s ifconfig.me)"
echo "       VM_USER      = $USER"
echo "       VM_SSH_KEY   = <private key that matches an authorized_keys entry on this VM>"
echo "       MINDBRIDGE_DOMAIN = <your domain pointed at this VM's IP>"
echo "  2. Push to main — CD will build images and deploy automatically."
echo "  3. Log out and back in (or run 'newgrp docker') for the docker group to apply."
