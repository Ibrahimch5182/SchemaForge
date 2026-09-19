#!/usr/bin/env bash
# One-shot host preparation for a single small EC2 instance (Ubuntu 22.04/24.04 or Amazon Linux 2023).
# Installs Docker Engine + the Compose plugin, adds swap (small hosts), creates the model directory.
# Run on the instance as a sudo-capable user:   bash bootstrap_ec2.sh
# It does not touch AWS APIs and needs no credentials.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/opt/schemaforge/models}"
SWAP_GB="${SWAP_GB:-4}"

. /etc/os-release
case "$ID" in
  ubuntu)
    sudo apt-get update -y
    sudo apt-get install -y ca-certificates curl git
    curl -fsSL https://get.docker.com | sudo sh   # official convenience script (Docker Engine + compose plugin)
    ;;
  amzn)
    sudo dnf install -y docker git
    sudo systemctl enable --now docker
    # Compose v2 plugin (Amazon Linux's repos do not ship it)
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo curl -fsSL "https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-$(uname -m)" \
      -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    ;;
  *) echo "Unsupported OS: $ID (use Ubuntu or Amazon Linux 2023)" >&2; exit 1 ;;
esac
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true

# Swap: a safety net against OOM while the 2.4 GB model loads on a small host (not a substitute for RAM).
if ! swapon --show | grep -q .; then
  sudo fallocate -l "${SWAP_GB}G" /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
fi

sudo mkdir -p "$MODEL_DIR"
sudo chown "$USER":"$USER" "$MODEL_DIR"

echo
free -h | head -2
docker --version
echo "Done. Log out and back in (docker group), then copy the model artifacts into $MODEL_DIR."
