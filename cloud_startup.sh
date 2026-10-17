#!/bin/bash
# Rex Clemens Capital — Google Cloud VM Startup Script
# Ce script s'execute automatiquement a chaque demarrage du VM.
# Pas besoin de SSH !

set -e
LOG="/var/log/rex-clemens-startup.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== REX CLEMENS STARTUP $(date) ==="

APP_DIR="/opt/rex-clemens"
REPO="https://github.com/ashbornarise/Azazel-ABBADON.git"

# 1. Installer Python et dependances systeme
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git

# 2. Cloner ou mettre a jour le repo
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR"
    git pull origin main || true
else
    git clone "$REPO" "$APP_DIR"
    cd "$APP_DIR"
fi

# 3. Creer le venv et installer les packages
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
venv/bin/pip install --upgrade pip -q
venv/bin/pip install numpy pandas scikit-learn yfinance lightgbm xgboost optuna -q

# 4. Creer le dossier data
mkdir -p data

# 5. Creer le service systemd
cat > /etc/systemd/system/rex-clemens.service << 'EOF'
[Unit]
Description=Rex Clemens Capital Trading Engine
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/rex-clemens
ExecStart=/opt/rex-clemens/venv/bin/python /opt/rex-clemens/run_realtime.py
Restart=always
RestartSec=30
Environment=PORT=8080

[Install]
WantedBy=multi-user.target
EOF

# 6. Activer et demarrer
systemctl daemon-reload
systemctl enable rex-clemens
systemctl restart rex-clemens

echo "=== REX CLEMENS READY $(date) ==="
echo "Dashboard: http://$(curl -s ifconfig.me):8080"
