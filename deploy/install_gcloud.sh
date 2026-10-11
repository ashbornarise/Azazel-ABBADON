#!/bin/bash
# Installation sur Google Cloud VM (e2-micro gratuit)
# Usage : bash deploy/install_gcloud.sh

set -e

echo "=== Rex Clemens Capital — Installation Google Cloud ==="

# Mise a jour
sudo apt update && sudo apt upgrade -y

# Python + pip
sudo apt install -y python3 python3-pip python3-venv git

# Environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Dependances
pip install pandas numpy scikit-learn yfinance flask

# Dossier data
mkdir -p data/cache

# Test rapide
echo "Test des imports..."
python3 -c "import pandas, numpy, sklearn, yfinance; print('OK — toutes les dependances installees')"

# Creer le service systemd
sudo tee /etc/systemd/system/rex-clemens.service > /dev/null <<EOF
[Unit]
Description=Rex Clemens Capital — Robot de trading educatif
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python run_live.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Activer le service
sudo systemctl daemon-reload
sudo systemctl enable rex-clemens
sudo systemctl start rex-clemens

# Ouvrir le port 8777
sudo iptables -A INPUT -p tcp --dport 8777 -j ACCEPT

echo ""
echo "=== Installation terminee ==="
echo "Dashboard : http://$(curl -s ifconfig.me):8777"
echo "Status    : sudo systemctl status rex-clemens"
echo "Logs      : sudo journalctl -u rex-clemens -f"
echo "Arreter   : sudo systemctl stop rex-clemens"
echo ""
echo "RAPPEL : aucun argent reel, simulation uniquement."
