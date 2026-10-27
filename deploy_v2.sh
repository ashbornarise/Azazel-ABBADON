#!/bin/bash
# Rex Clemens Capital — Deploy V3.1 Engine on Google Cloud
# Usage: bash deploy_v2.sh
# VM: 34.71.20.144 | Service: rex-clemens (port 8080)

set -e
echo "=== REX CLEMENS V3.1 — DEPLOIEMENT ==="
echo "40 features | 18 protections | 6 classifieurs | 326 EA"
echo ""

cd /opt/rex-clemens

# 1. Pull le nouveau code
echo "[1/5] Git pull..."
git pull origin main

# 2. Installer les dependances dans le venv
echo "[2/5] Installation des dependances..."
./venv/bin/pip install --quiet \
    optuna lightgbm xgboost yfinance scikit-learn \
    pandas numpy torch beautifulsoup4

# 3. Mettre a jour le service systemd
echo "[3/5] Mise a jour du service systemd..."
sudo tee /etc/systemd/system/rex-clemens.service > /dev/null << 'EOF'
[Unit]
Description=Rex Clemens Capital V3.1 — ML Trading Engine
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/rex-clemens
Environment=PORT=8080
ExecStart=/opt/rex-clemens/venv/bin/python /opt/rex-clemens/run_realtime.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 4. Redemarrer
echo "[4/5] Redemarrage du service..."
sudo systemctl daemon-reload
sudo systemctl restart rex-clemens

# 5. Verifier
echo "[5/5] Verification..."
sleep 5
sudo systemctl status rex-clemens --no-pager
echo ""
echo "=== DEPLOIEMENT V3.1 TERMINE ==="
echo "Dashboard: http://34.71.20.144:8080"
echo "Capital: 100$ (simulation)"
echo "Moteur: V3.1 — 40 features, 6 classifieurs (LSTM+GRU+RF+LGB+XGB+GBM)"
echo "Protections: 18 gardes de survie actives"
echo ""
echo "Le premier cycle ML prend ~2-3 min. Patiente puis rafraichis le dashboard."
