"""
WSGI pour PythonAnywhere — sert le dashboard + l'API.
Remplacer le contenu du fichier WSGI de PythonAnywhere par ceci.
"""

import json
import sys
from pathlib import Path

# Adapter le chemin a ton pseudo PythonAnywhere
PROJECT_DIR = Path.home() / "Azazel-ABBADON"
sys.path.insert(0, str(PROJECT_DIR))

from flask import Flask, send_file, jsonify

app = Flask(__name__)

ETAT_PATH = PROJECT_DIR / "data" / "etat_daily.json"


@app.route("/")
def dashboard():
    return send_file(PROJECT_DIR / "dashboard_pa.html")


@app.route("/api/etat")
def api_etat():
    try:
        data = json.loads(ETAT_PATH.read_text())
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "Pas encore de donnees. Lancer run_daily.py."})


application = app
