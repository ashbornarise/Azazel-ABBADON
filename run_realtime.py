"""
Rex Clemens Capital — Trading temps reel
=========================================
Le robot trade toutes les heures sur des donnees reelles.
Les prix se rafraichissent toutes les 30 secondes.

Usage : python run_realtime.py
Dashboard : http://localhost:8777
"""

import http.server
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.strategies.ml_strategy import MLStrategy

ETAT_PATH = Path("data/etat_realtime.json")
PORT = 8777
CAPITAL_INIT = 100.0

ACTIFS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "MSFT": "Microsoft",
    "SOL-USD": "Solana",
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


etat = {
    "capital": CAPITAL_INIT,
    "capital_initial": CAPITAL_INIT,
    "cash": CAPITAL_INIT,
    "valeur_positions": 0.0,
    "pnl": 0.0,
    "pnl_pct": 0.0,
    "drawdown": 0.0,
    "plus_haut": CAPITAL_INIT,
    "nb_trades": 0,
    "trades_gagnants": 0,
    "trades_perdants": 0,
    "gain_total": 0.0,
    "perte_totale": 0.0,
    "positions": {},
    "signaux": {},
    "prix_historique": {},
    "equity": [],
    "timestamps_equity": [],
    "trades": [],
    "derniere_analyse": "-",
    "prochaine_analyse": "-",
    "timestamp": "-",
    "actif": True,
}

lock = threading.Lock()


def sauver():
    ETAT_PATH.parent.mkdir(exist_ok=True)
    with open(ETAT_PATH, "w") as f:
        json.dump(etat, f, cls=NumpyEncoder)


def calculer_valeur_totale():
    val_pos = 0.0
    for sym, pos in etat["positions"].items():
        if pos.get("en_position") and sym in etat["signaux"]:
            prix = etat["signaux"][sym].get("prix", 0)
            val_pos += pos["quantite"] * prix
    etat["valeur_positions"] = round(val_pos, 2)
    total = etat["cash"] + val_pos
    etat["capital"] = round(total, 2)
    etat["pnl"] = round(total - CAPITAL_INIT, 2)
    etat["pnl_pct"] = round((total / CAPITAL_INIT - 1) * 100, 2)
    etat["plus_haut"] = max(etat["plus_haut"], total)
    if etat["plus_haut"] > 0:
        etat["drawdown"] = round((total - etat["plus_haut"]) / etat["plus_haut"] * 100, 2)


def fetch_prix():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for sym, nom in ACTIFS.items():
        try:
            t = yf.Ticker(sym)
            h = t.history(period="1d", interval="1m")
            if h.empty:
                h = t.history(period="5d", interval="5m")
            if h.empty:
                continue
            prix = float(h["Close"].iloc[-1])

            if sym not in etat["signaux"]:
                etat["signaux"][sym] = {"nom": nom, "decision": "ATTENTE", "exposition": 0, "prix": prix}
            etat["signaux"][sym]["prix"] = round(prix, 2)

            if sym not in etat["prix_historique"]:
                etat["prix_historique"][sym] = []
            etat["prix_historique"][sym].append([now, round(prix, 2)])
            if len(etat["prix_historique"][sym]) > 300:
                etat["prix_historique"][sym] = etat["prix_historique"][sym][-300:]
        except Exception as e:
            print(f"  [PRIX] {sym}: {e}")

    with lock:
        calculer_valeur_totale()
        etat["timestamp"] = now
        ts = datetime.now(timezone.utc).strftime("%H:%M")
        etat["equity"].append(round(etat["capital"], 2))
        etat["timestamps_equity"].append(ts)
        if len(etat["equity"]) > 300:
            etat["equity"] = etat["equity"][-300:]
            etat["timestamps_equity"] = etat["timestamps_equity"][-300:]
        sauver()


def analyser_et_trader():
    now = datetime.now(timezone.utc)
    etat["derniere_analyse"] = now.strftime("%Y-%m-%d %H:%M")
    etat["prochaine_analyse"] = (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")

    print(f"\n{'='*50}")
    print(f"[TRADE] Analyse a {now.strftime('%H:%M:%S')} UTC")
    print(f"  Cash: {etat['cash']:.2f}$ | Capital: {etat['capital']:.2f}$")

    for sym, nom in ACTIFS.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="2y", interval="1d")
            if hist.empty or len(hist) < 300:
                print(f"  {nom}: pas assez de donnees ({len(hist)} jours)")
                continue

            prix_serie = hist["Close"].dropna()
            prix_actuel = float(prix_serie.iloc[-1])

            ml = MLStrategy(taille_min_train=252, pas_reentrain=21, n_arbres=50, seed=42)
            signal = ml.calculer_exposition(prix_serie)
            expo = float(signal.iloc[-1])

            if expo >= 0.6:
                decision = "ACHAT"
            elif expo <= 0.4:
                decision = "CASH"
            else:
                decision = "NEUTRE"

            etat["signaux"][sym] = {
                "nom": nom,
                "decision": decision,
                "exposition": round(expo, 3),
                "prix": round(prix_actuel, 2),
            }

            pos = etat["positions"].get(sym, {})

            if decision == "ACHAT" and not pos.get("en_position"):
                montant = etat["cash"] * 0.20
                if montant >= 1:
                    qte = montant / prix_actuel
                    etat["positions"][sym] = {
                        "en_position": True,
                        "prix_entree": prix_actuel,
                        "quantite": qte,
                        "montant_entree": round(montant, 2),
                        "date": now.strftime("%Y-%m-%d %H:%M"),
                    }
                    etat["cash"] -= montant
                    etat["nb_trades"] += 1
                    etat["trades"].append({
                        "date": now.strftime("%m-%d %H:%M"),
                        "sym": sym, "nom": nom,
                        "action": "ACHAT",
                        "prix": round(prix_actuel, 2),
                        "montant": round(montant, 2),
                        "capital": round(etat["cash"], 2),
                    })
                    print(f"  {nom}: ACHAT {montant:.2f}$ a {prix_actuel:.2f} (expo {expo:.0%})")
                else:
                    print(f"  {nom}: ACHAT souhaite mais cash insuffisant ({etat['cash']:.2f}$)")

            elif decision == "CASH" and pos.get("en_position"):
                valeur = pos["quantite"] * prix_actuel
                pnl = valeur - pos["montant_entree"]
                etat["cash"] += valeur
                if pnl >= 0:
                    etat["trades_gagnants"] += 1
                    etat["gain_total"] += pnl
                else:
                    etat["trades_perdants"] += 1
                    etat["perte_totale"] += abs(pnl)
                etat["positions"][sym] = {"en_position": False}
                etat["trades"].append({
                    "date": now.strftime("%m-%d %H:%M"),
                    "sym": sym, "nom": nom,
                    "action": "VENTE",
                    "prix": round(prix_actuel, 2),
                    "montant": round(valeur, 2),
                    "pnl": round(pnl, 2),
                    "capital": round(etat["cash"], 2),
                })
                print(f"  {nom}: VENTE {valeur:.2f}$ (PnL: {pnl:+.2f}$)")

            else:
                print(f"  {nom}: {decision} | {prix_actuel:.2f} | expo {expo:.0%}")

            if len(etat["trades"]) > 60:
                etat["trades"] = etat["trades"][-60:]

        except Exception as e:
            print(f"  {nom}: ERREUR - {e}")

    with lock:
        calculer_valeur_totale()
        sauver()

    total_pos = sum(1 for p in etat["positions"].values() if p.get("en_position"))
    print(f"  Positions ouvertes: {total_pos} | Capital: {etat['capital']:.2f}$ | PnL: {etat['pnl']:+.2f}$")


def loop_prix():
    while True:
        try:
            fetch_prix()
        except Exception as e:
            print(f"[PRIX] Erreur globale: {e}")
        time.sleep(30)


def loop_trade():
    analyser_et_trader()
    while True:
        time.sleep(3600)
        try:
            analyser_et_trader()
        except Exception as e:
            print(f"[TRADE] Erreur globale: {e}")


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/etat":
            with lock:
                try:
                    data = ETAT_PATH.read_text()
                except FileNotFoundError:
                    data = "{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data.encode())
            return

        if self.path == "/" or self.path == "":
            self.path = "/dashboard_realtime.html"
        super().do_GET()

    def log_message(self, fmt, *args):
        pass


def main():
    ETAT_PATH.parent.mkdir(exist_ok=True)
    sauver()

    server = http.server.HTTPServer(("", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[SERVEUR] Dashboard sur http://localhost:{PORT}")

    threading.Thread(target=loop_prix, daemon=True).start()
    print("[PRIX] Rafraichissement toutes les 30 secondes")

    print("[TRADE] Premiere analyse en cours...")
    print()

    try:
        loop_trade()
    except KeyboardInterrupt:
        print("\n[SERVEUR] Arret.")
        server.shutdown()


if __name__ == "__main__":
    main()
