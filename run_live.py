"""
Rex Clemens Capital — Simulation live
═════════════════════════════════════
Lance le robot en simulation : l'agent ML trade jour par jour,
le dashboard web se rafraichit en temps reel.

Usage : py run_live.py
Puis ouvrir http://localhost:8777 dans le navigateur.
"""

import http.server
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import generer_marche_aleatoire
from src.ml.features import calculer_features
from src.strategies.ml_strategy import MLStrategy
from src.strategies.buy_hold import BuyHold
from src.strategies.ma_cross import MACross
from src.survie import ModuleSurvie

ETAT_PATH = Path("data/etat_robot.json")
PORT = 8777


def etat_initial():
    return {
        "capital_initial": 100,
        "capital": 100,
        "drawdown": 0.0,
        "perte_jour": 0.0,
        "jour": 0,
        "total_jours": 0,
        "position": 0.0,
        "exposition_ml": 0.0,
        "montant_investi": 0.0,
        "etat": "ACTIF",
        "dernier_signal": "attente",
        "nb_trades": 0,
        "trades_gagnants": 0,
        "trades_perdants": 0,
        "gain_total": 0.0,
        "perte_totale": 0.0,
        "equity": [],
        "equity_bh": [],
        "equity_ma": [],
        "dates": [],
        "positions": [],
        "prix_courant": 100.0,
        "prix_liste": [],
        "journal": [],
        "accuracy": 0.0,
        "sharpe": 0.0,
        "rendement": 0.0,
        "pire_dd": 0.0,
        "regles_survie": {
            "risque_trade": {"ok": True, "valeur": 0.0, "seuil": 1.0},
            "levier": {"ok": True, "valeur": 1.0, "seuil": 1.0},
            "perte_jour": {"ok": True, "valeur": 0.0, "seuil": 3.0},
            "drawdown": {"ok": True, "valeur": 0.0, "seuil": 10.0},
            "taille_pos": {"ok": True, "valeur": 0.0, "seuil": 25.0},
            "donnees": {"ok": True, "valeur": 0, "seuil": 1},
            "connexion": {"ok": True, "valeur": 0, "seuil": 3},
        },
        "termine": False,
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


def sauver_etat(etat):
    ETAT_PATH.parent.mkdir(exist_ok=True)
    with open(ETAT_PATH, "w") as f:
        json.dump(etat, f, cls=NumpyEncoder)


def simulation_agent():
    """Simule le robot jour par jour avec un delai visible."""
    print("[AGENT] Generation des donnees simulees (2500 jours)...")
    prix = generer_marche_aleatoire(periodes=2500, seed=42)

    ml_strat = MLStrategy(taille_min_train=252, pas_reentrain=21, n_arbres=50, seed=42)
    bh_strat = BuyHold()
    ma_strat = MACross(rapide=20, lente=100)

    etat = etat_initial()
    etat["total_jours"] = len(prix)
    sauver_etat(etat)

    CAPITAL_INIT = 100.0
    capital_ml = CAPITAL_INIT
    capital_bh = CAPITAL_INIT
    capital_ma = CAPITAL_INIT
    plus_haut_ml = capital_ml
    cout = 0.001

    expo_ml_prev = 0.0
    expo_bh_prev = 0.0
    expo_ma_prev = 0.0
    en_position = False
    capital_entree = capital_ml
    trades_gagnants = 0
    trades_perdants = 0
    gain_total = 0.0
    perte_totale = 0.0

    print("[AGENT] Calcul des signaux ML (walk-forward)...")
    signal_ml = ml_strat.calculer_exposition(prix)
    signal_bh = bh_strat.calculer_exposition(prix)
    signal_ma = ma_strat.calculer_exposition(prix)

    signal_ml = signal_ml.shift(1).fillna(0.0)
    signal_bh = signal_bh.shift(1).fillna(0.0)
    signal_ma = signal_ma.shift(1).fillna(0.0)

    features = calculer_features(prix)

    print(f"[AGENT] Demarrage de la simulation live sur {len(prix)} jours")
    print(f"[AGENT] Dashboard : http://localhost:{PORT}")
    print()

    nb_trades = 0

    for i in range(1, len(prix)):
        date = prix.index[i]
        p = prix.iloc[i]
        p_prev = prix.iloc[i - 1]
        ret = (p - p_prev) / p_prev if p_prev > 0 else 0.0

        expo_ml = float(signal_ml.iloc[i])
        expo_bh = float(signal_bh.iloc[i])
        expo_ma = float(signal_ma.iloc[i])

        trade_ml = abs(expo_ml - expo_ml_prev)
        trade_bh = abs(expo_bh - expo_bh_prev)
        trade_ma = abs(expo_ma - expo_ma_prev)

        capital_ml *= (1 + expo_ml * ret - trade_ml * cout)
        capital_bh *= (1 + expo_bh * ret - trade_bh * cout)
        capital_ma *= (1 + expo_ma * ret - trade_ma * cout)

        entre_maintenant = expo_ml >= 0.4 and not en_position
        sort_maintenant = expo_ml < 0.4 and en_position

        if entre_maintenant:
            en_position = True
            capital_entree = capital_ml
            nb_trades += 1

        if sort_maintenant:
            en_position = False
            pnl = capital_ml - capital_entree
            if pnl >= 0:
                trades_gagnants += 1
                gain_total += pnl
            else:
                trades_perdants += 1
                perte_totale += abs(pnl)

        plus_haut_ml = max(plus_haut_ml, capital_ml)
        dd = (capital_ml - plus_haut_ml) / plus_haut_ml * 100

        montant_investi = capital_ml * expo_ml

        if expo_ml > 0.6:
            signal_txt = "ACHAT"
        elif expo_ml < 0.4:
            signal_txt = "CASH"
        else:
            signal_txt = "NEUTRE"

        etat_robot = "ACTIF"
        if dd <= -10:
            etat_robot = "ARRET_TOTAL"
        elif dd <= -3:
            etat_robot = "VIGILANCE"

        date_str = date.strftime("%Y-%m-%d")
        etat["jour"] = i
        etat["capital"] = round(capital_ml, 2)
        etat["drawdown"] = round(dd, 2)
        etat["position"] = round(expo_ml, 3)
        etat["exposition_ml"] = round(expo_ml, 3)
        etat["montant_investi"] = round(montant_investi, 2)
        etat["etat"] = etat_robot
        etat["dernier_signal"] = signal_txt
        etat["nb_trades"] = nb_trades
        etat["trades_gagnants"] = trades_gagnants
        etat["trades_perdants"] = trades_perdants
        etat["gain_total"] = round(gain_total, 2)
        etat["perte_totale"] = round(perte_totale, 2)
        etat["prix_courant"] = round(p, 2)

        etat["equity"].append(round(capital_ml, 2))
        etat["equity_bh"].append(round(capital_bh, 2))
        etat["equity_ma"].append(round(capital_ma, 2))
        etat["dates"].append(date_str)
        etat["positions"].append(round(expo_ml, 3))
        etat["prix_liste"].append(round(p, 2))

        etat["regles_survie"]["drawdown"]["valeur"] = round(abs(dd), 2)
        etat["regles_survie"]["drawdown"]["ok"] = abs(dd) < 10
        etat["regles_survie"]["taille_pos"]["valeur"] = round(expo_ml * 25, 1)
        etat["regles_survie"]["taille_pos"]["ok"] = expo_ml <= 1.0

        if len(etat["equity"]) >= 252:
            eq = pd.Series(etat["equity"][-252:])
            r = eq.pct_change().dropna()
            if r.std() > 0:
                etat["sharpe"] = round(float(r.mean() / r.std() * np.sqrt(252)), 2)
            total_r = (capital_ml / CAPITAL_INIT) - 1
            etat["rendement"] = round(total_r * 100, 2)
            etat["pire_dd"] = round(float((eq / eq.cummax() - 1).min() * 100), 2)

        if entre_maintenant or sort_maintenant:
            pnl_trade = round(capital_ml - capital_entree, 2) if sort_maintenant else 0.0
            etat["journal"].append({
                "date": date_str,
                "action": "OUVRE" if entre_maintenant else "FERME",
                "type": signal_txt,
                "expo": round(expo_ml, 3),
                "montant": round(montant_investi, 2),
                "prix": round(p, 2),
                "capital": round(capital_ml, 2),
                "pnl": pnl_trade,
            })
            if len(etat["journal"]) > 80:
                etat["journal"] = etat["journal"][-80:]

        sauver_etat(etat)

        if i % 50 == 0:
            print(f"  Jour {i:4d}/{len(prix)} | {date_str} | "
                  f"Capital {capital_ml:,.0f} | DD {dd:.1f}% | "
                  f"Signal {signal_txt:6s} | Expo {expo_ml:.2f}")

        time.sleep(0.08)

    etat["termine"] = True
    sauver_etat(etat)
    print()
    print(f"[AGENT] Simulation terminee. {nb_trades} trades sur {len(prix)} jours.")
    print(f"[AGENT] Capital final : {capital_ml:.2f} $")
    print(f"[AGENT] Trades gagnants : {trades_gagnants} | Perdants : {trades_perdants}")
    win_rate = trades_gagnants / max(1, trades_gagnants + trades_perdants) * 100
    print(f"[AGENT] Taux de reussite : {win_rate:.1f}%")
    print(f"[AGENT] Gains : +{gain_total:.2f} $ | Pertes : -{perte_totale:.2f} $")


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/etat":
            try:
                data = ETAT_PATH.read_text()
            except FileNotFoundError:
                data = json.dumps(etat_initial())
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data.encode())
            return

        if self.path == "/" or self.path == "":
            self.path = "/dashboard.html"
        super().do_GET()

    def log_message(self, format, *args):
        pass


def main():
    ETAT_PATH.parent.mkdir(exist_ok=True)
    sauver_etat(etat_initial())

    server = http.server.HTTPServer(("", PORT), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[SERVEUR] Dashboard live sur http://localhost:{PORT}")
    print()

    try:
        simulation_agent()
        print("\n[SERVEUR] Simulation finie. Dashboard toujours actif (Ctrl+C pour quitter).")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVEUR] Arret.")
        server.shutdown()


if __name__ == "__main__":
    main()
