"""
Rex Clemens Capital — Execution quotidienne
════════════════════════════════════════════
Lance une fois par jour : telecharge les prix, calcule le signal ML,
sauvegarde le resultat. Concu pour PythonAnywhere (tache planifiee).

Usage : py run_daily.py
"""

import json
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_reel import telecharger
from src.ml.features import calculer_features
from src.strategies.ml_strategy import MLStrategy
from src.strategies.buy_hold import BuyHold
from src.strategies.ma_cross import MACross
from src.backtest import backtest

ETAT_PATH = Path("data/etat_daily.json")
JOURNAL_PATH = Path("data/journal_daily.jsonl")

ACTIFS_SUIVIS = {
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


def analyser_actif(symbole, nom, config):
    """Analyse un actif et retourne le signal du jour."""
    prix = telecharger(symbole, periode="2y", cache=False)

    if len(prix) < 300:
        return None

    cout = config["backtest"]["cout_par_trade"]
    ml = MLStrategy(taille_min_train=252, pas_reentrain=21, n_arbres=50, seed=42)

    signal = ml.calculer_exposition(prix)
    expo_aujourd = float(signal.iloc[-1])
    prix_aujourd = float(prix.iloc[-1])

    r_ml = backtest(prix, ml, cout=cout)
    r_bh = backtest(prix, BuyHold(), cout=cout)

    if expo_aujourd >= 0.6:
        decision = "ACHAT"
    elif expo_aujourd <= 0.4:
        decision = "CASH"
    else:
        decision = "NEUTRE"

    return {
        "symbole": symbole,
        "nom": nom,
        "prix": round(prix_aujourd, 2),
        "exposition": round(expo_aujourd, 3),
        "decision": decision,
        "sharpe_ml": round(r_ml.sharpe, 2),
        "sharpe_bh": round(r_bh.sharpe, 2),
        "rendement_ml": round(r_ml.rendement_annuel * 100, 1),
        "max_dd_ml": round(r_ml.pire_baisse * 100, 1),
    }


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    ETAT_PATH.parent.mkdir(exist_ok=True)
    maintenant = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"[{maintenant}] Analyse quotidienne Rex Clemens Capital")
    print()

    resultats = []
    for symbole, nom in ACTIFS_SUIVIS.items():
        print(f"  Analyse {nom} ({symbole})...")
        try:
            r = analyser_actif(symbole, nom, config)
            if r:
                resultats.append(r)
                print(f"    Prix: {r['prix']} | Signal: {r['decision']} | Expo: {r['exposition']:.0%} | Sharpe: {r['sharpe_ml']}")
        except Exception as e:
            print(f"    ERREUR: {e}")

    etat = {
        "date": maintenant,
        "actifs": resultats,
        "nb_achat": sum(1 for r in resultats if r["decision"] == "ACHAT"),
        "nb_cash": sum(1 for r in resultats if r["decision"] == "CASH"),
        "nb_neutre": sum(1 for r in resultats if r["decision"] == "NEUTRE"),
    }

    with open(ETAT_PATH, "w") as f:
        json.dump(etat, f, cls=NumpyEncoder, indent=2)

    with open(JOURNAL_PATH, "a") as f:
        f.write(json.dumps(etat, cls=NumpyEncoder) + "\n")

    print()
    print(f"  Resume: {etat['nb_achat']} ACHAT | {etat['nb_cash']} CASH | {etat['nb_neutre']} NEUTRE")
    print(f"  Sauvegarde: {ETAT_PATH}")
    print()
    print("  RAPPEL: simulation uniquement, aucun argent reel.")


if __name__ == "__main__":
    main()
