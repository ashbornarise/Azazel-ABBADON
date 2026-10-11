"""
Agent ML — Rex Clemens Capital
══════════════════════════════
Orchestre l'entraînement, l'évaluation et le rapport du modèle ML.

Usage :
    py -m src.ml.agent
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import generer_marche_aleatoire, separer_construction_test
from ..backtest import backtest, calculer_stats
from ..strategies.ml_strategy import MLStrategy
from ..strategies.buy_hold import BuyHold
from ..strategies.random_control import RandomControl
from .features import preparer_xy
from .model import walk_forward


def rapport_ml(resultat_wf, resultat_backtest) -> str:
    """Génère un rapport lisible du modèle ML."""
    lignes = [
        "=" * 60,
        "  RAPPORT ML -- Random Forest Walk-Forward",
        "=" * 60,
        "",
        f"  Precision (accuracy) : {resultat_wf.accuracy:.1%}",
        f"  Predictions totales  : {resultat_wf.n_prediction}",
        f"  Taille entrainement  : {resultat_wf.n_entrainement} jours min",
        "",
        "  Importance des features :",
    ]

    imp_triees = sorted(
        resultat_wf.importance_features.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for nom, imp in imp_triees:
        barre = "#" * int(imp * 50)
        lignes.append(f"    {nom:20s} {imp:.3f} {barre}")

    lignes.extend([
        "",
        f"  Rendement annualise  : {resultat_backtest.rendement_annuel:.1%}",
        f"  Sharpe               : {resultat_backtest.sharpe:.2f}",
        f"  Pire drawdown        : {resultat_backtest.pire_baisse:.1%}",
        f"  Trades               : {resultat_backtest.nb_trades}",
        "",
        "  AVERTISSEMENTS :",
        "  - Ce test utilise des donnees SIMULEES.",
        "  - Un bon resultat ici ne prouve PAS que le modele marchera.",
        "  - Risque majeur : surajustement (overfitting).",
        "  - Les frais reels seront probablement plus eleves.",
        "  - Comparer TOUJOURS au temoin aleatoire.",
        "=" * 60,
    ])

    return "\n".join(lignes)


def run_agent():
    """Point d'entrée de l'agent ML."""
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    print("Generation des donnees simulees...")
    prix = generer_marche_aleatoire(periodes=2500, seed=1)
    construction, test_final = separer_construction_test(prix)

    # ── Walk-forward sur la période de construction ──
    print("\nWalk-forward sur la periode de construction...")
    X, y = preparer_xy(construction)
    resultat_wf = walk_forward(X, y, taille_min_train=252, pas=21)

    # ── Backtest de la stratégie ML ──
    print("\nBacktest des strategies...")
    cout = config["backtest"]["cout_par_trade"]

    ml_strat = MLStrategy(taille_min_train=252, pas_reentrain=21)
    bh_strat = BuyHold()
    rand_strat = RandomControl(seed=42)

    strategies = [ml_strat, bh_strat, rand_strat]

    for phase_nom, phase_prix in [("CONSTRUCTION", construction), ("TEST FINAL", test_final)]:
        print(f"\n--- {phase_nom} ({len(phase_prix)} jours) ---")
        for strat in strategies:
            r = backtest(phase_prix, strat, cout=cout)
            print(f"  {r}")

    # ── Rapport ML ──
    r_ml = backtest(test_final, ml_strat, cout=cout)
    print("\n" + rapport_ml(resultat_wf, r_ml))

    # ── Verdict honnête ──
    r_rand = backtest(test_final, rand_strat, cout=cout)
    print("\n" + "=" * 60)
    print("  VERDICT HONNETE")
    print("=" * 60)

    if r_ml.sharpe <= r_rand.sharpe:
        print("  [ECHEC] Le ML ne bat PAS le temoin aleatoire.")
        print("  -> Le modele n'a probablement pas d'avantage reel.")
    elif r_ml.sharpe < 0.5:
        print("  [PRUDENCE] Le ML bat le temoin mais le Sharpe reste faible.")
        print("  -> Prudence, peut etre du bruit.")
    else:
        print("  [OK] Le ML bat le temoin avec un Sharpe raisonnable.")
        print("  -> A confirmer sur des VRAIES donnees.")

    print()
    print("  Rappel : sur des donnees SIMULEES, aucun avantage")
    print("  ne devrait etre reel et reproductible.")
    print("=" * 60)


if __name__ == "__main__":
    run_agent()
