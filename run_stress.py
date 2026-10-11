"""
═══════════════════════════════════════════════════════════════
  Rex Clemens Capital — Banc de stress complet
═══════════════════════════════════════════════════════════════
  Exécute le banc de stress sur toutes les stratégies :
    1. Acheter et garder (référence)
    2. Croisement de moyennes mobiles
    3. Témoin aléatoire (NE DOIT PAS avoir l'air bon)

  Résultat : verdict réussi / échoué pour chaque stratégie.

  Usage : py run_stress.py
═══════════════════════════════════════════════════════════════
"""

import tomllib
from pathlib import Path

from src.data import generer_marche_aleatoire, separer_construction_test
from src.backtest import backtest
from src.stress import banc_de_stress
from src.report import rapport_backtest, rapport_stress
from src.strategies.buy_hold import BuyHold
from src.strategies.ma_cross import MACross
from src.strategies.random_control import RandomControl


def main():
    # ── Charger la configuration ──
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    # ── Données simulées (marche aléatoire) ──
    print("Génération des données simulées...")
    prix = generer_marche_aleatoire(periodes=2500, seed=1)
    construction, test_final = separer_construction_test(prix)

    # ── Stratégies à tester ──
    strategies = [
        BuyHold(),
        MACross(rapide=20, lente=100),
        RandomControl(seed=42),
    ]

    # ═══ PARTIE 1 : Backtest classique ═══
    print("\n" + "=" * 60)
    print("  PARTIE 1 — Backtest classique")
    print("=" * 60)

    for phase_nom, phase_prix in [("CONSTRUCTION", construction), ("TEST FINAL", test_final)]:
        print(f"\n--- {phase_nom} ({len(phase_prix)} jours) ---")
        resultats = []
        for strat in strategies:
            r = backtest(phase_prix, strat, cout=config["backtest"]["cout_par_trade"])
            resultats.append(r)

        print(rapport_backtest(resultats, titre=f"Backtest — {phase_nom}"))

    # ═══ PARTIE 2 : Banc de stress ═══
    print("\n" + "=" * 60)
    print("  PARTIE 2 — Banc de stress")
    print("=" * 60)

    n_mc = config.get("stress", {}).get("monte_carlo_iterations", 100)
    # Réduire pour la démo (sinon c'est long)
    n_mc_demo = min(n_mc, 100)
    print(f"  Monte Carlo : {n_mc_demo} itérations")
    print(f"  Crises historiques : 5 scénarios")
    print(f"  Chocs synthétiques : 8 scénarios")

    for strat in strategies:
        print(f"\n{'─' * 60}")
        print(f"  Stratégie : {strat.nom}")
        if strat.fiche:
            print(f"  Avantage : {strat.fiche.avantage}")
        print(f"{'─' * 60}")

        resultat = banc_de_stress(
            strat, config, prix_historique=construction,
            n_monte_carlo=n_mc_demo,
        )

        print(rapport_stress(resultat))
        print()

    # ═══ VERDICT FINAL ═══
    print("\n" + "=" * 60)
    print("  VERDICT FINAL")
    print("=" * 60)
    print()
    print("  Ce test utilise des données SIMULÉES (marche aléatoire).")
    print("  Sur des données aléatoires, AUCUNE stratégie ne devrait")
    print("  avoir un avantage réel et reproductible.")
    print()
    print("  Si le témoin aléatoire a l'air bon → le test est biaisé.")
    print("  Si le croisement de MA bat systématiquement le buy & hold")
    print("  sur des données aléatoires → c'est de la chance, pas")
    print("  un avantage réel.")
    print()
    print("  Prochaine étape : tester sur des VRAIES données (yfinance)")
    print("  et comparer sur plusieurs actifs et plusieurs périodes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
