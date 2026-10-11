"""
Rex Clemens Capital — Test sur donnees REELLES
═══════════════════════════════════════════════
Compare le ML, le Buy&Hold, le MA Cross et le temoin aleatoire
sur de vrais actifs : actions, crypto, devises.

Usage : py run_reel.py
"""

import tomllib

from src.data_reel import telecharger_tous, telecharger
from src.data import separer_construction_test, verifier_qualite
from src.backtest import backtest, calculer_stats
from src.strategies.ml_strategy import MLStrategy
from src.strategies.buy_hold import BuyHold
from src.strategies.ma_cross import MACross
from src.strategies.random_control import RandomControl


def tester_actif(nom, prix, config):
    """Teste les 4 strategies sur un actif."""
    cout = config["backtest"]["cout_par_trade"]

    # Qualite
    problemes = verifier_qualite(prix)
    if problemes:
        print(f"    Problemes: {', '.join(problemes)}")

    construction, test_final = separer_construction_test(prix)

    if len(construction) < 300:
        print(f"    Pas assez de donnees ({len(construction)} jours, min 300)")
        return None

    strategies = [
        MLStrategy(taille_min_train=252, pas_reentrain=21, n_arbres=50, seed=42),
        BuyHold(),
        MACross(rapide=20, lente=100),
        RandomControl(seed=42),
    ]

    resultats = {}
    for strat in strategies:
        try:
            r_constr = backtest(construction, strat, cout=cout)
            r_test = backtest(test_final, strat, cout=cout)
            resultats[strat.nom] = {
                "construction": r_constr,
                "test": r_test,
            }
        except Exception as e:
            print(f"    {strat.nom}: ERREUR - {e}")

    return resultats


def afficher_resultats(nom, resultats):
    """Affiche les resultats d'un actif."""
    print(f"\n  {'Strategie':<22s} | {'Phase':<13s} | {'Rdt/an':>8s} | {'Sharpe':>7s} | {'Max DD':>8s} | {'Trades':>6s}")
    print(f"  {'-'*22}-+-{'-'*13}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}")

    for snom, phases in resultats.items():
        for phase_nom, r in [("Construction", phases["construction"]), ("Test final", phases["test"])]:
            print(f"  {snom:<22s} | {phase_nom:<13s} | {r.rendement_annuel:>7.1%} | {r.sharpe:>6.2f} | {r.pire_baisse:>7.1%} | {r.nb_trades:>5d}")
        print()


def verdict(resultats):
    """Verdict honnete pour un actif."""
    if not resultats:
        return

    ml = resultats.get("ml_random_forest", {}).get("test")
    bh = resultats.get("acheter_garder", {}).get("test")
    rand = resultats.get("temoin_aleatoire", {}).get("test")

    if not ml or not bh or not rand:
        return

    print(f"  VERDICT:")
    if ml.sharpe <= rand.sharpe:
        print(f"    [ECHEC] Le ML ne bat pas le temoin aleatoire (Sharpe ML {ml.sharpe:.2f} vs Random {rand.sharpe:.2f})")
        print(f"    -> Pas d'avantage detecte.")
    elif ml.sharpe < bh.sharpe:
        print(f"    [MOYEN] Le ML bat le temoin ({ml.sharpe:.2f} vs {rand.sharpe:.2f}) mais pas le Buy&Hold ({bh.sharpe:.2f})")
        print(f"    -> Le modele capte un signal, mais pas assez pour justifier les frais.")
    elif ml.sharpe >= bh.sharpe and ml.sharpe > 0.5:
        print(f"    [BON] Le ML bat Buy&Hold ET le temoin (Sharpe {ml.sharpe:.2f})")
        print(f"    -> Signal potentiel. A CONFIRMER sur d'autres periodes.")
    else:
        print(f"    [PRUDENCE] Resultats mitiges (Sharpe ML {ml.sharpe:.2f})")


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    print("=" * 70)
    print("  REX CLEMENS CAPITAL — TEST SUR DONNEES REELLES")
    print("=" * 70)
    print()
    print("  RAPPELS:")
    print("  - Aucun argent reel")
    print("  - Les resultats passes ne garantissent PAS les gains futurs")
    print("  - Frais estimes (0.1%), frais reels probablement plus eleves")
    print("  - Risque de surajustement (overfitting)")
    print()

    # Telecharger les actifs
    actifs_a_tester = {
        "Apple (AAPL)": "AAPL",
        "Bitcoin (BTC-USD)": "BTC-USD",
        "Ethereum (ETH-USD)": "ETH-USD",
        "EUR/USD (EURUSD=X)": "EURUSD=X",
        "Microsoft (MSFT)": "MSFT",
        "Tesla (TSLA)": "TSLA",
        "Solana (SOL-USD)": "SOL-USD",
    }

    tous_resultats = {}

    for nom, symbole in actifs_a_tester.items():
        print(f"\n{'='*70}")
        print(f"  {nom}")
        print(f"{'='*70}")

        try:
            prix = telecharger(symbole, periode="5y")
            print(f"  {len(prix)} jours | {prix.index[0].strftime('%Y-%m-%d')} -> {prix.index[-1].strftime('%Y-%m-%d')}")
            print(f"  Prix: {prix.iloc[0]:.2f} -> {prix.iloc[-1]:.2f} ({(prix.iloc[-1]/prix.iloc[0]-1)*100:+.1f}%)")

            resultats = tester_actif(nom, prix, config)
            if resultats:
                afficher_resultats(nom, resultats)
                verdict(resultats)
                tous_resultats[nom] = resultats
        except Exception as e:
            print(f"  ERREUR: {e}")

    # Resume final
    print(f"\n{'='*70}")
    print(f"  RESUME — TOUS LES ACTIFS (TEST FINAL)")
    print(f"{'='*70}")
    print(f"\n  {'Actif':<25s} | {'ML Sharpe':>10s} | {'B&H Sharpe':>10s} | {'Random':>8s} | {'ML gagne?'}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}")

    for nom, resultats in tous_resultats.items():
        ml = resultats.get("ml_random_forest", {}).get("test")
        bh = resultats.get("acheter_garder", {}).get("test")
        rand = resultats.get("temoin_aleatoire", {}).get("test")
        if ml and bh and rand:
            gagne = "OUI" if ml.sharpe > bh.sharpe and ml.sharpe > rand.sharpe else "NON"
            print(f"  {nom:<25s} | {ml.sharpe:>9.2f} | {bh.sharpe:>9.2f} | {rand.sharpe:>7.2f} | {gagne}")

    print()
    print("  Un robot qui bat le Buy&Hold sur des donnees passees")
    print("  ne le battra pas forcement demain.")
    print("  Le temoin aleatoire est la pour le rappeler.")
    print("=" * 70)


if __name__ == "__main__":
    main()
