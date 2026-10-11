"""
Moteur de backtest — Rex Clemens Capital
════════════════════════════════════════
Pas de bibliothèque externe : on écrit soi-même pour comprendre.
Règles :
  - Le signal du jour est exécuté le LENDEMAIN (pas de triche sur le futur).
  - Frais et glissement obligatoires.
  - Séparation construction / test final.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategies.base import StrategieBase
from .survie import ModuleSurvie


@dataclass
class ResultatBacktest:
    """Résultats d'un backtest."""
    nom_strategie: str
    equity: pd.Series
    buy_hold: pd.Series
    nb_trades: int
    rendement_annuel: float
    sharpe: float
    pire_baisse: float
    violations_survie: int

    def __str__(self):
        return (
            f"{self.nom_strategie:20s} | "
            f"rdt/an {self.rendement_annuel:6.1%} | "
            f"Sharpe {self.sharpe:5.2f} | "
            f"pire baisse {self.pire_baisse:6.1%} | "
            f"{self.nb_trades} trades | "
            f"violations survie: {self.violations_survie}"
        )


def calculer_stats(equity: pd.Series) -> tuple[float, float, float]:
    """Calcule rendement annualisé, ratio de Sharpe et pire drawdown."""
    r = equity.pct_change().dropna()
    if len(r) < 2:
        return 0.0, 0.0, 0.0
    years = len(r) / 252
    if years <= 0 or equity.iloc[-1] <= 0:
        return 0.0, 0.0, 0.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    maxdd = (equity / equity.cummax() - 1).min()
    return cagr, sharpe, maxdd


def backtest(
    prix: pd.Series,
    strategie: StrategieBase,
    cout: float = 0.001,
    module_survie: ModuleSurvie | None = None,
) -> ResultatBacktest:
    """
    Exécute un backtest.

    Args:
        prix: série de prix (index = dates).
        strategie: instance d'une stratégie.
        cout: frais + glissement par changement de position.
        module_survie: si fourni, filtre l'exposition via les règles de survie.

    Returns:
        ResultatBacktest avec toutes les métriques.
    """
    # Signal de la stratégie
    signal = strategie.calculer_exposition(prix)
    signal = strategie.valider_exposition(signal)

    # On agit le LENDEMAIN : pas de triche sur le futur
    position = signal.shift(1).fillna(0.0)

    # Appliquer le module de survie si fourni
    violations = 0
    if module_survie is not None:
        capital = module_survie.capital
        positions_filtrees = []
        for i, (date, expo) in enumerate(position.items()):
            prix_courant = prix.iloc[i]
            prix_prec = prix.iloc[i - 1] if i > 0 else None
            date_obj = date.date() if hasattr(date, "date") else None

            decision = module_survie.evaluer(
                exposition_demandee=expo,
                prix=prix_courant,
                prix_precedent=prix_prec,
                date=date_obj,
            )
            positions_filtrees.append(decision.exposition_autorisee)
            if decision.bloque or decision.reduit:
                violations += 1

            # Mettre à jour le capital simulé
            if i > 0 and prix.iloc[i - 1] > 0:
                ret_jour = (prix.iloc[i] - prix.iloc[i - 1]) / prix.iloc[i - 1]
                if np.isfinite(ret_jour):
                    capital *= (1 + positions_filtrees[-1] * ret_jour)
                module_survie.mettre_a_jour_capital(capital, date_obj)

        position = pd.Series(positions_filtrees, index=position.index)

    # Rendements — protéger contre NaN/inf (crises extrêmes, prix à 0)
    ret = prix.pct_change().fillna(0.0)
    ret = ret.replace([np.inf, -np.inf], 0.0)
    ret = ret.clip(lower=-0.99)  # un actif ne peut pas perdre plus de 99% en un jour

    trades = position.diff().abs().fillna(0.0)
    strat_ret = position * ret - trades * cout
    equity = (1 + strat_ret).cumprod()
    buy_hold_equity = (1 + ret).cumprod()

    # Stats
    cagr, sharpe, maxdd = calculer_stats(equity)

    return ResultatBacktest(
        nom_strategie=strategie.nom,
        equity=equity,
        buy_hold=buy_hold_equity,
        nb_trades=int(trades.sum()),
        rendement_annuel=cagr,
        sharpe=sharpe,
        pire_baisse=maxdd,
        violations_survie=violations,
    )


# ─── Point d'entrée pour test rapide ─────────────────────────

if __name__ == "__main__":
    from .strategies.ma_cross import MACross
    from .strategies.buy_hold import BuyHold
    from .strategies.random_control import RandomControl

    # Données simulées (marche aléatoire) — remplacer par de vraies données
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=2500)
    prices = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx)))),
        index=idx,
    )

    split = int(len(prices) * 0.70)

    for strategie in [BuyHold(), MACross(), RandomControl()]:
        for nom, p in [("construction", prices[:split]), ("test final", prices[split:])]:
            result = backtest(p, strategie)
            print(f"{nom:13s} {result}")
        print()
