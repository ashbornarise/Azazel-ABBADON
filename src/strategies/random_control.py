"""
Stratégie témoin aléatoire.
═══════════════════════════
Cette stratégie ne doit JAMAIS avoir l'air bonne.
Si elle a l'air bonne, c'est que le banc de test est biaisé.
"""

import numpy as np
import pandas as pd

from .base import StrategieBase, FicheHypothese


class RandomControl(StrategieBase):
    """
    Stratégie témoin : exposition aléatoire (0 ou 1) chaque jour.
    Sert de contrôle : si elle a un bon Sharpe, c'est le signe
    que le test est biaisé ou que la période est trompeuse.
    """

    nom = "temoin_aleatoire"
    fiche = FicheHypothese(
        avantage="AUCUN — c'est un contrôle aléatoire.",
        pourquoi_existe="Il n'existe pas. C'est le point.",
        qui_perd="Le propriétaire du compte, en frais.",
        fin_avantage="N'a jamais marché — c'est un contrôle.",
        trades_minimum=0,
    )

    def __init__(self, seed: int = 42):
        self.seed = seed

    def calculer_exposition(self, prix: pd.Series) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        return pd.Series(
            rng.choice([0.0, 1.0], size=len(prix)),
            index=prix.index,
        )
