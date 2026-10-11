"""Stratégie de référence : acheter et garder."""

import pandas as pd

from .base import StrategieBase, FicheHypothese


class BuyHold(StrategieBase):
    """
    Référence passive : exposition à 1.0 en permanence.
    Toute stratégie active doit justifier pourquoi elle ferait mieux.
    """

    nom = "acheter_garder"
    fiche = FicheHypothese(
        avantage="Aucun — c'est la référence passive.",
        pourquoi_existe="Le marché tend à monter sur le très long terme (prime de risque actions).",
        qui_perd="Personne : on ne fait que suivre le marché.",
        fin_avantage="Marché baissier prolongé.",
        trades_minimum=0,
    )

    def calculer_exposition(self, prix: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=prix.index)
