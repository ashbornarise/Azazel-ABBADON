"""Stratégie de croisement de moyennes mobiles."""

import pandas as pd

from .base import StrategieBase, FicheHypothese


class MACross(StrategieBase):
    """
    Croisement de moyennes mobiles.
    Signal = 1 quand la MA rapide > MA lente, 0 sinon.
    """

    nom = "croisement_ma"
    fiche = FicheHypothese(
        avantage="Capturer les tendances de moyen terme en étant investi "
                 "quand la tendance courte dépasse la tendance longue.",
        pourquoi_existe="Les tendances de prix persistent parfois à cause du comportement "
                        "moutonnier des investisseurs et des flux institutionnels.",
        qui_perd="Les investisseurs qui achètent au sommet ou vendent au creux, "
                 "et les market makers dans les marchés directionnels.",
        fin_avantage="Marchés sans tendance (range), frais trop élevés par rapport "
                     "au gain capturé, ou adoption massive de la même stratégie.",
        trades_minimum=100,
    )

    def __init__(self, rapide: int = 20, lente: int = 100):
        self.rapide = rapide
        self.lente = lente

    def calculer_exposition(self, prix: pd.Series) -> pd.Series:
        ma_rapide = prix.rolling(self.rapide).mean()
        ma_lente = prix.rolling(self.lente).mean()
        signal = (ma_rapide > ma_lente).astype(float)
        return self.valider_exposition(signal)
