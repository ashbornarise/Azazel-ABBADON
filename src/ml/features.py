"""
Feature engineering pour le ML — Rex Clemens Capital
════════════════════════════════════════════════════
Toutes les features sont calculées à partir du PASSÉ uniquement.
Aucun regard vers le futur (look-ahead bias interdit).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculer_features(prix: pd.Series, fenetre_max: int = 60) -> pd.DataFrame:
    """
    Calcule les features à partir d'une série de prix.

    Chaque feature au jour N utilise uniquement les données ≤ N.
    """
    df = pd.DataFrame(index=prix.index)

    ret = prix.pct_change()

    # Rendements passés (momentum court terme)
    df["ret_1j"] = ret
    df["ret_5j"] = prix.pct_change(5)
    df["ret_20j"] = prix.pct_change(20)

    # Volatilité réalisée (écart-type des rendements)
    df["vol_10j"] = ret.rolling(10).std()
    df["vol_20j"] = ret.rolling(20).std()

    # Ratio de moyennes mobiles (position relative dans la tendance)
    ma5 = prix.rolling(5).mean()
    ma20 = prix.rolling(20).mean()
    ma60 = prix.rolling(fenetre_max).mean()

    df["ma_ratio_5_20"] = ma5 / ma20 - 1
    df["ma_ratio_20_60"] = ma20 / ma60 - 1

    # RSI simplifié (14 jours)
    delta = ret.copy()
    gain = delta.clip(lower=0).rolling(14).mean()
    perte = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / perte.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Distance au plus haut sur 20 jours (drawdown local)
    df["dist_haut_20j"] = prix / prix.rolling(20).max() - 1

    # Amplitude journalière normalisée (proxy de volatilité intraday)
    df["amplitude_5j"] = (
        (prix.rolling(5).max() - prix.rolling(5).min())
        / prix.rolling(5).mean()
    )

    # Skewness des rendements récents
    df["skew_20j"] = ret.rolling(20).skew()

    return df


def preparer_xy(
    prix: pd.Series,
    fenetre_max: int = 60,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prépare X (features) et y (cible) pour l'entraînement.

    Cible : le rendement du LENDEMAIN est-il positif ?
    (binaire : 1 = hausse, 0 = baisse)

    Le décalage garantit qu'on prédit l'avenir, pas le présent.
    """
    features = calculer_features(prix, fenetre_max)

    # Cible = rendement du jour SUIVANT > 0
    ret_demain = prix.pct_change().shift(-1)
    cible = (ret_demain > 0).astype(float)

    # Supprimer les lignes incomplètes
    masque = features.notna().all(axis=1) & cible.notna()
    return features[masque], cible[masque]
