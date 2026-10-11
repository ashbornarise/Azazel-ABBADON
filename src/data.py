"""
Module de données — Rex Clemens Capital
═══════════════════════════════════════
Chargement, nettoyage et contrôles de qualité des données.
Sources : yfinance (actions), ccxt (cryptos) — ajoutées quand nécessaire.
Pour l'instant : données simulées pour les tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def generer_marche_aleatoire(
    debut: str = "2015-01-01",
    periodes: int = 2500,
    prix_initial: float = 100.0,
    drift: float = 0.0003,
    volatilite: float = 0.01,
    seed: int = 1,
) -> pd.Series:
    """
    Génère une marche aléatoire (données simulées).
    À utiliser pour tester le code UNIQUEMENT.
    Un bon résultat sur des données aléatoires ne prouve rien.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(debut, periods=periodes)
    rendements = rng.normal(drift, volatilite, periodes)
    prix = prix_initial * np.exp(np.cumsum(rendements))
    return pd.Series(prix, index=idx, name="prix")


def separer_construction_test(
    prix: pd.Series,
    ratio: float = 0.70,
) -> tuple[pd.Series, pd.Series]:
    """
    Sépare les données en construction (70%) et test final (30%).
    Le test final ne doit JAMAIS être touché pendant la construction.
    """
    n = int(len(prix) * ratio)
    return prix.iloc[:n], prix.iloc[n:]


def verifier_qualite(prix: pd.Series) -> list[str]:
    """
    Contrôles de qualité sur les données de prix.
    Retourne une liste de problèmes détectés (vide = tout va bien).
    """
    problemes = []

    if prix.isna().any():
        n_nan = prix.isna().sum()
        problemes.append(f"{n_nan} valeurs manquantes (NaN)")

    if (prix <= 0).any():
        n_zero = (prix <= 0).sum()
        problemes.append(f"{n_zero} prix ≤ 0")

    # Doublons d'index
    if prix.index.duplicated().any():
        n_dup = prix.index.duplicated().sum()
        problemes.append(f"{n_dup} dates en double")

    # Sauts de prix > 50%
    rendements = prix.pct_change().dropna()
    sauts = rendements.abs() > 0.50
    if sauts.any():
        n_sauts = sauts.sum()
        problemes.append(f"{n_sauts} sauts de prix > 50%")

    return problemes


def nettoyer(prix: pd.Series) -> pd.Series:
    """Nettoyage basique : supprime NaN, dédoublonne, trie par date."""
    prix = prix.dropna()
    prix = prix[~prix.index.duplicated(keep="first")]
    prix = prix.sort_index()
    return prix
