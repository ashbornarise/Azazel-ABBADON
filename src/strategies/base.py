"""
Interface de base pour les stratégies de trading.
══════════════════════════════════════════════════
Une stratégie ne renvoie qu'une exposition cible entre 0 et 1.
Elle ne peut pas contourner la couche de survie.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class FicheHypothese:
    """
    Fiche d'hypothèse obligatoire avant d'écrire une stratégie.
    Une stratégie sans fiche remplie n'est pas acceptée.
    """
    avantage: str           # Quel avantage cette stratégie est-elle censée avoir ?
    pourquoi_existe: str    # Pourquoi cet avantage existerait-il ?
    qui_perd: str           # Qui perd de l'autre côté du trade ?
    fin_avantage: str       # Qu'est-ce qui la ferait cesser de marcher ?
    trades_minimum: int     # Combien de trades pour distinguer avantage vs chance ?


class StrategieBase(ABC):
    """
    Classe de base pour toutes les stratégies.

    Une stratégie :
    - reçoit des données de prix historiques
    - renvoie une exposition cible entre 0.0 (cash) et 1.0 (tout investi)
    - ne peut PAS renvoyer > 1.0 (pas de levier) ni < 0.0 (pas de short)
    - ne doit JAMAIS accéder à des données futures (triche)
    """

    nom: str = "base"
    fiche: Optional[FicheHypothese] = None

    @abstractmethod
    def calculer_exposition(self, prix: pd.Series) -> pd.Series:
        """
        Calcule la série d'exposition cible.

        Args:
            prix: série de prix historiques (index = dates).

        Returns:
            Série de même longueur, valeurs entre 0.0 et 1.0.
            Le signal du jour N est exécuté le jour N+1 (pas de triche).
        """
        ...

    def valider_exposition(self, exposition: pd.Series) -> pd.Series:
        """Clamp l'exposition entre 0 et 1 (sécurité)."""
        return exposition.clip(lower=0.0, upper=1.0)
