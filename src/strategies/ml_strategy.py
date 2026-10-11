"""
Stratégie ML (Random Forest) — Rex Clemens Capital
═══════════════════════════════════════════════════
Utilise un modèle de machine learning pour estimer la probabilité
de hausse du lendemain. L'exposition = probabilité de hausse.

Walk-forward : le modèle est ré-entraîné périodiquement sur
le passé uniquement. Aucune fuite du futur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .base import StrategieBase, FicheHypothese
from ..ml.features import calculer_features


class MLStrategy(StrategieBase):
    """
    Stratégie basée sur un Random Forest.
    Exposition = probabilité estimée de hausse du lendemain.
    """

    nom = "ml_random_forest"
    fiche = FicheHypothese(
        avantage="Combiner plusieurs signaux techniques (momentum, volatilité, "
                 "tendance, RSI) via un modèle non-linéaire pour estimer la "
                 "probabilité de hausse.",
        pourquoi_existe="Un modèle non-linéaire peut capter des combinaisons "
                        "de signaux que les règles simples ne voient pas. "
                        "Reste à prouver que cet avantage persiste hors échantillon.",
        qui_perd="Les traders qui utilisent un seul indicateur ou qui "
                 "réagissent en retard aux changements de régime.",
        fin_avantage="Surajustement (overfitting), changement de régime du marché, "
                     "frais trop élevés, ou adoption massive des mêmes features.",
        trades_minimum=200,
    )

    def __init__(
        self,
        taille_min_train: int = 252,
        pas_reentrain: int = 21,
        n_arbres: int = 100,
        seed: int = 42,
    ):
        self.taille_min_train = taille_min_train
        self.pas_reentrain = pas_reentrain
        self.n_arbres = n_arbres
        self.seed = seed

    def calculer_exposition(self, prix: pd.Series) -> pd.Series:
        features = calculer_features(prix)
        exposition = pd.Series(0.0, index=prix.index)

        masque_valide = features.notna().all(axis=1)
        idx_valides = features.index[masque_valide]

        if len(idx_valides) < self.taille_min_train + self.pas_reentrain:
            return self.valider_exposition(exposition)

        ret_demain = prix.pct_change().shift(-1)
        cible = (ret_demain > 0).astype(float)

        clf = None
        scaler = None
        dernier_entrainement = -1

        for i, date in enumerate(idx_valides):
            pos_dans_valides = i

            besoin_entrain = (
                clf is None
                or (pos_dans_valides - dernier_entrainement) >= self.pas_reentrain
            )

            if besoin_entrain and pos_dans_valides >= self.taille_min_train:
                X_train = features.loc[idx_valides[:pos_dans_valides]]
                y_train = cible.loc[idx_valides[:pos_dans_valides]]

                valide_train = y_train.notna()
                X_train = X_train[valide_train]
                y_train = y_train[valide_train]

                if len(X_train) >= self.taille_min_train // 2:
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)

                    clf = RandomForestClassifier(
                        n_estimators=self.n_arbres,
                        max_depth=5,
                        min_samples_leaf=20,
                        random_state=self.seed,
                        n_jobs=1,
                    )
                    clf.fit(X_train_s, y_train)
                    dernier_entrainement = pos_dans_valides

            if clf is not None and scaler is not None:
                X_pred = features.loc[[date]]
                X_pred_s = scaler.transform(X_pred)
                proba = clf.predict_proba(X_pred_s)
                proba_hausse = proba[0, 1] if proba.shape[1] == 2 else 0.5
                exposition.loc[date] = proba_hausse

        return self.valider_exposition(exposition)
