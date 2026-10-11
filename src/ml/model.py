"""
Modèle ML avec walk-forward validation — Rex Clemens Capital
═════════════════════════════════════════════════════════════
Walk-forward : on entraîne sur le passé, on prédit le jour suivant,
on avance d'un pas, on recommence. Jamais de fuite du futur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class ResultatWalkForward:
    """Résultats de la validation walk-forward."""
    predictions: pd.Series
    probas: pd.Series
    accuracy: float
    n_entrainement: int
    n_prediction: int
    importance_features: dict[str, float]


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    taille_min_train: int = 252,
    pas: int = 21,
    n_arbres: int = 100,
    seed: int = 42,
) -> ResultatWalkForward:
    """
    Validation walk-forward (expanding window).

    À chaque pas :
      1. Entraîner sur tout le passé disponible (≥ taille_min_train jours)
      2. Prédire les `pas` jours suivants
      3. Avancer la fenêtre
    """
    predictions = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    probas = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    importances_cumulees = np.zeros(X.shape[1])
    n_modeles = 0

    for debut_test in range(taille_min_train, len(X), pas):
        fin_test = min(debut_test + pas, len(X))

        X_train = X.iloc[:debut_test]
        y_train = y.iloc[:debut_test]
        X_test = X.iloc[debut_test:fin_test]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=n_arbres,
            max_depth=5,
            min_samples_leaf=20,
            random_state=seed,
            n_jobs=1,
        )
        clf.fit(X_train_s, y_train)

        pred = clf.predict(X_test_s)
        proba = clf.predict_proba(X_test_s)
        proba_hausse = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]

        predictions = pd.concat([predictions, pd.Series(pred, index=X_test.index)])
        probas = pd.concat([probas, pd.Series(proba_hausse, index=X_test.index)])

        importances_cumulees += clf.feature_importances_
        n_modeles += 1

    importance_features = {}
    if n_modeles > 0:
        imp_moy = importances_cumulees / n_modeles
        for nom, imp in zip(X.columns, imp_moy):
            importance_features[nom] = float(imp)

    accuracy = 0.0
    if len(predictions) > 0:
        y_vrai = y.loc[predictions.index]
        accuracy = float((predictions == y_vrai).mean())

    return ResultatWalkForward(
        predictions=predictions,
        probas=probas,
        accuracy=accuracy,
        n_entrainement=taille_min_train,
        n_prediction=len(predictions),
        importance_features=importance_features,
    )
