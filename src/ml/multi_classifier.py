"""
Multi-Classifier Benchmark — Rex Clemens Capital
==================================================
Inspire de freqtrade (FreqAI) et PyTrader (9 classifiers).
Compare RF, LightGBM, XGBoost, GradientBoosting et selectionne
automatiquement le meilleur modele par validation walk-forward.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


def _try_import_lgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except ImportError:
        return None


def _try_import_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        return None


def _try_import_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError:
        return None


@dataclass
class ClassifierResult:
    nom: str
    accuracy_val: float
    log_loss_val: float
    proba_hausse: float
    feature_importance: dict


def construire_classifieurs(seed: int = 42) -> dict:
    classifieurs = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=20,
            random_state=seed, n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, random_state=seed,
        ),
    }

    LGBMClassifier = _try_import_lgbm()
    if LGBMClassifier:
        classifieurs["LightGBM"] = LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            min_child_samples=20, random_state=seed,
            verbose=-1, n_jobs=1,
        )

    XGBClassifier = _try_import_xgb()
    if XGBClassifier:
        classifieurs["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            min_child_weight=20, random_state=seed,
            verbosity=0, n_jobs=1, use_label_encoder=False,
            eval_metric="logloss",
        )

    CatBoostCls = _try_import_catboost()
    if CatBoostCls:
        classifieurs["CatBoost"] = CatBoostCls(
            iterations=100, depth=5, learning_rate=0.05,
            random_seed=seed, verbose=0,
        )

    return classifieurs


def benchmark_classifieurs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_pred: np.ndarray,
    feat_names: list[str],
    seed: int = 42,
    classifieurs_custom: dict | None = None,
) -> tuple[ClassifierResult, dict[str, ClassifierResult]]:
    """
    Entraine tous les classifieurs, evalue sur validation,
    retourne le meilleur et le detail de chacun.
    Accepte classifieurs_custom (ex: issus d'Optuna) en priorite.
    """
    classifieurs = classifieurs_custom or construire_classifieurs(seed)
    resultats = {}

    for nom, clf in classifieurs.items():
        try:
            clf.fit(X_train, y_train)

            y_pred_val = clf.predict(X_val)
            y_proba_val = clf.predict_proba(X_val)
            acc = accuracy_score(y_val, y_pred_val)

            try:
                ll = log_loss(y_val, y_proba_val, labels=[0, 1])
            except Exception:
                ll = 999.0

            proba_pred = clf.predict_proba(X_pred)
            proba_h = float(proba_pred[0, 1]) if proba_pred.shape[1] == 2 else 0.5

            if hasattr(clf, "feature_importances_"):
                imp = clf.feature_importances_
                feat_imp = {
                    feat_names[i]: round(float(imp[i]) * 100, 1)
                    for i in range(len(feat_names))
                }
            else:
                feat_imp = {}

            resultats[nom] = ClassifierResult(
                nom=nom,
                accuracy_val=round(acc * 100, 1),
                log_loss_val=round(ll, 4),
                proba_hausse=round(proba_h * 100, 1),
                feature_importance=dict(sorted(feat_imp.items(), key=lambda x: -x[1])),
            )
        except Exception as e:
            print(f"  [BENCH] {nom}: {e}")

    if not resultats:
        return None, {}

    meilleur = min(resultats.values(), key=lambda r: r.log_loss_val)
    return meilleur, resultats


def seuil_dynamique(
    probas_historiques: np.ndarray,
    n_std: float = 1.0,
) -> tuple[float, float]:
    """
    Calcule les seuils d'achat/vente dynamiques : mean +/- N*std.
    Inspire de freqtrade freqai_test_strat (mean +/- 1.25*std).
    """
    if len(probas_historiques) < 20:
        return 55.0, 40.0

    mean = np.mean(probas_historiques)
    std = np.std(probas_historiques)
    seuil_achat = mean + n_std * std
    seuil_vente = mean - n_std * std

    seuil_achat = max(52.0, min(75.0, seuil_achat))
    seuil_vente = max(25.0, min(48.0, seuil_vente))

    return round(seuil_achat, 1), round(seuil_vente, 1)
