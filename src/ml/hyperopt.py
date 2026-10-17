"""
Hyperparameter Optimization — Rex Clemens Capital
===================================================
Optuna-based hyperopt pour les 4 classifieurs (RF, GBM, LightGBM, XGBoost).
Inspiré de freqtrade Hyperopt + PyTrader grid search.
Optimise par LogLoss sur validation walk-forward.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class HyperoptResult:
    nom: str
    meilleurs_params: dict
    log_loss_val: float
    accuracy_val: float
    n_trials: int


def _try_import_optuna():
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna
    except ImportError:
        return None


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


def _objectif_rf(trial, X_train, y_train, X_val, y_val, seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 30),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
    clf = RandomForestClassifier(**params, random_state=seed, n_jobs=1)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)
    return log_loss(y_val, proba, labels=[0, 1])


def _objectif_gbm(trial, X_train, y_train, X_val, y_val, seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }
    clf = GradientBoostingClassifier(**params, random_state=seed)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)
    return log_loss(y_val, proba, labels=[0, 1])


def _objectif_lgbm(trial, X_train, y_train, X_val, y_val, seed):
    LGBMClassifier = _try_import_lgbm()
    if not LGBMClassifier:
        return 999.0
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    clf = LGBMClassifier(**params, random_state=seed, verbose=-1, n_jobs=1)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)
    return log_loss(y_val, proba, labels=[0, 1])


def _objectif_xgb(trial, X_train, y_train, X_val, y_val, seed):
    XGBClassifier = _try_import_xgb()
    if not XGBClassifier:
        return 999.0
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    clf = XGBClassifier(
        **params, random_state=seed, verbosity=0,
        n_jobs=1, use_label_encoder=False, eval_metric="logloss",
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)
    return log_loss(y_val, proba, labels=[0, 1])


OBJECTIFS = {
    "RandomForest": _objectif_rf,
    "GradientBoosting": _objectif_gbm,
    "LightGBM": _objectif_lgbm,
    "XGBoost": _objectif_xgb,
}


def optimiser_classifieur(
    nom: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    seed: int = 42,
) -> HyperoptResult | None:
    """Optimise un classifieur avec Optuna, retourne les meilleurs params."""
    optuna = _try_import_optuna()
    if optuna is None:
        return None

    if nom not in OBJECTIFS:
        return None

    objectif = OBJECTIFS[nom]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        lambda trial: objectif(trial, X_train, y_train, X_val, y_val, seed),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_trial
    from sklearn.metrics import accuracy_score
    clf = _reconstruire_classifieur(nom, best.params, seed)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_val, clf.predict(X_val))

    return HyperoptResult(
        nom=nom,
        meilleurs_params=best.params,
        log_loss_val=round(best.value, 4),
        accuracy_val=round(acc * 100, 1),
        n_trials=n_trials,
    )


def optimiser_tous(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    seed: int = 42,
) -> dict[str, HyperoptResult]:
    """Optimise tous les classifieurs disponibles, retourne les résultats."""
    resultats = {}
    for nom in OBJECTIFS:
        res = optimiser_classifieur(nom, X_train, y_train, X_val, y_val, n_trials, seed)
        if res is not None:
            resultats[nom] = res
    return resultats


def _reconstruire_classifieur(nom: str, params: dict, seed: int = 42):
    """Reconstruit un classifieur avec des params optimisés."""
    if nom == "RandomForest":
        return RandomForestClassifier(**params, random_state=seed, n_jobs=1)
    elif nom == "GradientBoosting":
        return GradientBoostingClassifier(**params, random_state=seed)
    elif nom == "LightGBM":
        LGBMClassifier = _try_import_lgbm()
        return LGBMClassifier(**params, random_state=seed, verbose=-1, n_jobs=1)
    elif nom == "XGBoost":
        XGBClassifier = _try_import_xgb()
        return XGBClassifier(
            **params, random_state=seed, verbosity=0,
            n_jobs=1, use_label_encoder=False, eval_metric="logloss",
        )
    raise ValueError(f"Classifieur inconnu: {nom}")


def construire_classifieurs_optimises(
    resultats_hyperopt: dict[str, HyperoptResult],
    seed: int = 42,
) -> dict:
    """Construit les classifieurs avec les params optimisés."""
    return {
        nom: _reconstruire_classifieur(nom, res.meilleurs_params, seed)
        for nom, res in resultats_hyperopt.items()
    }
