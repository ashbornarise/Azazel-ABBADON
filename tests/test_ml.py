"""
Tests du module ML — Rex Clemens Capital
════════════════════════════════════════
Vérifie les features, le walk-forward et la stratégie ML.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.features import calculer_features, preparer_xy
from src.ml.model import walk_forward
from src.strategies.ml_strategy import MLStrategy
from src.data import generer_marche_aleatoire


@pytest.fixture
def prix():
    return generer_marche_aleatoire(periodes=600, seed=1)


@pytest.fixture
def prix_long():
    return generer_marche_aleatoire(periodes=1500, seed=1)


# ── Tests features ──

class TestFeatures:
    def test_nb_colonnes(self, prix):
        f = calculer_features(prix)
        assert f.shape[1] == 11

    def test_meme_index(self, prix):
        f = calculer_features(prix)
        assert f.index.equals(prix.index)

    def test_pas_de_futur_dans_features(self, prix):
        f_complet = calculer_features(prix)
        f_partiel = calculer_features(prix.iloc[:100])
        cols = f_complet.columns
        idx = prix.index[99]
        for col in cols:
            val_c = f_complet.loc[idx, col]
            val_p = f_partiel.loc[idx, col]
            if pd.notna(val_c) and pd.notna(val_p):
                assert abs(val_c - val_p) < 1e-10, (
                    f"Feature {col} au jour 99 change quand on ajoute des données futures"
                )

    def test_valeurs_rsi_bornees(self, prix):
        f = calculer_features(prix)
        rsi = f["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()


# ── Tests preparer_xy ──

class TestPreparerXY:
    def test_xy_meme_index(self, prix):
        X, y = preparer_xy(prix)
        assert X.index.equals(y.index)

    def test_cible_binaire(self, prix):
        _, y = preparer_xy(prix)
        assert set(y.unique()).issubset({0.0, 1.0})

    def test_pas_de_nan(self, prix):
        X, y = preparer_xy(prix)
        assert not X.isna().any().any()
        assert not y.isna().any()


# ── Tests walk-forward ──

class TestWalkForward:
    def test_predictions_existent(self, prix):
        X, y = preparer_xy(prix)
        res = walk_forward(X, y, taille_min_train=200, pas=50, n_arbres=10)
        assert res.n_prediction > 0

    def test_accuracy_entre_0_et_1(self, prix):
        X, y = preparer_xy(prix)
        res = walk_forward(X, y, taille_min_train=200, pas=50, n_arbres=10)
        assert 0.0 <= res.accuracy <= 1.0

    def test_importance_features_complete(self, prix):
        X, y = preparer_xy(prix)
        res = walk_forward(X, y, taille_min_train=200, pas=50, n_arbres=10)
        assert len(res.importance_features) == X.shape[1]
        assert all(v >= 0 for v in res.importance_features.values())

    def test_probas_entre_0_et_1(self, prix):
        X, y = preparer_xy(prix)
        res = walk_forward(X, y, taille_min_train=200, pas=50, n_arbres=10)
        assert (res.probas >= 0).all() and (res.probas <= 1).all()


# ── Tests stratégie ML ──

class TestMLStrategy:
    def test_fiche_hypothese_remplie(self):
        strat = MLStrategy()
        assert strat.fiche is not None
        assert strat.fiche.avantage != ""
        assert strat.fiche.trades_minimum >= 100

    def test_exposition_entre_0_et_1(self, prix):
        strat = MLStrategy(taille_min_train=200, pas_reentrain=50, n_arbres=10)
        expo = strat.calculer_exposition(prix)
        assert (expo >= 0).all() and (expo <= 1).all()

    def test_meme_longueur_que_prix(self, prix):
        strat = MLStrategy(taille_min_train=200, pas_reentrain=50, n_arbres=10)
        expo = strat.calculer_exposition(prix)
        assert len(expo) == len(prix)

    def test_debut_a_zero(self, prix):
        """Avant d'avoir assez de données, l'exposition doit être 0."""
        strat = MLStrategy(taille_min_train=200, pas_reentrain=50, n_arbres=10)
        expo = strat.calculer_exposition(prix)
        assert expo.iloc[0] == 0.0

    def test_backtest_compatible(self, prix_long):
        """Vérifie que la stratégie ML fonctionne avec le moteur de backtest."""
        from src.backtest import backtest
        strat = MLStrategy(taille_min_train=252, pas_reentrain=50, n_arbres=10)
        resultat = backtest(prix_long, strat, cout=0.001)
        assert resultat.nom_strategie == "ml_random_forest"
        assert np.isfinite(resultat.sharpe)
        assert np.isfinite(resultat.rendement_annuel)
