"""Tests pour les modules V2 : protections, multi-classifier, candlestick patterns, hyperopt."""

import numpy as np
import pandas as pd
import pytest

from src.ml.features import calculer_features_v2
from src.ml.protections import (
    GestionnaireProtections,
    MaxDrawdownProtection,
    StoplossGuard,
    CooldownProtection,
    FeeFilter,
    TechnicalStopLoss,
)
from src.ml.multi_classifier import (
    construire_classifieurs,
    benchmark_classifieurs,
    seuil_dynamique,
)
from src.ml.candlestick_patterns import (
    detect_engulfing,
    detect_harami,
    detect_three_soldiers_crows,
    detect_macd_cross,
    calculer_patterns,
)
from src.ml.hyperopt import (
    optimiser_classifieur,
    optimiser_tous,
    construire_classifieurs_optimises,
)


@pytest.fixture
def prix_ohlcv():
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    np.random.seed(42)
    close = pd.Series(np.cumsum(np.random.randn(400)) + 100, index=dates)
    high = close + np.random.uniform(0.5, 2, 400)
    low = close - np.random.uniform(0.5, 2, 400)
    open_ = close + np.random.uniform(-1, 1, 400)
    volume = pd.Series(np.random.randint(1000, 5000, 400), index=dates, dtype=float)
    return close, high, low, open_, volume


class TestFeaturesV2:
    def test_nombre_features(self, prix_ohlcv):
        close, high, low, open_, volume = prix_ohlcv
        feat = calculer_features_v2(close, volume=volume, high=high, low=low, open_=open_)
        assert feat.shape[1] == 31

    def test_pas_de_nan_total(self, prix_ohlcv):
        close, high, low, open_, volume = prix_ohlcv
        feat = calculer_features_v2(close, volume=volume, high=high, low=low, open_=open_)
        valid = feat.dropna()
        assert len(valid) > 100

    def test_features_v2_sans_ohlcv(self, prix_ohlcv):
        close, *_ = prix_ohlcv
        feat = calculer_features_v2(close)
        assert feat.shape[1] == 31


class TestProtections:
    def test_max_drawdown_ok(self):
        p = MaxDrawdownProtection(seuil_pct=-5.0)
        bloque, _ = p.est_bloque(drawdown_pct=-3.0)
        assert not bloque

    def test_max_drawdown_bloque(self):
        p = MaxDrawdownProtection(seuil_pct=-5.0)
        bloque, raison = p.est_bloque(drawdown_pct=-6.0)
        assert bloque
        assert "MaxDrawdown" in raison

    def test_stoploss_guard(self):
        s = StoplossGuard(pertes_max=2, cooldown_seconds=1)
        s.enregistrer_trade(-1.0)
        bloque, _ = s.est_bloque()
        assert not bloque
        s.enregistrer_trade(-1.0)
        bloque, raison = s.est_bloque()
        assert bloque
        assert "StoplossGuard" in raison

    def test_stoploss_reset_sur_gain(self):
        s = StoplossGuard(pertes_max=3)
        s.enregistrer_trade(-1.0)
        s.enregistrer_trade(-1.0)
        s.enregistrer_trade(5.0)
        bloque, _ = s.est_bloque()
        assert not bloque

    def test_fee_filter_rentable(self):
        f = FeeFilter(fee_pct=0.1)
        ok, _ = f.est_rentable(0.50, 100.0)
        assert ok

    def test_fee_filter_pas_rentable(self):
        f = FeeFilter(fee_pct=0.1)
        ok, raison = f.est_rentable(0.001, 100.0)
        assert not ok
        assert "FeeFilter" in raison

    def test_gestionnaire_global(self):
        g = GestionnaireProtections()
        ok, _ = g.peut_trader("BTC-USD", -2.0)
        assert ok
        ok, raison = g.peut_trader("BTC-USD", -6.0)
        assert not ok


class TestMultiClassifier:
    def test_classifieurs_disponibles(self):
        clf = construire_classifieurs()
        assert "RandomForest" in clf
        assert "GradientBoosting" in clf
        assert len(clf) >= 2

    def test_benchmark(self, prix_ohlcv):
        close, high, low, open_, volume = prix_ohlcv
        feat = calculer_features_v2(close, volume=volume, high=high, low=low, open_=open_)
        valid = feat.dropna()
        ret = close.pct_change().shift(-1)
        cible = (ret > 0).astype(float)

        from sklearn.preprocessing import StandardScaler
        X = valid.iloc[:200]
        y = cible.loc[X.index].dropna()
        X = X.loc[y.index]
        X_val = valid.iloc[200:250]
        y_val = cible.loc[X_val.index].dropna()
        X_val = X_val.loc[y_val.index]

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        X_v = scaler.transform(X_val)
        X_p = scaler.transform(valid.iloc[[-1]])

        meilleur, tous = benchmark_classifieurs(
            X_s, y.values, X_v, y_val.values, X_p, list(feat.columns),
        )
        assert meilleur is not None
        assert len(tous) >= 2
        assert 0 <= meilleur.proba_hausse <= 100

    def test_seuil_dynamique(self):
        probas = np.random.uniform(40, 60, 50)
        sa, sv = seuil_dynamique(probas)
        assert sa > sv
        assert 52 <= sa <= 75
        assert 25 <= sv <= 48

    def test_seuil_dynamique_court(self):
        sa, sv = seuil_dynamique(np.array([50.0, 55.0]))
        assert sa == 55.0
        assert sv == 40.0


class TestCandlestickPatterns:
    def test_engulfing(self, prix_ohlcv):
        close, high, low, open_, _ = prix_ohlcv
        scores = detect_engulfing(open_, high, low, close)
        assert scores.dtype == float
        assert scores.min() >= -1.0
        assert scores.max() <= 1.0

    def test_harami(self, prix_ohlcv):
        close, high, low, open_, _ = prix_ohlcv
        scores = detect_harami(open_, high, low, close)
        assert scores.dtype == float

    def test_macd_cross(self, prix_ohlcv):
        close, *_ = prix_ohlcv
        scores = detect_macd_cross(close)
        n_signals = (scores != 0).sum()
        assert n_signals >= 0

    def test_calculer_patterns_complet(self, prix_ohlcv):
        close, high, low, open_, _ = prix_ohlcv
        df = calculer_patterns(close, high=high, low=low, open_=open_)
        assert "pattern_score" in df.columns
        assert "engulfing" in df.columns
        assert "macd_cross" in df.columns
        assert len(df) == len(close)


class TestHyperopt:
    def test_optimiser_rf(self, prix_ohlcv):
        close, high, low, open_, volume = prix_ohlcv
        feat = calculer_features_v2(close, volume=volume, high=high, low=low, open_=open_)
        valid = feat.dropna()
        ret = close.pct_change().shift(-1)
        cible = (ret > 0).astype(float)

        from sklearn.preprocessing import StandardScaler
        X = valid.iloc[:200]
        y = cible.loc[X.index].dropna()
        X = X.loc[y.index]
        X_val = valid.iloc[200:250]
        y_val = cible.loc[X_val.index].dropna()
        X_val = X_val.loc[y_val.index]

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        X_v = scaler.transform(X_val)

        res = optimiser_classifieur("RandomForest", X_s, y.values, X_v, y_val.values, n_trials=5)
        assert res is not None
        assert res.nom == "RandomForest"
        assert res.log_loss_val < 2.0
        assert len(res.meilleurs_params) >= 3

    def test_optimiser_tous(self, prix_ohlcv):
        close, high, low, open_, volume = prix_ohlcv
        feat = calculer_features_v2(close, volume=volume, high=high, low=low, open_=open_)
        valid = feat.dropna()
        ret = close.pct_change().shift(-1)
        cible = (ret > 0).astype(float)

        from sklearn.preprocessing import StandardScaler
        X = valid.iloc[:200]
        y = cible.loc[X.index].dropna()
        X = X.loc[y.index]
        X_val = valid.iloc[200:250]
        y_val = cible.loc[X_val.index].dropna()
        X_val = X_val.loc[y_val.index]

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        X_v = scaler.transform(X_val)

        tous = optimiser_tous(X_s, y.values, X_v, y_val.values, n_trials=3)
        assert len(tous) >= 2
        clf_opt = construire_classifieurs_optimises(tous)
        assert "RandomForest" in clf_opt


class TestTechnicalStopLoss:
    def test_stop_fixe_declenche(self):
        sl = TechnicalStopLoss(stop_fixe_pct=-5.0, take_profit_pct=20.0)
        sl.ouvrir_position("BTC", 100.0)
        stop, _ = sl.verifier_stop("BTC", 96.0)
        assert stop is False
        stop, _ = sl.verifier_stop("BTC", 95.01)
        assert stop is False
        stop, raison = sl.verifier_stop("BTC", 94.99)
        assert stop is True
        assert "StopLoss fixe" in raison

    def test_trailing_stop_declenche(self):
        sl = TechnicalStopLoss(stop_fixe_pct=-10.0, trailing_pct=-3.0, trailing_activation_pct=2.0)
        sl.ouvrir_position("ETH", 100.0)
        sl.mettre_a_jour_prix("ETH", 103.0)
        stop, _ = sl.verifier_stop("ETH", 103.0)
        assert stop is False
        stop, raison = sl.verifier_stop("ETH", 99.9)
        assert stop is True
        assert "TrailingStop" in raison

    def test_trailing_pas_actif_sous_seuil(self):
        sl = TechnicalStopLoss(stop_fixe_pct=-10.0, trailing_pct=-3.0, trailing_activation_pct=2.0)
        sl.ouvrir_position("SOL", 100.0)
        sl.mettre_a_jour_prix("SOL", 101.5)
        stop, _ = sl.verifier_stop("SOL", 98.5)
        assert stop is False

    def test_fermer_position(self):
        sl = TechnicalStopLoss()
        sl.ouvrir_position("BTC", 100.0)
        sl.fermer_position("BTC")
        stop, _ = sl.verifier_stop("BTC", 80.0)
        assert stop is False

    def test_take_profit_declenche(self):
        sl = TechnicalStopLoss(stop_fixe_pct=-10.0, take_profit_pct=8.0)
        sl.ouvrir_position("BTC", 100.0)
        stop, _ = sl.verifier_stop("BTC", 107.0)
        assert stop is False
        stop, raison = sl.verifier_stop("BTC", 108.0)
        assert stop is True
        assert "TakeProfit" in raison

    def test_gestionnaire_inclut_stop_technique(self):
        g = GestionnaireProtections(stop_fixe_pct=-5.0, trailing_pct=-3.0)
        assert g.stop_technique is not None
        g.stop_technique.ouvrir_position("BTC", 100.0)
        stop, raison = g.stop_technique.verifier_stop("BTC", 94.0)
        assert stop is True
