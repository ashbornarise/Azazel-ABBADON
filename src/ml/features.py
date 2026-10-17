"""
Feature engineering pour le ML — Rex Clemens Capital
════════════════════════════════════════════════════
Toutes les features sont calculées à partir du PASSÉ uniquement.
Aucun regard vers le futur (look-ahead bias interdit).

V2 : 25 features inspirées de freqtrade, nautilus_trader, OpenBB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculer_features(prix: pd.Series, fenetre_max: int = 60) -> pd.DataFrame:
    """
    Calcule les features à partir d'une série de prix.
    Chaque feature au jour N utilise uniquement les données <= N.
    """
    df = pd.DataFrame(index=prix.index)
    ret = prix.pct_change()

    # === Bloc 1 : Momentum (déjà existant) ===
    df["ret_1j"] = ret
    df["ret_5j"] = prix.pct_change(5)
    df["ret_20j"] = prix.pct_change(20)

    # === Bloc 2 : Volatilité ===
    df["vol_10j"] = ret.rolling(10).std()
    df["vol_20j"] = ret.rolling(20).std()

    # === Bloc 3 : Moyennes mobiles ===
    ma5 = prix.rolling(5).mean()
    ma10 = prix.rolling(10).mean()
    ma20 = prix.rolling(20).mean()
    ma60 = prix.rolling(fenetre_max).mean()

    df["ma_ratio_5_20"] = ma5 / ma20 - 1
    df["ma_ratio_20_60"] = ma20 / ma60 - 1

    # === Bloc 4 : RSI ===
    delta = ret.copy()
    gain = delta.clip(lower=0).rolling(14).mean()
    perte = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / perte.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # === Bloc 5 : Drawdown et range ===
    df["dist_haut_20j"] = prix / prix.rolling(20).max() - 1
    df["amplitude_5j"] = (
        (prix.rolling(5).max() - prix.rolling(5).min())
        / prix.rolling(5).mean()
    )
    df["skew_20j"] = ret.rolling(20).skew()

    return df


def calculer_features_v2(
    prix: pd.Series,
    volume: pd.Series | None = None,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    open_: pd.Series | None = None,
    fenetre_max: int = 60,
) -> pd.DataFrame:
    """
    Features V2 : 25 indicateurs inspirés de freqtrade, nautilus, OpenBB.
    Accepte optionnellement OHLCV pour des indicateurs plus riches.
    """
    df = calculer_features(prix, fenetre_max)
    ret = prix.pct_change()

    # === MACD (freqtrade standard) ===
    ema12 = prix.ewm(span=12, adjust=False).mean()
    ema26 = prix.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd_line - macd_signal) / prix

    # === Stochastique (14 jours) ===
    low_ser = low if low is not None else prix
    high_ser = high if high is not None else prix
    low14 = low_ser.rolling(14).min()
    high14 = high_ser.rolling(14).max()
    denom = (high14 - low14).replace(0, np.nan)
    stoch_k = ((prix - low14) / denom) * 100
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_k.rolling(3).mean()

    # === Bollinger Bands (position relative) ===
    bb_ma = prix.rolling(20).mean()
    bb_std = prix.rolling(20).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"] = (prix - bb_lower) / bb_width

    # === ATR normalisé (Average True Range) ===
    if high is not None and low is not None:
        tr1 = high - low
        tr2 = (high - prix.shift(1)).abs()
        tr3 = (low - prix.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        tr = (prix.rolling(2).max() - prix.rolling(2).min()).fillna(0)
    df["atr_14"] = tr.rolling(14).mean() / prix

    # === ADX simplifié (force de tendance) ===
    ma10 = prix.rolling(10).mean()
    ma30 = prix.rolling(30).mean()
    trend_strength = ((ma10 - ma30) / prix).abs()
    df["trend_strength"] = trend_strength.rolling(14).mean()

    # === Momentum long terme ===
    df["ret_60j"] = prix.pct_change(60)

    # === Volume features ===
    if volume is not None and not volume.isna().all():
        vol_ma = volume.rolling(20).mean().replace(0, np.nan)
        df["volume_ratio"] = volume / vol_ma
        df["volume_trend"] = volume.rolling(5).mean() / vol_ma
    else:
        df["volume_ratio"] = 0.0
        df["volume_trend"] = 0.0

    # === Kurtosis (queues épaisses = risque) ===
    df["kurt_20j"] = ret.rolling(20).kurt()

    # === Ratio haussier (% de jours positifs sur 20j) ===
    df["pct_hausse_20j"] = (ret > 0).rolling(20).mean()

    # === Distance au plus bas 20j (rebond potentiel) ===
    df["dist_bas_20j"] = prix / prix.rolling(20).min() - 1

    # === ADX simplifie (force directionnelle, freqtrade recommendation #1) ===
    if high is not None and low is not None:
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        atr_s = tr.rolling(14).mean().replace(0, np.nan)
        plus_di = 100 * plus_dm.rolling(14).mean() / atr_s
        minus_di = 100 * minus_dm.rolling(14).mean() / atr_s
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        df["adx_14"] = dx.rolling(14).mean()
    else:
        df["adx_14"] = trend_strength * 1000

    # === Bollinger %B et Width (freqtrade recommendation #3) ===
    df["bb_width"] = (bb_upper - bb_lower) / bb_ma.replace(0, np.nan)

    # === Features temporelles (freqtrade recommendation #8) ===
    if hasattr(prix.index, "dayofweek"):
        df["jour_semaine"] = prix.index.dayofweek / 6.0
        df["heure"] = prix.index.hour / 23.0 if hasattr(prix.index, "hour") else 0.5
    else:
        df["jour_semaine"] = 0.5
        df["heure"] = 0.5

    # === Candlestick patterns (extraits des robots MetaTrader 5) ===
    from .candlestick_patterns import calculer_patterns
    patterns = calculer_patterns(prix, high=high, low=low, open_=open_)
    df["pattern_score"] = patterns["pattern_score"]
    df["engulfing"] = patterns["engulfing"]
    df["macd_cross"] = patterns["macd_cross"]

    # === CCI (Commodity Channel Index, MT5 Free Robots period=37) ===
    tp = prix if high is None else (high + low + prix) / 3
    tp_sma = tp.rolling(20).mean()
    mean_dev = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_20"] = (tp - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))

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
