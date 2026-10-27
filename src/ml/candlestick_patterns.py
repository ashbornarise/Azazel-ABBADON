"""
Candlestick Pattern Detection — Rex Clemens Capital
=====================================================
Patterns extraits des Free Robots MetaTrader 5 (MetaQuotes).
Chaque pattern renvoie un score entre -1 (bearish) et +1 (bullish).
Pas de look-ahead : chaque signal au bar N utilise uniquement N et avant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _avg_body(o: pd.Series, c: pd.Series, period: int = 12) -> pd.Series:
    return (c - o).abs().rolling(period).mean()


def detect_engulfing(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
    sma_period: int = 5, body_period: int = 12,
) -> pd.Series:
    """Bullish/Bearish Engulfing (MT5 Free Robot pattern)."""
    body = (c - o).abs()
    avg_b = _avg_body(o, c, body_period)
    sma = c.rolling(sma_period).mean()
    mid_oc = (o + c) / 2

    bull = (
        (o.shift(1) > c.shift(1)) &
        ((c - o) > avg_b) &
        (c > o.shift(1)) &
        (mid_oc.shift(1) < sma.shift(1)) &
        (o < c.shift(1))
    )

    bear = (
        (o.shift(1) < c.shift(1)) &
        ((o - c) > avg_b) &
        (c < o.shift(1)) &
        (mid_oc.shift(1) > sma.shift(1)) &
        (o > c.shift(1))
    )

    score = pd.Series(0.0, index=o.index)
    score[bull] = 1.0
    score[bear] = -1.0
    return score


def detect_harami(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
    sma_period: int = 5, body_period: int = 12,
) -> pd.Series:
    """Bullish/Bearish Harami (MT5 Free Robot pattern)."""
    avg_b = _avg_body(o, c, body_period)
    sma = c.rolling(sma_period).mean()
    mid = (h + l) / 2

    bull = (
        (c > o) &
        ((o.shift(1) - c.shift(1)) > avg_b.shift(1)) &
        (c < o.shift(1)) &
        (o > c.shift(1)) &
        (mid.shift(1) < sma.shift(1))
    )

    bear = (
        (c < o) &
        ((c.shift(1) - o.shift(1)) > avg_b.shift(1)) &
        (c > o.shift(1)) &
        (o < c.shift(1)) &
        (mid.shift(1) > sma.shift(1))
    )

    score = pd.Series(0.0, index=o.index)
    score[bull] = 1.0
    score[bear] = -1.0
    return score


def detect_three_soldiers_crows(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
    body_period: int = 12,
) -> pd.Series:
    """3 White Soldiers / 3 Black Crows (MT5 Free Robot pattern)."""
    avg_b = _avg_body(o, c, body_period)
    mid = (h + l) / 2

    ws1 = (c.shift(2) - o.shift(2)) > avg_b.shift(2)
    ws2 = (c.shift(1) - o.shift(1)) > avg_b.shift(1)
    ws3 = (c - o) > avg_b
    ws_mid = (mid.shift(1) > mid.shift(2)) & (mid > mid.shift(1))
    soldiers = ws1 & ws2 & ws3 & ws_mid

    bc1 = (o.shift(2) - c.shift(2)) > avg_b.shift(2)
    bc2 = (o.shift(1) - c.shift(1)) > avg_b.shift(1)
    bc3 = (o - c) > avg_b
    bc_mid = (mid.shift(1) < mid.shift(2)) & (mid < mid.shift(1))
    crows = bc1 & bc2 & bc3 & bc_mid

    score = pd.Series(0.0, index=o.index)
    score[soldiers] = 1.0
    score[crows] = -1.0
    return score


def detect_darkcloud_piercing(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
    sma_period: int = 5, body_period: int = 12,
) -> pd.Series:
    """Dark Cloud Cover / Piercing Line (MT5 Free Robot pattern)."""
    avg_b = _avg_body(o, c, body_period)
    sma = c.rolling(sma_period).mean()
    mid_oc = (o + c) / 2

    piercing = (
        ((c - o) > avg_b) &
        ((o.shift(1) - c.shift(1)) > avg_b.shift(1)) &
        (c > c.shift(1)) &
        (c < o.shift(1)) &
        (mid_oc.shift(1) < sma.shift(1)) &
        (o < l.shift(1))
    )

    dark_cloud = (
        ((c.shift(1) - o.shift(1)) > avg_b.shift(1)) &
        (c < c.shift(1)) &
        (c > o.shift(1)) &
        (mid_oc.shift(1) > sma.shift(1)) &
        (o > h.shift(1))
    )

    score = pd.Series(0.0, index=o.index)
    score[piercing] = 1.0
    score[dark_cloud] = -1.0
    return score


def detect_macd_cross(
    prix: pd.Series,
    fast: int = 12, slow: int = 26, signal_period: int = 9,
) -> pd.Series:
    """MACD cross signal (MT5 MACD Sample EA pattern)."""
    ema_fast = prix.ewm(span=fast, adjust=False).mean()
    ema_slow = prix.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    ema_trend = prix.ewm(span=slow, adjust=False).mean()

    buy = (
        (macd < 0) &
        (macd > signal) &
        (macd.shift(1) < signal.shift(1)) &
        (ema_trend > ema_trend.shift(1))
    )

    sell = (
        (macd > 0) &
        (macd < signal) &
        (macd.shift(1) > signal.shift(1)) &
        (ema_trend < ema_trend.shift(1))
    )

    score = pd.Series(0.0, index=prix.index)
    score[buy] = 1.0
    score[sell] = -1.0
    return score


def calculer_patterns(
    prix: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    open_: pd.Series | None = None,
) -> pd.DataFrame:
    """Calcule tous les patterns et renvoie un DataFrame de scores."""
    o = open_ if open_ is not None else prix.shift(1)
    h = high if high is not None else prix
    l = low if low is not None else prix
    c = prix

    df = pd.DataFrame(index=prix.index)
    df["engulfing"] = detect_engulfing(o, h, l, c)
    df["harami"] = detect_harami(o, h, l, c)
    df["soldiers_crows"] = detect_three_soldiers_crows(o, h, l, c)
    df["darkcloud_piercing"] = detect_darkcloud_piercing(o, h, l, c)
    df["macd_cross"] = detect_macd_cross(c)

    df["pattern_score"] = (
        df["engulfing"] * 0.25
        + df["harami"] * 0.20
        + df["soldiers_crows"] * 0.25
        + df["darkcloud_piercing"] * 0.15
        + df["macd_cross"] * 0.15
    )

    return df
