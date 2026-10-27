"""
REX CLEMENS — Strategies EA Extraites
======================================
Strategies reverse-engineered depuis les meilleurs robots MT5 :
- Gold_PRO_V2 : ZigZag market structure + Pin Bar + ATR zones + EMA200 trend
- Gold_Trend_Revers : RSI mean-reversion + MA50 trend + consecutive candles

Ces strategies sont les ARMES EXCLUSIVES de DeepSeek.
Cortex n'y a pas acces. Avantage asymetrique.
"""

import numpy as np
from typing import Optional


def detecter_pin_bar(opens, highs, lows, closes, index: int = -1,
                     pin_ratio: float = 2.8) -> dict:
    """Pin Bar detection (Gold_PRO_V2 strategy).

    Un pin bar = meche longue dans une direction, petit corps.
    Signal de retournement puissant utilise par les meilleurs EA Gold.
    """
    i = index if index >= 0 else len(closes) + index
    if i < 1 or i >= len(closes):
        return {"signal": 0, "type": "none"}

    o, h, l, c = opens[i], highs[i], lows[i], closes[i]
    body = max(abs(o - c), 1e-10)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    if lower_shadow > body * pin_ratio and lower_shadow > upper_shadow:
        bullish_body = c > o
        return {
            "signal": 1.0 if bullish_body else 0.7,
            "type": "bullish_pin",
            "body": round(body, 4),
            "lower_shadow": round(lower_shadow, 4),
            "ratio": round(lower_shadow / body, 1),
        }

    if upper_shadow > body * pin_ratio and upper_shadow > lower_shadow:
        bearish_body = c < o
        return {
            "signal": -1.0 if bearish_body else -0.7,
            "type": "bearish_pin",
            "body": round(body, 4),
            "upper_shadow": round(upper_shadow, 4),
            "ratio": round(upper_shadow / body, 1),
        }

    return {"signal": 0, "type": "none"}


def detecter_structure_marche(highs, lows, lookback: int = 50) -> dict:
    """Market structure detection via swing points (ZigZag simplifie).

    Gold_PRO_V2 logic : Higher Highs + Higher Lows = uptrend (+1)
                        Lower Highs + Lower Lows = downtrend (-1)
    """
    n = len(highs)
    if n < lookback:
        return {"trend": 0, "h1": 0, "h2": 0, "l1": 0, "l2": 0}

    swing_highs = []
    swing_lows = []
    window = 5

    for i in range(window, n - window):
        if all(highs[i] >= highs[i - j] for j in range(1, window + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, min(window + 1, n - i))):
            swing_highs.append((i, highs[i]))
        if all(lows[i] <= lows[i - j] for j in range(1, window + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, min(window + 1, n - i))):
            swing_lows.append((i, lows[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"trend": 0, "h1": 0, "h2": 0, "l1": 0, "l2": 0}

    h1 = swing_highs[-1][1]
    h2 = swing_highs[-2][1]
    l1 = swing_lows[-1][1]
    l2 = swing_lows[-2][1]

    if h1 > h2 and l1 > l2:
        trend = 1
    elif h1 < h2 and l1 < l2:
        trend = -1
    else:
        trend = 0

    return {
        "trend": trend,
        "h1": round(h1, 4),
        "h2": round(h2, 4),
        "l1": round(l1, 4),
        "l2": round(l2, 4),
    }


def calculer_atr(highs, lows, closes, period: int = 14) -> float:
    """ATR (Average True Range) — mesure de volatilite (Gold_PRO_V2)."""
    n = len(closes)
    if n < period + 1:
        return 0.0

    trs = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

    if len(trs) < period:
        return 0.0
    return float(np.mean(trs[-period:]))


def detecter_consecutive_candles(opens, closes, min_count: int = 3,
                                 min_size_pct: float = 0.001) -> dict:
    """Consecutive bullish/bearish candles (Gold_Trend_Revers strategy).

    3+ bougies haussières consecutives de taille minimum = signal BUY fort.
    5+ bougies baissières = signal SELL (le reversal attend l'epuisement).
    """
    n = len(closes)
    if n < min_count + 1:
        return {"bullish_streak": 0, "bearish_streak": 0, "signal": 0}

    bull_count = 0
    bear_count = 0

    for i in range(n - 1, max(n - 10, 0), -1):
        body_pct = abs(closes[i] - opens[i]) / max(closes[i], 1e-10)
        if closes[i] > opens[i] and body_pct >= min_size_pct:
            if bear_count == 0:
                bull_count += 1
            else:
                break
        elif closes[i] < opens[i] and body_pct >= min_size_pct:
            if bull_count == 0:
                bear_count += 1
            else:
                break
        else:
            break

    signal = 0
    if bull_count >= 3:
        signal = min(bull_count / 5.0, 1.0)
    elif bear_count >= 3:
        signal = -min(bear_count / 5.0, 1.0)

    return {
        "bullish_streak": bull_count,
        "bearish_streak": bear_count,
        "signal": round(signal, 2),
    }


def calculer_rsi(closes, period: int = 14) -> float:
    """RSI — Relative Strength Index (Gold_Trend_Revers, period 14)."""
    if len(closes) < period + 1:
        return 50.0

    gains, losses = [], []
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(abs(diff))

    avg_gain = sum(gains) / period if gains else 0.001
    avg_loss = sum(losses) / period if losses else 0.001
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)


def analyser_ea_complet(opens, highs, lows, closes, volumes=None) -> dict:
    """Analyse complete avec toutes les strategies EA extraites.

    Retourne un dictionnaire complet pour alimenter le prompt de DeepSeek.
    """
    n = len(closes)
    if n < 60:
        return {"disponible": False}

    pin = detecter_pin_bar(opens, highs, lows, closes)
    structure = detecter_structure_marche(highs, lows)
    atr = calculer_atr(highs, lows, closes)
    candles = detecter_consecutive_candles(opens, closes)
    rsi = calculer_rsi(closes)

    ma50 = sum(closes[-50:]) / 50
    ma200 = sum(closes[-200:]) / 200 if n >= 200 else sum(closes[-50:]) / 50
    prix = closes[-1]

    gold_pro_signal = 0
    if structure["trend"] == 1 and pin["signal"] > 0 and prix > ma200:
        gold_pro_signal = 1
    elif structure["trend"] == -1 and pin["signal"] < 0 and prix < ma200:
        gold_pro_signal = -1

    trend_revers_signal = 0
    if prix > ma50 and rsi > 30 and candles["bullish_streak"] >= 3:
        trend_revers_signal = 1
    elif prix < ma50 and rsi < 70 and candles["bearish_streak"] >= 5:
        trend_revers_signal = -1

    atr_pct = round(atr / max(prix, 1e-10) * 100, 3)

    return {
        "disponible": True,
        "gold_pro_v2": {
            "signal": gold_pro_signal,
            "structure_trend": structure["trend"],
            "pin_bar": pin["type"],
            "pin_signal": pin["signal"],
            "atr_pct": atr_pct,
            "prix_vs_ema200": "above" if prix > ma200 else "below",
        },
        "trend_revers": {
            "signal": trend_revers_signal,
            "rsi": rsi,
            "prix_vs_ma50": "above" if prix > ma50 else "below",
            "bullish_streak": candles["bullish_streak"],
            "bearish_streak": candles["bearish_streak"],
        },
        "score_combine": round((gold_pro_signal + trend_revers_signal) / 2, 2),
    }
