"""
Liquidity Zones & Wyckoff Phase Detection — Rex Clemens Capital
================================================================
Cartographie les zones de liquidité (pools de stops, equal H/L,
niveaux ronds) et détecte les phases Wyckoff en temps réel.

Inspiré de ICT (Inner Circle Trader), Smart Money Concepts,
et Wyckoff Method (Richard Wyckoff, 1930).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ─── LIQUIDITY ZONES ────────────────────────────────────────────

@dataclass
class LiquidityZone:
    prix: float
    type: str  # "buy_side" (au-dessus, stops des shorts) ou "sell_side" (en-dessous, stops des longs)
    force: float  # 0-1, nombre de touches
    touches: int
    swept: bool = False
    ts_creation: str = ""
    ts_swept: str = ""


def detecter_equal_levels(series: pd.Series, tolerance_pct: float = 0.002, min_touches: int = 2) -> list[dict]:
    """
    Détecte les equal highs / equal lows — zones où le prix
    a touché le même niveau plusieurs fois (pool de liquidité).
    """
    if len(series) < 10:
        return []

    levels = []
    values = series.dropna().values
    used = set()

    for i in range(len(values)):
        if i in used:
            continue
        touches = [i]
        for j in range(i + 1, len(values)):
            if j in used:
                continue
            if abs(values[j] - values[i]) / max(values[i], 1e-10) < tolerance_pct:
                touches.append(j)
                used.add(j)
        if len(touches) >= min_touches:
            levels.append({
                "prix": round(float(np.mean([values[t] for t in touches])), 6),
                "touches": len(touches),
                "force": min(len(touches) / 5.0, 1.0),
                "indices": touches,
            })
    return levels


def detecter_swing_points(high: pd.Series, low: pd.Series, lookback: int = 5) -> tuple[pd.Series, pd.Series]:
    """
    Détecte les swing highs et swing lows.
    Un swing high = plus haut local (plus haut que les N barres avant ET après).
    """
    swing_highs = pd.Series(np.nan, index=high.index)
    swing_lows = pd.Series(np.nan, index=low.index)

    for i in range(lookback, len(high) - lookback):
        window_h = high.iloc[i - lookback:i + lookback + 1]
        if high.iloc[i] == window_h.max():
            swing_highs.iloc[i] = high.iloc[i]

        window_l = low.iloc[i - lookback:i + lookback + 1]
        if low.iloc[i] == window_l.min():
            swing_lows.iloc[i] = low.iloc[i]

    return swing_highs, swing_lows


def mapper_zones_liquidite(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    volume: pd.Series | None = None,
) -> dict:
    """
    Cartographie complète des zones de liquidité :
    1. Equal Highs (buy-side liquidity — stops des shorts au-dessus)
    2. Equal Lows (sell-side liquidity — stops des longs en-dessous)
    3. Round numbers (niveaux psychologiques)
    4. Previous session H/L
    5. Sweep detection
    """
    h = high if high is not None else close
    l = low if low is not None else close
    prix_actuel = float(close.iloc[-1])

    # 1. Equal Highs / Equal Lows via swing points
    swing_h, swing_l = detecter_swing_points(h, l)
    eq_highs = detecter_equal_levels(swing_h.dropna(), tolerance_pct=0.003, min_touches=2)
    eq_lows = detecter_equal_levels(swing_l.dropna(), tolerance_pct=0.003, min_touches=2)

    # 2. Niveaux ronds (psychologiques)
    round_levels = []
    if prix_actuel > 1:
        magnitude = 10 ** max(0, int(np.log10(prix_actuel)) - 1)
        base = int(prix_actuel / magnitude) * magnitude
        for mult in range(-3, 4):
            level = base + mult * magnitude
            if level > 0:
                round_levels.append({
                    "prix": float(level),
                    "type": "round_number",
                    "distance_pct": round((level - prix_actuel) / prix_actuel * 100, 2),
                })

    # 3. Previous session high/low (dernières 20 barres)
    recent_high = float(h.iloc[-20:].max())
    recent_low = float(l.iloc[-20:].min())

    # 4. Zones de liquidité composites
    zones = []

    for eq in eq_highs:
        swept = prix_actuel > eq["prix"]
        zones.append(LiquidityZone(
            prix=eq["prix"],
            type="buy_side",
            force=eq["force"],
            touches=eq["touches"],
            swept=swept,
        ))

    for eq in eq_lows:
        swept = prix_actuel < eq["prix"]
        zones.append(LiquidityZone(
            prix=eq["prix"],
            type="sell_side",
            force=eq["force"],
            touches=eq["touches"],
            swept=swept,
        ))

    # 5. Nearest unswept zones (les plus dangereuses/intéressantes)
    buy_side_unswept = [z for z in zones if z.type == "buy_side" and not z.swept]
    sell_side_unswept = [z for z in zones if z.type == "sell_side" and not z.swept]

    nearest_buy = min(buy_side_unswept, key=lambda z: z.prix - prix_actuel, default=None) if buy_side_unswept else None
    nearest_sell = max(sell_side_unswept, key=lambda z: prix_actuel - z.prix, default=None) if sell_side_unswept else None

    return {
        "prix_actuel": prix_actuel,
        "zones": zones,
        "equal_highs": eq_highs,
        "equal_lows": eq_lows,
        "round_levels": round_levels,
        "session_high": recent_high,
        "session_low": recent_low,
        "nearest_buy_side": nearest_buy,
        "nearest_sell_side": nearest_sell,
        "nb_zones": len(zones),
    }


# ─── WYCKOFF PHASES ─────────────────────────────────────────────

WYCKOFF_PHASES = {
    "accumulation": {
        "description": "Smart money achète discrètement — prix en range bas",
        "events": ["PS", "SC", "AR", "ST", "Spring", "Test", "SOS", "LPS", "BU"],
        "bias": "bullish",
    },
    "markup": {
        "description": "Tendance haussière — demand > supply",
        "events": ["SOS", "BU", "LPS"],
        "bias": "bullish",
    },
    "distribution": {
        "description": "Smart money vend discrètement — prix en range haut",
        "events": ["PSY", "BC", "AR", "SOW", "UT", "UTAD", "LPSY"],
        "bias": "bearish",
    },
    "markdown": {
        "description": "Tendance baissière — supply > demand",
        "events": ["SOW", "LPSY"],
        "bias": "bearish",
    },
}


@dataclass
class WyckoffEvent:
    type: str       # PS, SC, AR, ST, Spring, SOS, etc.
    index: int
    prix: float
    volume_ratio: float
    description: str


@dataclass
class WyckoffAnalysis:
    phase: str          # accumulation, markup, distribution, markdown
    confidence: float   # 0-1
    bias: str           # bullish, bearish, neutral
    events: list[WyckoffEvent] = field(default_factory=list)
    support: float = 0.0
    resistance: float = 0.0
    description: str = ""


def detecter_phase_wyckoff(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    volume: pd.Series | None = None,
    lookback: int = 50,
) -> WyckoffAnalysis:
    """
    Analyse Wyckoff complète : détecte la phase actuelle et les événements.

    Schématique Wyckoff :
    ┌─────────────────────────────────────────────────┐
    │  Distribution                                    │
    │  PSY → BC → AR → SOW → UT/UTAD → LPSY          │
    │                                      ↓           │
    │                                   MARKDOWN       │
    │                                      ↓           │
    │  Accumulation                                    │
    │  PS → SC → AR → ST → Spring → SOS → LPS → BU   │
    │                                      ↓           │
    │                                    MARKUP        │
    └─────────────────────────────────────────────────┘
    """
    h = high if high is not None else close
    l = low if low is not None else close
    c = close
    n = min(lookback, len(c))
    recent = slice(-n, None)

    c_r = c.iloc[recent]
    h_r = h.iloc[recent]
    l_r = l.iloc[recent]

    support = float(l_r.min())
    resistance = float(h_r.max())
    range_size = resistance - support
    if range_size == 0:
        range_size = 1e-10

    prix_position = (float(c.iloc[-1]) - support) / range_size

    # Volume analysis
    has_volume = volume is not None and not volume.isna().all()
    if has_volume:
        v_r = volume.iloc[recent]
        vol_mean = float(v_r.mean()) or 1.0
        vol_trend = float(v_r.iloc[-10:].mean()) / float(v_r.iloc[:10].mean() or 1.0)
    else:
        vol_mean = 1.0
        vol_trend = 1.0

    # Price trend
    ret_total = float(c.iloc[-1] / c.iloc[-n] - 1) if c.iloc[-n] != 0 else 0
    volatility = float(c_r.pct_change().std()) if len(c_r) > 5 else 0.01

    # Range-bound detection
    mid_range = support + range_size * 0.5
    range_bound = float((c_r - mid_range).abs().mean()) / range_size < 0.25

    events = []

    # ─── Event Detection ───

    # Preliminary Support (PS) : premier rallye après chute
    if len(c_r) > 20:
        first_half = c_r.iloc[:n // 2]
        if float(first_half.pct_change().sum()) < -0.05:
            # Chute significative dans la première moitié
            for i in range(n // 4, n // 2):
                if float(c_r.iloc[i]) > float(c_r.iloc[i - 1]) and float(c_r.iloc[i - 1]) < float(c_r.iloc[i - 2]):
                    events.append(WyckoffEvent("PS", i, float(c_r.iloc[i]), 1.0, "Premier support après chute"))
                    break

    # Selling Climax (SC) : pic de volume + prix au plus bas
    if has_volume:
        for i in range(5, n):
            is_low = float(l_r.iloc[i]) <= float(l_r.iloc[max(0, i - 10):i + 1].min())
            vol_spike = float(v_r.iloc[i]) > vol_mean * 2.0
            if is_low and vol_spike:
                events.append(WyckoffEvent("SC", i, float(c_r.iloc[i]),
                                           float(v_r.iloc[i]) / vol_mean,
                                           "Selling Climax — volume extrême + prix au plus bas"))
                break

    # Spring : prix casse le support puis remonte (piège baissier)
    for i in range(10, n - 1):
        broke_support = float(l_r.iloc[i]) < support * 1.001
        closed_above = float(c_r.iloc[i]) > support
        if broke_support and closed_above:
            vol_ratio = float(v_r.iloc[i]) / vol_mean if has_volume else 1.0
            events.append(WyckoffEvent("Spring", i, float(c_r.iloc[i]), vol_ratio,
                                       "Spring — faux breakout sous le support"))

    # Upthrust / UTAD : prix casse la résistance puis redescend (piège haussier)
    for i in range(10, n - 1):
        broke_resist = float(h_r.iloc[i]) > resistance * 0.999
        closed_below = float(c_r.iloc[i]) < resistance
        if broke_resist and closed_below:
            vol_ratio = float(v_r.iloc[i]) / vol_mean if has_volume else 1.0
            events.append(WyckoffEvent("UT", i, float(c_r.iloc[i]), vol_ratio,
                                       "Upthrust — faux breakout au-dessus de la résistance"))

    # Sign of Strength (SOS) : breakout haussier avec volume
    if len(c_r) > 5:
        last_5 = c_r.iloc[-5:]
        if float(last_5.iloc[-1]) > resistance * 0.98:
            if has_volume and float(v_r.iloc[-5:].mean()) > vol_mean * 1.3:
                events.append(WyckoffEvent("SOS", n - 1, float(c.iloc[-1]),
                                           float(v_r.iloc[-1]) / vol_mean if has_volume else 1.0,
                                           "Sign of Strength — breakout avec volume"))

    # Sign of Weakness (SOW) : breakdown baissier avec volume
    if len(c_r) > 5:
        if float(c_r.iloc[-1]) < support * 1.02:
            if has_volume and float(v_r.iloc[-5:].mean()) > vol_mean * 1.3:
                events.append(WyckoffEvent("SOW", n - 1, float(c.iloc[-1]),
                                           float(v_r.iloc[-1]) / vol_mean if has_volume else 1.0,
                                           "Sign of Weakness — breakdown avec volume"))

    # ─── Phase Scoring ───
    scores = {
        "accumulation": 0.0,
        "markup": 0.0,
        "distribution": 0.0,
        "markdown": 0.0,
    }

    # Position dans le range
    if prix_position < 0.3:
        scores["accumulation"] += 0.3
    elif prix_position > 0.7:
        scores["distribution"] += 0.3
    if prix_position > 0.5:
        scores["markup"] += 0.1
    if prix_position < 0.5:
        scores["markdown"] += 0.1

    # Range-bound → accumulation ou distribution
    if range_bound:
        if prix_position < 0.5:
            scores["accumulation"] += 0.2
        else:
            scores["distribution"] += 0.2
    else:
        if ret_total > 0:
            scores["markup"] += 0.3
        else:
            scores["markdown"] += 0.3

    # Volume trend
    if has_volume:
        if vol_trend < 0.8:
            scores["accumulation"] += 0.15
            scores["distribution"] += 0.15
        elif vol_trend > 1.3:
            scores["markup"] += 0.15
            scores["markdown"] += 0.15

    # Events bonus
    event_types = {e.type for e in events}
    if "Spring" in event_types:
        scores["accumulation"] += 0.25
    if "SC" in event_types:
        scores["accumulation"] += 0.2
    if "SOS" in event_types:
        scores["markup"] += 0.25
        scores["accumulation"] += 0.1
    if "UT" in event_types:
        scores["distribution"] += 0.25
    if "SOW" in event_types:
        scores["markdown"] += 0.25
        scores["distribution"] += 0.1

    # Trend momentum
    if ret_total > 0.05:
        scores["markup"] += 0.2
    elif ret_total < -0.05:
        scores["markdown"] += 0.2

    # Normalize
    total = sum(scores.values()) or 1.0
    for k in scores:
        scores[k] /= total

    best_phase = max(scores, key=scores.get)
    confidence = scores[best_phase]
    bias = WYCKOFF_PHASES[best_phase]["bias"]

    return WyckoffAnalysis(
        phase=best_phase,
        confidence=round(confidence, 2),
        bias=bias,
        events=events,
        support=round(support, 6),
        resistance=round(resistance, 6),
        description=WYCKOFF_PHASES[best_phase]["description"],
    )


def analyse_complete(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    volume: pd.Series | None = None,
) -> dict:
    """Point d'entrée unique : liquidité + Wyckoff."""
    liq = mapper_zones_liquidite(close, high, low, volume)
    wyckoff = detecter_phase_wyckoff(close, high, low, volume)

    return {
        "liquidite": liq,
        "wyckoff": {
            "phase": wyckoff.phase,
            "confidence": wyckoff.confidence,
            "bias": wyckoff.bias,
            "support": wyckoff.support,
            "resistance": wyckoff.resistance,
            "events": [{"type": e.type, "prix": e.prix, "vol_ratio": e.volume_ratio, "desc": e.description} for e in wyckoff.events],
            "description": wyckoff.description,
        },
    }
