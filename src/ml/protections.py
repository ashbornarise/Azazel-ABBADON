"""
Protections de trading — Rex Clemens Capital
=============================================
Inspirees de freqtrade + EA31337 + AdaptiveTrader Pro + MACD_RSI_STOCH V5.
Ces protections ARRETENT le trading quand les conditions de survie
sont menacees. Elles ne peuvent pas etre desactivees par la strategie.

V2 : ATR sizing, drawdown circuit breaker avec recovery, trailing par paliers,
     SL structurel (swing low/high), filtre sessions, limite gain/perte journaliere,
     equity conditional tasks.

V3 : +300 EA analysis (13 repos, 2 .mq5 sources, 106 .ex5 catalogues).
     Nouveaux filtres de survie : spread gate, max hold time, D1 alignment,
     partial close ladder (40/30/30), Bollinger squeeze gate,
     mass index reversal bulge, weighted history scoring,
     breakout extension guard + retest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class TradeRecord:
    sym: str
    action: str
    prix: float
    pnl: float
    timestamp: datetime


@dataclass
class MaxDrawdownProtection:
    """Arrete tout trading si le drawdown depasse un seuil."""
    seuil_pct: float = -5.0
    fenetre_trades: int = 10

    def est_bloque(self, drawdown_pct: float, **_) -> tuple[bool, str]:
        if drawdown_pct <= self.seuil_pct:
            return True, f"MaxDrawdown: {drawdown_pct:.1f}% <= {self.seuil_pct}%"
        return False, ""


@dataclass
class StoplossGuard:
    """Bloque le trading apres N pertes consecutives."""
    pertes_max: int = 3
    cooldown_seconds: int = 600

    _pertes_consecutives: int = field(default=0, init=False, repr=False)
    _bloque_jusqua: datetime | None = field(default=None, init=False, repr=False)

    def enregistrer_trade(self, pnl: float):
        if pnl < 0:
            self._pertes_consecutives += 1
            if self._pertes_consecutives >= self.pertes_max:
                self._bloque_jusqua = datetime.now(timezone.utc)
        else:
            self._pertes_consecutives = 0

    def est_bloque(self, **_) -> tuple[bool, str]:
        if self._bloque_jusqua is not None:
            elapsed = (datetime.now(timezone.utc) - self._bloque_jusqua).total_seconds()
            if elapsed < self.cooldown_seconds:
                reste = int(self.cooldown_seconds - elapsed)
                return True, f"StoplossGuard: {self._pertes_consecutives} pertes, pause {reste}s"
            self._pertes_consecutives = 0
            self._bloque_jusqua = None
        return False, ""


@dataclass
class CooldownProtection:
    """Impose une pause apres chaque trade."""
    cooldown_seconds: int = 300

    _derniers_trades: dict = field(default_factory=dict, init=False, repr=False)

    def enregistrer_trade(self, sym: str):
        self._derniers_trades[sym] = datetime.now(timezone.utc)

    def est_bloque(self, sym: str = "", **_) -> tuple[bool, str]:
        if sym in self._derniers_trades:
            elapsed = (datetime.now(timezone.utc) - self._derniers_trades[sym]).total_seconds()
            if elapsed < self.cooldown_seconds:
                reste = int(self.cooldown_seconds - elapsed)
                return True, f"Cooldown {sym}: pause {reste}s"
        return False, ""


@dataclass
class FeeFilter:
    """Bloque un trade si le gain potentiel ne couvre pas 2x les frais."""
    fee_pct: float = 0.1

    def gain_minimum(self, montant: float) -> float:
        return montant * self.fee_pct / 100.0 * 2.0

    def est_rentable(self, pnl_estime: float, montant: float) -> tuple[bool, str]:
        seuil = self.gain_minimum(montant)
        if pnl_estime < seuil:
            return False, f"FeeFilter: gain {pnl_estime:.4f}$ < 2x fees {seuil:.4f}$"
        return True, ""


@dataclass
class TechnicalStopLoss:
    """Stop-loss + take-profit technique par trade."""
    stop_fixe_pct: float = -5.0
    trailing_pct: float = -3.0
    trailing_activation_pct: float = 2.0
    take_profit_pct: float = 8.0

    _positions: dict = field(default_factory=dict, init=False, repr=False)

    def ouvrir_position(self, sym: str, prix_entree: float):
        self._positions[sym] = {
            "prix_entree": prix_entree,
            "prix_max": prix_entree,
        }

    def fermer_position(self, sym: str):
        self._positions.pop(sym, None)

    def mettre_a_jour_prix(self, sym: str, prix: float):
        if sym in self._positions:
            pos = self._positions[sym]
            if prix > pos["prix_max"]:
                pos["prix_max"] = prix

    def verifier_stop(self, sym: str, prix: float) -> tuple[bool, str]:
        if sym not in self._positions:
            return False, ""
        pos = self._positions[sym]
        entry = pos["prix_entree"]
        peak = pos["prix_max"]

        gain_vs_entree = (prix - entry) / entry * 100
        if gain_vs_entree >= self.take_profit_pct:
            return True, f"TakeProfit {sym}: +{gain_vs_entree:.1f}% >= +{self.take_profit_pct}%"

        if gain_vs_entree <= self.stop_fixe_pct:
            return True, f"StopLoss fixe {sym}: {gain_vs_entree:.1f}% <= {self.stop_fixe_pct}%"

        gain_peak = (peak - entry) / entry * 100
        if gain_peak >= self.trailing_activation_pct:
            perte_vs_peak = (prix - peak) / peak * 100
            if perte_vs_peak <= self.trailing_pct:
                return True, f"TrailingStop {sym}: -{abs(perte_vs_peak):.1f}% depuis pic ({peak:.2f})"

        return False, ""

    def etat(self) -> dict:
        return {
            sym: {
                "prix_entree": p["prix_entree"],
                "prix_max": round(p["prix_max"], 2),
                "gain_depuis_entree": round((p["prix_max"] - p["prix_entree"]) / p["prix_entree"] * 100, 1),
            }
            for sym, p in self._positions.items()
        }


@dataclass
class DrawdownCircuitBreaker:
    """Arret total si drawdown > seuil, reprise seulement quand equity remonte.
    Inspire de 2-Pair Correlation EA (expert-mt5-main)."""
    seuil_pct: float = -8.0
    recovery_pct: float = 95.0
    _tripped: bool = field(default=False, init=False, repr=False)
    _capital_at_trip: float = field(default=0.0, init=False, repr=False)

    def est_bloque(self, drawdown_pct: float = 0.0, capital: float = 0.0,
                   plus_haut: float = 0.0, **_) -> tuple[bool, str]:
        if not self._tripped and drawdown_pct <= self.seuil_pct:
            self._tripped = True
            self._capital_at_trip = capital
        if self._tripped:
            if plus_haut > 0:
                recovery = capital / plus_haut * 100
                if recovery >= self.recovery_pct:
                    self._tripped = False
                    return False, ""
            return True, (f"CircuitBreaker: DD {drawdown_pct:.1f}% a declenche l'arret. "
                          f"Reprise quand equity >= {self.recovery_pct}% du pic")
        return False, ""


@dataclass
class SessionFilter:
    """Filtre les heures de faible liquidite. Inspire de EA31337.
    Autorise le trading pendant London (07-16), NY (13-21), Tokyo (00-08) UTC."""
    sessions: dict = field(default_factory=lambda: {
        "london": (7, 16),
        "new_york": (13, 21),
        "tokyo": (0, 8),
    })
    actif: bool = True

    def est_bloque(self, **_) -> tuple[bool, str]:
        if not self.actif:
            return False, ""
        h = datetime.now(timezone.utc).hour
        for nom, (debut, fin) in self.sessions.items():
            if debut <= fin:
                if debut <= h < fin:
                    return False, ""
            else:
                if h >= debut or h < fin:
                    return False, ""
        return True, f"SessionFilter: heure {h}h UTC hors sessions (London/NY/Tokyo)"


@dataclass
class DailyLimits:
    """Limite gain/perte journaliere. Inspire de maxperdida-ganancia (mt5_mql5_eas)."""
    max_perte_jour: float = 3.0
    max_gain_jour: float = 15.0
    _trades_jour: list = field(default_factory=list, init=False, repr=False)
    _date_courante: str = field(default="", init=False, repr=False)

    def _reset_si_nouveau_jour(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._date_courante:
            self._date_courante = today
            self._trades_jour = []

    def enregistrer_trade(self, pnl: float):
        self._reset_si_nouveau_jour()
        self._trades_jour.append(pnl)

    def est_bloque(self, **_) -> tuple[bool, str]:
        self._reset_si_nouveau_jour()
        if not self._trades_jour:
            return False, ""
        pnl_jour = sum(self._trades_jour)
        if pnl_jour <= -self.max_perte_jour:
            return True, f"DailyLimit: perte jour {pnl_jour:.2f}$ >= -{self.max_perte_jour}$, arret"
        if pnl_jour >= self.max_gain_jour:
            return True, f"DailyLimit: gain jour +{pnl_jour:.2f}$ >= +{self.max_gain_jour}$, securise"
        return False, ""


@dataclass
class EquityTasks:
    """Taches conditionnelles sur l'equity. Inspire de EA31337.
    Ferme le pire perdant si drawdown > seuil, prend les gains si profit > seuil."""
    dd_coupe_pire: float = -6.0
    profit_ferme_tout: float = 20.0

    def evaluer(self, drawdown_pct: float, pnl_pct: float,
                positions: dict, signaux: dict) -> Optional[str]:
        if pnl_pct >= self.profit_ferme_tout:
            return "__FERMER_TOUT__"
        if drawdown_pct <= self.dd_coupe_pire and positions:
            pire_sym = None
            pire_pnl = 0.0
            for sym, pos in positions.items():
                if not pos.get("en_position"):
                    continue
                sig = signaux.get(sym, {})
                prix = sig.get("prix", pos.get("prix_entree", 0))
                if prix and pos.get("prix_entree"):
                    pnl = (prix - pos["prix_entree"]) / pos["prix_entree"] * 100
                    if pnl < pire_pnl:
                        pire_pnl = pnl
                        pire_sym = sym
            if pire_sym:
                return pire_sym
        return None


def calculer_atr(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """Calcule l'ATR a partir de listes OHLC. Retourne 0 si pas assez de donnees."""
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return 0.0
    return sum(trs[-period:]) / period


def calculer_sl_structurel(lows: list, n: int = 15) -> float:
    """SL base sur le plus bas des N dernieres bougies (swing low).
    Inspire de MACD_RSI_STOCH V5 (mt5_mql5_eas)."""
    if not lows or len(lows) < 2:
        return 0.0
    recent = lows[-n:] if len(lows) >= n else lows
    return min(recent)


def calculer_tp_structurel(highs: list, n: int = 15) -> float:
    """TP base sur le plus haut des N dernieres bougies (swing high)."""
    if not highs or len(highs) < 2:
        return 0.0
    recent = highs[-n:] if len(highs) >= n else highs
    return max(recent)


def position_sizing_atr(capital: float, risk_pct: float, prix_entree: float,
                        atr: float, multiplier: float = 1.5) -> float:
    """Calcule le montant a investir base sur l'ATR.
    Formule: montant = (capital * risk%) / (ATR / prix * multiplier)
    Inspire de AdaptiveTrader Pro (expert-mt5-main) + metatrader5-master."""
    if atr <= 0 or prix_entree <= 0:
        return 0.0
    risk_amount = capital * risk_pct / 100.0
    atr_pct = (atr * multiplier) / prix_entree
    if atr_pct <= 0:
        return 0.0
    montant = risk_amount / atr_pct
    return min(montant, capital * 0.30)


def trailing_paliers(prix_entree: float, sl_actuel: float, prix_actuel: float,
                     palier_ratio: float = 1.75) -> float:
    """Trailing SL par paliers. Le SL avance a chaque palier de N*risk.
    Inspire de MACD_RSI_STOCH V5 (mt5_mql5_eas).
    Retourne le nouveau SL (ou l'ancien si pas de changement)."""
    if prix_entree <= 0 or sl_actuel <= 0:
        return sl_actuel
    risk_initial = prix_entree - sl_actuel
    if risk_initial <= 0:
        return sl_actuel
    gain = prix_actuel - prix_entree
    if gain <= 0:
        return sl_actuel
    nb_paliers = int(gain / (risk_initial * palier_ratio))
    if nb_paliers >= 1:
        nouveau_sl = prix_entree + risk_initial * 0.1 * nb_paliers
        return max(sl_actuel, round(nouveau_sl, 4))
    return sl_actuel


@dataclass
class SpreadFilter:
    """Bloque les trades quand le spread est trop large.
    Inspire de Gold_PRO_V2 + tous les Nkanven EAs."""
    max_spread_pct: float = 0.15

    def est_bloque(self, spread: float = 0.0, prix: float = 0.0, **_) -> tuple[bool, str]:
        if prix <= 0 or spread <= 0:
            return False, ""
        spread_pct = spread / prix * 100
        if spread_pct > self.max_spread_pct:
            return True, f"SpreadFilter: spread {spread_pct:.3f}% > {self.max_spread_pct}%"
        return False, ""


@dataclass
class MaxHoldTime:
    """Ferme les positions ouvertes trop longtemps.
    Inspire de ORB.mq5 (orb-mt5-main)."""
    max_minutes: int = 1440

    _positions_time: dict = field(default_factory=dict, init=False, repr=False)

    def ouvrir(self, sym: str):
        self._positions_time[sym] = datetime.now(timezone.utc)

    def fermer(self, sym: str):
        self._positions_time.pop(sym, None)

    def verifier(self, sym: str) -> tuple[bool, str]:
        if sym not in self._positions_time:
            return False, ""
        elapsed = (datetime.now(timezone.utc) - self._positions_time[sym]).total_seconds() / 60
        if elapsed >= self.max_minutes:
            return True, f"MaxHoldTime: {sym} ouvert depuis {int(elapsed)}min >= {self.max_minutes}min"
        return False, ""


@dataclass
class D1AlignmentFilter:
    """Skip les entries qui vont contre la bougie D1 d'hier.
    Reduit le DD de 10.3% a 4.2% dans les backtests ORB.
    Inspire de ORB.mq5 (orb-mt5-main)."""
    min_body_pct: float = 0.3

    def est_aligne(self, d1_open: float, d1_close: float, is_buy: bool, **_) -> tuple[bool, str]:
        if d1_open <= 0 or d1_close <= 0:
            return True, ""
        body_pct = abs(d1_close - d1_open) / d1_open * 100
        if body_pct < self.min_body_pct:
            return True, ""
        d1_bullish = d1_close > d1_open
        if is_buy and not d1_bullish:
            return False, f"D1Alignment: achat bloque, D1 baissier ({body_pct:.2f}%)"
        if not is_buy and d1_bullish:
            return False, f"D1Alignment: vente bloquee, D1 haussier ({body_pct:.2f}%)"
        return True, ""


@dataclass
class PartialCloseLadder:
    """Ferme progressivement 40/30/30 aux paliers TP1/TP2/TP3.
    Inspire de Gold_PRO_V2 (50% at BE) + Gold_ATR_Breakout_v3 (40/30/30).
    Retourne le % du volume a fermer (0 si rien)."""
    tp1_ratio: float = 1.0
    tp2_ratio: float = 2.0
    pct_tp1: float = 0.40
    pct_tp2: float = 0.30

    def evaluer(self, profit_r: float, deja_ferme: set) -> tuple[float, str]:
        if "tp1" not in deja_ferme and profit_r >= self.tp1_ratio:
            return self.pct_tp1, f"PartialClose: TP1 atteint ({profit_r:.1f}R), fermer {self.pct_tp1*100:.0f}%"
        if "tp2" not in deja_ferme and profit_r >= self.tp2_ratio:
            return self.pct_tp2, f"PartialClose: TP2 atteint ({profit_r:.1f}R), fermer {self.pct_tp2*100:.0f}%"
        return 0.0, ""


@dataclass
class WeightedHistoryScore:
    """Lot sizing pondere par l'historique recent des 3 derniers trades.
    Plus nuance que le simple comptage de pertes consecutives.
    Inspire de MidNightAngel.mq5 (MT4-MT5 Collections)."""
    _history: list = field(default_factory=list, init=False, repr=False)

    def enregistrer(self, pnl: float):
        self._history.append(pnl)
        if len(self._history) > 3:
            self._history = self._history[-3:]

    def multiplicateur_lot(self) -> tuple[float, str]:
        if len(self._history) < 3:
            return 1.0, "WeightedHistory: pas assez d'historique"
        score = 0
        for i, p in enumerate(self._history):
            weight = i + 1
            score += weight if p > 0 else -weight
        if score <= -4:
            return 0.5, f"WeightedHistory: score {score}, lot x0.5 (protection)"
        if score <= -2:
            return 0.75, f"WeightedHistory: score {score}, lot x0.75"
        if score >= 4:
            return 1.25, f"WeightedHistory: score {score}, lot x1.25 (confiance)"
        return 1.0, f"WeightedHistory: score {score}, lot x1.0"


def bollinger_squeeze_gate(closes: list, period: int = 20, bb_mult: float = 2.0,
                           kc_mult: float = 1.5, atr_period: int = 14) -> tuple[bool, str]:
    """Bloque les entries quand les Bollinger Bands sont compressees dans le Keltner Channel.
    Inspire de RB_XAUUSD_MT5.mq5 (xauusd-mt5-expert-advisor)."""
    if len(closes) < max(period, atr_period) + 1:
        return True, "SqueezeGate: pas assez de donnees"
    sma = sum(closes[-period:]) / period
    variance = sum((c - sma) ** 2 for c in closes[-period:]) / period
    std = variance ** 0.5
    bb_upper = sma + bb_mult * std
    bb_lower = sma - bb_mult * std

    highs = closes
    lows = closes
    trs = []
    for i in range(-atr_period, 0):
        tr = abs(closes[i] - closes[i - 1])
        trs.append(tr)
    atr = sum(trs) / len(trs) if trs else 0
    kc_upper = sma + kc_mult * atr
    kc_lower = sma - kc_mult * atr

    squeeze = bb_upper < kc_upper and bb_lower > kc_lower
    if squeeze:
        return True, "SqueezeGate: BB dans KC, volatilite morte"
    return False, ""


def mass_index_reversal(highs: list, lows: list, ema_period: int = 9,
                        sum_period: int = 25) -> tuple[bool, str]:
    """Detecte un bulge Mass Index (>27 puis <26.5) signalant un retournement imminent.
    Inspire de pyrobot/indicators.py (python-trading-robot)."""
    if len(highs) < sum_period + ema_period * 2 + 2:
        return False, ""
    diffs = [h - l for h, l in zip(highs, lows)]

    alpha = 2.0 / (ema_period + 1)
    ema1 = [0.0] * len(diffs)
    ema1[0] = diffs[0]
    for i in range(1, len(diffs)):
        ema1[i] = alpha * diffs[i] + (1 - alpha) * ema1[i - 1]

    ema2 = [0.0] * len(ema1)
    ema2[0] = ema1[0]
    for i in range(1, len(ema1)):
        ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i - 1]

    ratios = []
    for i in range(len(ema2)):
        ratios.append(ema1[i] / ema2[i] if ema2[i] != 0 else 1.0)

    if len(ratios) < sum_period + 1:
        return False, ""
    mi_prev = sum(ratios[-sum_period - 1:-1])
    mi_now = sum(ratios[-sum_period:])
    if mi_prev > 27.0 and mi_now < 26.5:
        return True, f"MassIndex: bulge {mi_prev:.1f} -> {mi_now:.1f}, retournement probable"
    return False, ""


def force_index(closes: list, volumes: list, period: int = 13) -> float:
    """Force Index : prix_change * volume_change. Positif = pression acheteuse.
    Inspire de pyrobot/indicators.py (python-trading-robot)."""
    if len(closes) < period + 1 or len(volumes) < period + 1:
        return 0.0
    fi_raw = []
    for i in range(1, len(closes)):
        fi_raw.append((closes[i] - closes[i - 1]) * volumes[i])
    alpha = 2.0 / (period + 1)
    ema = fi_raw[0]
    for v in fi_raw[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def chaikin_money_flow(highs: list, lows: list, closes: list,
                       volumes: list, period: int = 20) -> float:
    """Chaikin Money Flow : accumulation/distribution.
    Positif = achat institutionnel, negatif = distribution.
    Inspire de pyrobot/indicators.py (python-trading-robot)."""
    if len(closes) < period:
        return 0.0
    mfv_sum = 0.0
    vol_sum = 0.0
    for i in range(-period, 0):
        hl = highs[i] - lows[i]
        if hl > 0:
            mfm = (2 * closes[i] - highs[i] - lows[i]) / hl
        else:
            mfm = 0.0
        mfv_sum += mfm * volumes[i]
        vol_sum += volumes[i]
    return mfv_sum / vol_sum if vol_sum > 0 else 0.0


def pin_bar_score(open_p: float, high: float, low: float, close: float,
                  min_ratio: float = 2.0) -> tuple[int, float]:
    """Detecte un pin-bar. Retourne (direction, score).
    direction: 1=bullish, -1=bearish, 0=pas de pin.
    Inspire de Gold_PRO_V2.mq5."""
    body = max(abs(open_p - close), high * 1e-10)
    upper_shadow = high - max(open_p, close)
    lower_shadow = min(open_p, close) - low
    if lower_shadow > body * min_ratio and lower_shadow > upper_shadow:
        color_ok = close > open_p
        return (1 if color_ok else 0), lower_shadow / body
    if upper_shadow > body * min_ratio and upper_shadow > lower_shadow:
        color_ok = close < open_p
        return (-1 if color_ok else 0), upper_shadow / body
    return 0, 0.0


def stochastic_signal(k_prev: float, d_prev: float, k_now: float, d_now: float,
                      buy_level: float = 20.0, sell_level: float = 80.0) -> int:
    """Signal stochastique en zone extreme. 1=BUY, -1=SELL, 0=neutre.
    Inspire de Stochastic-Oscillator-EA.mq5 (experts-master)."""
    if d_prev <= buy_level and k_prev <= d_prev and k_now <= buy_level and d_now <= k_now:
        return 1
    if d_prev >= sell_level and k_prev >= d_prev and k_now >= sell_level and d_now >= k_now:
        return -1
    return 0


def calculer_stochastic(highs: list, lows: list, closes: list,
                        k_period: int = 14, d_period: int = 3) -> tuple[float, float]:
    """Calcule %K et %D du Stochastic Oscillator."""
    if len(closes) < k_period + d_period:
        return 50.0, 50.0
    k_values = []
    for i in range(len(closes) - k_period - d_period + 1, len(closes)):
        start = max(0, i - k_period + 1)
        hh = max(highs[start:i + 1])
        ll = min(lows[start:i + 1])
        if hh - ll > 0:
            k_values.append((closes[i] - ll) / (hh - ll) * 100)
        else:
            k_values.append(50.0)
    if len(k_values) < d_period:
        return 50.0, 50.0
    k_now = k_values[-1]
    d_now = sum(k_values[-d_period:]) / d_period
    return k_now, d_now


def orb_half_range_filter(range_high: float, range_low: float,
                          range_close: float, is_buy: bool) -> tuple[bool, str]:
    """N'autorise le breakout que dans la direction ou le range a cloture.
    Top half = longs, bottom half = shorts. +0.615R/trade correct vs +0.049R wrong.
    Inspire de ORB.mq5 (orb-mt5-main)."""
    if range_high <= range_low:
        return True, ""
    close_pos = (range_close - range_low) / (range_high - range_low)
    if is_buy and close_pos < 0.50:
        return False, f"ORB HalfRange: achat bloque, close en bas du range ({close_pos:.0%})"
    if not is_buy and close_pos >= 0.50:
        return False, f"ORB HalfRange: vente bloquee, close en haut du range ({close_pos:.0%})"
    return True, ""


@dataclass
class BreakoutRetestGuard:
    """Si le prix s'etend trop loin d'un breakout (>max_ext ATR), attend un retest.
    Inspire de RB_XAUUSD_MT5.mq5 (xauusd-mt5-expert-advisor)."""
    max_extension_atr: float = 0.85
    retest_window_bars: int = 8
    retest_zone_atr: float = 0.45

    _pending: dict = field(default_factory=dict, init=False, repr=False)

    def evaluer_breakout(self, sym: str, breakout_level: float, prix: float,
                         atr: float) -> str:
        if atr <= 0:
            return "ENTER"
        extension = abs(prix - breakout_level) / atr
        if extension > self.max_extension_atr:
            self._pending[sym] = {
                "level": breakout_level, "atr": atr,
                "bars_restants": self.retest_window_bars,
            }
            return "WAIT_RETEST"
        return "ENTER"

    def verifier_retest(self, sym: str, close: float, low: float,
                        high: float) -> str:
        if sym not in self._pending:
            return "NO_PENDING"
        p = self._pending[sym]
        p["bars_restants"] -= 1
        if p["bars_restants"] <= 0:
            del self._pending[sym]
            return "EXPIRED"
        zone = p["atr"] * self.retest_zone_atr
        touched = low <= p["level"] + zone and high >= p["level"] - zone
        resumed = close > p["level"]
        if touched and resumed:
            del self._pending[sym]
            return "ENTER"
        return "WAITING"


def fibonacci_trailing_bb(entry: float, ema: float, bb_upper: float,
                          bb_lower: float, close: float, current_sl: float,
                          is_buy: bool) -> tuple[float, float]:
    """Trail le SL a travers les niveaux Fib (0.236, 0.382, 0.5, 0.618) de EMA-to-BB.
    Etend dynamiquement le TP quand la bande opposee s'ecarte.
    Inspire de DayTradingEA_v2.mq5 (mql5_ea-main)."""
    fibs = [0.236, 0.382, 0.5, 0.618]
    new_sl = current_sl
    for f in fibs:
        if is_buy:
            target = ema + f * (bb_upper - ema)
            if close > target and target > new_sl:
                new_sl = target
        else:
            target = ema - f * (ema - bb_lower)
            if close < target and (new_sl == 0 or target < new_sl):
                new_sl = target
    new_tp = bb_upper if is_buy else bb_lower
    return round(new_sl, 6), round(new_tp, 6)


def adaptive_time_trailing(candles_open: int, entry: float, current_price: float,
                           last_candle_range: float, trail_distance: float,
                           current_sl: float) -> float:
    """Trailing en 2 phases selon l'age de la position.
    Phase minor (1-3 bougies): SL a entry + trail_distance.
    Phase major (4+): SL = max(prix - candle_range, prix - trail_distance).
    Inspire de fxchartai_openea.mq5 (openea-main)."""
    if current_price <= entry:
        return current_sl
    if 1 <= candles_open <= 3:
        minor_sl = entry + trail_distance * 0.3
        return max(current_sl, round(minor_sl, 6))
    if candles_open >= 4:
        candle_sl = current_price - last_candle_range
        pips_sl = current_price - trail_distance
        major_sl = max(candle_sl, pips_sl)
        return max(current_sl, round(major_sl, 6))
    return current_sl


def ichimoku_dynamic_sl(span_a: float, span_b: float, kijun: float,
                        is_buy: bool) -> float:
    """Place le SL au meilleur support structurel dans le nuage Ichimoku.
    Inspire de E_EntriesManagement.mqh (MT4-MT5-Trading-EA-Collections)."""
    if is_buy:
        if span_b < span_a:
            return min(span_b, kijun)
        return span_a
    else:
        if span_b > span_a:
            return max(span_b, kijun)
        return span_a


def calculer_ichimoku(highs: list, lows: list, closes: list,
                      tenkan_p: int = 9, kijun_p: int = 26,
                      senkou_b_p: int = 52) -> dict:
    """Calcule les 5 lignes Ichimoku a partir de listes OHLC."""
    def midpoint(data, period):
        if len(data) < period:
            return 0.0
        segment = data[-period:]
        return (max(segment) + min(segment)) / 2.0

    tenkan = midpoint(highs, tenkan_p) if len(highs) >= tenkan_p else 0.0
    kijun = midpoint(highs, kijun_p) if len(highs) >= kijun_p else 0.0

    tenkan_low = midpoint(lows, tenkan_p) if len(lows) >= tenkan_p else 0.0
    kijun_low = midpoint(lows, kijun_p) if len(lows) >= kijun_p else 0.0
    tenkan = (tenkan + tenkan_low) / 2
    kijun = (kijun + kijun_low) / 2

    span_a = (tenkan + kijun) / 2
    span_b_h = midpoint(highs, senkou_b_p) if len(highs) >= senkou_b_p else 0.0
    span_b_l = midpoint(lows, senkou_b_p) if len(lows) >= senkou_b_p else 0.0
    span_b = (span_b_h + span_b_l) / 2

    chikou = closes[-1] if closes else 0.0

    return {
        "tenkan": round(tenkan, 6),
        "kijun": round(kijun, 6),
        "span_a": round(span_a, 6),
        "span_b": round(span_b, 6),
        "chikou": round(chikou, 6),
    }


def walk_forward_backtest(returns: list, positions: list, cost_per_trade: float = 0.001,
                          train_ratio: float = 0.7) -> dict:
    """Walk-forward validation avec modele de couts (spread + slippage).
    Inspire de backtest.py (Robot_Trading_Brief).
    Retourne CAGR, Sharpe, max DD sur la partie out-of-sample."""
    import math
    n = len(returns)
    if n < 20 or len(positions) != n:
        return {"sharpe": 0.0, "max_dd": 0.0, "cagr": 0.0, "oos_trades": 0}

    split = int(n * train_ratio)
    oos_ret = returns[split:]
    oos_pos = positions[split:]

    strat_returns = []
    trades = 0
    for i in range(len(oos_ret)):
        pos_change = abs(oos_pos[i] - (oos_pos[i - 1] if i > 0 else 0))
        cost = pos_change * cost_per_trade
        r = oos_pos[i] * oos_ret[i] - cost
        strat_returns.append(r)
        if pos_change > 0:
            trades += 1

    if not strat_returns:
        return {"sharpe": 0.0, "max_dd": 0.0, "cagr": 0.0, "oos_trades": 0}

    mean_r = sum(strat_returns) / len(strat_returns)
    var_r = sum((r - mean_r) ** 2 for r in strat_returns) / len(strat_returns)
    std_r = var_r ** 0.5 if var_r > 0 else 1e-10
    sharpe = mean_r / std_r * (252 ** 0.5)

    equity = [1.0]
    for r in strat_returns:
        equity.append(equity[-1] * (1 + r))
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (e - peak) / peak
        if dd < max_dd:
            max_dd = dd

    total_return = equity[-1] / equity[0]
    years = len(oos_ret) / 252
    cagr = (total_return ** (1 / years) - 1) * 100 if years > 0 and total_return > 0 else 0.0

    return {
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd * 100, 2),
        "cagr": round(cagr, 2),
        "oos_trades": trades,
    }


def asian_range_breakout(highs_session: list, lows_session: list,
                         current_price: float,
                         tp_multiplier: float = 1.5) -> dict:
    """Calcule le range de la session asiatique (22h-7h UTC) et genere les signaux breakout.
    Inspire de DL_TradingBoundaries.mqh (MT4-MT5-Trading-EA-Collections)."""
    if not highs_session or not lows_session:
        return {"signal": 0, "upper": 0, "lower": 0, "tp": 0, "sl": 0}
    upper = max(highs_session)
    lower = min(lows_session)
    scope = upper - lower
    if scope <= 0:
        return {"signal": 0, "upper": upper, "lower": lower, "tp": 0, "sl": 0}
    if current_price > upper:
        return {
            "signal": 1,
            "upper": round(upper, 6),
            "lower": round(lower, 6),
            "tp": round(current_price + scope * tp_multiplier, 6),
            "sl": round(lower, 6),
        }
    if current_price < lower:
        return {
            "signal": -1,
            "upper": round(upper, 6),
            "lower": round(lower, 6),
            "tp": round(current_price - scope * tp_multiplier, 6),
            "sl": round(upper, 6),
        }
    return {"signal": 0, "upper": round(upper, 6), "lower": round(lower, 6), "tp": 0, "sl": 0}


class GestionnaireProtections:
    """Centralise toutes les protections."""

    def __init__(
        self,
        max_drawdown_pct: float = -5.0,
        pertes_max_consecutives: int = 3,
        cooldown_seconds: int = 300,
        fee_pct: float = 0.1,
        stop_fixe_pct: float = -5.0,
        trailing_pct: float = -3.0,
        trailing_activation_pct: float = 2.0,
    ):
        self.max_dd = MaxDrawdownProtection(seuil_pct=max_drawdown_pct)
        self.stoploss = StoplossGuard(
            pertes_max=pertes_max_consecutives,
            cooldown_seconds=cooldown_seconds * 2,
        )
        self.cooldown = CooldownProtection(cooldown_seconds=cooldown_seconds)
        self.fee_filter = FeeFilter(fee_pct=fee_pct)
        self.stop_technique = TechnicalStopLoss(
            stop_fixe_pct=stop_fixe_pct,
            trailing_pct=trailing_pct,
            trailing_activation_pct=trailing_activation_pct,
            take_profit_pct=8.0,
        )
        self.circuit_breaker = DrawdownCircuitBreaker(seuil_pct=-8.0, recovery_pct=95.0)
        self.session_filter = SessionFilter(actif=False)
        self.daily_limits = DailyLimits(max_perte_jour=3.0, max_gain_jour=15.0)
        self.equity_tasks = EquityTasks(dd_coupe_pire=-6.0, profit_ferme_tout=20.0)
        self.spread_filter = SpreadFilter(max_spread_pct=0.15)
        self.max_hold = MaxHoldTime(max_minutes=1440)
        self.d1_alignment = D1AlignmentFilter(min_body_pct=0.3)
        self.partial_ladder = PartialCloseLadder()
        self.weighted_history = WeightedHistoryScore()
        self.breakout_retest = BreakoutRetestGuard()

    def peut_trader(self, sym: str, drawdown_pct: float,
                    capital: float = 0.0, plus_haut: float = 0.0,
                    spread: float = 0.0, prix: float = 0.0) -> tuple[bool, str]:
        for protection in [self.max_dd, self.stoploss, self.cooldown,
                           self.session_filter, self.daily_limits]:
            bloque, raison = protection.est_bloque(sym=sym, drawdown_pct=drawdown_pct)
            if bloque:
                return False, raison
        bloque, raison = self.circuit_breaker.est_bloque(
            drawdown_pct=drawdown_pct, capital=capital, plus_haut=plus_haut)
        if bloque:
            return False, raison
        bloque, raison = self.spread_filter.est_bloque(spread=spread, prix=prix)
        if bloque:
            return False, raison
        return True, ""

    def apres_trade(self, sym: str, pnl: float):
        self.stoploss.enregistrer_trade(pnl)
        self.cooldown.enregistrer_trade(sym)
        self.daily_limits.enregistrer_trade(pnl)
        self.weighted_history.enregistrer(pnl)
