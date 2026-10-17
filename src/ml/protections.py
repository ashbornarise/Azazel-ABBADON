"""
Protections de trading — Rex Clemens Capital
=============================================
Inspirees de freqtrade + EA31337 + AdaptiveTrader Pro + MACD_RSI_STOCH V5.
Ces protections ARRETENT le trading quand les conditions de survie
sont menacees. Elles ne peuvent pas etre desactivees par la strategie.

V2 : ATR sizing, drawdown circuit breaker avec recovery, trailing par paliers,
     SL structurel (swing low/high), filtre sessions, limite gain/perte journaliere,
     equity conditional tasks.
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

    def peut_trader(self, sym: str, drawdown_pct: float,
                    capital: float = 0.0, plus_haut: float = 0.0) -> tuple[bool, str]:
        for protection in [self.max_dd, self.stoploss, self.cooldown,
                           self.session_filter, self.daily_limits]:
            bloque, raison = protection.est_bloque(sym=sym, drawdown_pct=drawdown_pct)
            if bloque:
                return False, raison
        bloque, raison = self.circuit_breaker.est_bloque(
            drawdown_pct=drawdown_pct, capital=capital, plus_haut=plus_haut)
        if bloque:
            return False, raison
        return True, ""

    def apres_trade(self, sym: str, pnl: float):
        self.stoploss.enregistrer_trade(pnl)
        self.cooldown.enregistrer_trade(sym)
        self.daily_limits.enregistrer_trade(pnl)
