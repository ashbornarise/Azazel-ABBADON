"""
Protections de trading — Rex Clemens Capital
=============================================
Inspirees de freqtrade (MaxDrawdown, StoplossGuard, CooldownPeriod).
Ces protections ARRETENT le trading quand les conditions de survie
sont menacees. Elles ne peuvent pas etre desactivees par la strategie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


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

    def peut_trader(self, sym: str, drawdown_pct: float) -> tuple[bool, str]:
        for protection in [self.max_dd, self.stoploss, self.cooldown]:
            bloque, raison = protection.est_bloque(sym=sym, drawdown_pct=drawdown_pct)
            if bloque:
                return False, raison
        return True, ""

    def apres_trade(self, sym: str, pnl: float):
        self.stoploss.enregistrer_trade(pnl)
        self.cooldown.enregistrer_trade(sym)
