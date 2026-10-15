"""
Risk Engine — Gestion du risque et garde-fous
Indépendant du modèle : ces règles ne peuvent pas être contournées.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rex_alpha.config import RexConfig


@dataclass
class CheckResult:
    ok: bool
    regle: str
    valeur: float
    seuil: float
    message: str = ""


@dataclass
class PortfolioState:
    capital: float
    cash: float
    positions: list[dict] = field(default_factory=list)
    pnl_jour: float = 0
    plus_haut: float = 0
    nb_erreurs_connexion: int = 0


class RiskManager:
    def __init__(self, config: RexConfig | None = None):
        self.config = config or RexConfig()
        self.arrete = False

    def verifier_tout(self, state: PortfolioState) -> list[CheckResult]:
        return [
            self.check_drawdown(state),
            self.check_perte_jour(state),
            self.check_levier(state),
            self.check_exposition(state),
            self.check_nb_positions(state),
            self.check_connexion(state),
        ]

    def peut_trader(self, state: PortfolioState) -> tuple[bool, list[str]]:
        if self.arrete:
            return False, ["ARRET_TOTAL: drawdown max atteint"]
        checks = self.verifier_tout(state)
        violations = [c for c in checks if not c.ok]
        return len(violations) == 0, [c.message for c in violations]

    def valider_ordre(self, state: PortfolioState, symbole: str,
                      montant: float, prix: float) -> tuple[bool, str]:
        ok, raisons = self.peut_trader(state)
        if not ok:
            return False, " | ".join(raisons)

        # Risque par trade
        risque_pct = montant / state.capital * 100
        if risque_pct > self.config.max_risque_par_trade_pct:
            return False, f"Risque par trade {risque_pct:.1f}% > max {self.config.max_risque_par_trade_pct}%"

        # Taille position
        poids = montant / state.capital
        if poids > self.config.max_poids_par_titre:
            return False, f"Poids {poids:.1%} > max {self.config.max_poids_par_titre:.0%}"

        # Cash suffisant
        if montant > state.cash:
            return False, f"Cash insuffisant: {state.cash:.2f}$ < {montant:.2f}$"

        return True, "OK"

    def calculer_taille_position(self, state: PortfolioState, prix: float,
                                 score_alpha: float) -> float:
        max_montant = state.capital * self.config.max_poids_par_titre
        max_cash = state.cash * 0.95
        base = min(max_montant, max_cash)
        facteur = min(score_alpha / 100, 1.0)
        montant = base * facteur
        return max(0, round(montant, 2))

    # ─── Checks individuels ──────────────────────────────

    def check_drawdown(self, state: PortfolioState) -> CheckResult:
        if state.plus_haut <= 0:
            return CheckResult(True, "drawdown", 0, self.config.max_drawdown_pct)
        dd = (state.capital - state.plus_haut) / state.plus_haut * 100
        ok = abs(dd) < self.config.max_drawdown_pct
        if not ok:
            self.arrete = True
        return CheckResult(
            ok=ok, regle="drawdown", valeur=round(dd, 2),
            seuil=self.config.max_drawdown_pct,
            message=f"Drawdown {dd:.1f}% {'OK' if ok else '> LIMITE ' + str(self.config.max_drawdown_pct) + '%'}",
        )

    def check_perte_jour(self, state: PortfolioState) -> CheckResult:
        perte_pct = abs(min(0, state.pnl_jour)) / state.capital * 100 if state.capital > 0 else 0
        ok = perte_pct < self.config.max_perte_jour_pct
        return CheckResult(
            ok=ok, regle="perte_jour", valeur=round(perte_pct, 2),
            seuil=self.config.max_perte_jour_pct,
            message=f"Perte jour {perte_pct:.1f}% {'OK' if ok else '> LIMITE'}",
        )

    def check_levier(self, state: PortfolioState) -> CheckResult:
        val_pos = sum(p.get("valeur_actuelle", 0) for p in state.positions)
        levier = val_pos / state.capital if state.capital > 0 else 0
        ok = levier <= self.config.levier_max
        return CheckResult(
            ok=ok, regle="levier", valeur=round(levier, 2),
            seuil=self.config.levier_max,
            message=f"Levier {levier:.2f}x {'OK' if ok else '> LIMITE'}",
        )

    def check_exposition(self, state: PortfolioState) -> CheckResult:
        val_pos = sum(p.get("valeur_actuelle", 0) for p in state.positions)
        expo = val_pos / state.capital if state.capital > 0 else 0
        ok = expo <= self.config.max_exposition_totale
        return CheckResult(
            ok=ok, regle="exposition", valeur=round(expo, 2),
            seuil=self.config.max_exposition_totale,
            message=f"Exposition {expo:.0%} {'OK' if ok else '> LIMITE'}",
        )

    def check_nb_positions(self, state: PortfolioState) -> CheckResult:
        nb = len([p for p in state.positions if p.get("quantite", 0) > 0])
        ok = nb <= self.config.max_positions
        return CheckResult(
            ok=ok, regle="nb_positions", valeur=nb,
            seuil=self.config.max_positions,
            message=f"Positions {nb} {'OK' if ok else '> LIMITE'}",
        )

    def check_connexion(self, state: PortfolioState) -> CheckResult:
        ok = state.nb_erreurs_connexion <= 3
        return CheckResult(
            ok=ok, regle="connexion", valeur=state.nb_erreurs_connexion,
            seuil=3,
            message=f"Erreurs connexion {state.nb_erreurs_connexion} {'OK' if ok else '> LIMITE'}",
        )
