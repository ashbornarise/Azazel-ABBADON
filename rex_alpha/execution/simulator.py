"""
Simulateur d'exécution — Rex Alpha Engine
Gère le portefeuille virtuel, les ordres simulés, les frais et le journal.
"""
from __future__ import annotations

from datetime import datetime, timezone

from rex_alpha.config import RexConfig
from rex_alpha.storage import RexDB
from rex_alpha.risk_engine import RiskManager
from rex_alpha.risk_engine.risk_manager import PortfolioState


class ExecutionSimulator:
    def __init__(self, db: RexDB, risk: RiskManager, config: RexConfig | None = None):
        self.db = db
        self.risk = risk
        self.config = config or RexConfig()
        self.capital = self.config.capital_initial
        self.cash = self.config.capital_initial
        self.positions: dict[str, dict] = {}
        self.plus_haut = self.capital
        self.pnl_jour = 0.0
        self.version = "0.1"

    def get_state(self) -> PortfolioState:
        pos_list = [
            {**v, "symbole": k} for k, v in self.positions.items()
            if v.get("quantite", 0) > 0
        ]
        return PortfolioState(
            capital=self.capital,
            cash=self.cash,
            positions=pos_list,
            pnl_jour=self.pnl_jour,
            plus_haut=self.plus_haut,
        )

    def acheter(self, symbole: str, prix: float, montant: float,
                raison: str = "") -> dict:
        state = self.get_state()
        ok, msg = self.risk.valider_ordre(state, symbole, montant, prix)
        if not ok:
            self.db.log("ORDRE_REFUSE", {
                "symbole": symbole, "sens": "ACHAT",
                "montant": montant, "raison": msg,
            }, self.version)
            return {"ok": False, "raison": msg}

        frais = montant * self.config.frais_par_trade_pct / 100
        slippage = prix * self.config.slippage_pct / 100
        prix_exec = prix + slippage
        quantite = (montant - frais) / prix_exec

        self.cash -= montant
        if symbole in self.positions and self.positions[symbole].get("quantite", 0) > 0:
            pos = self.positions[symbole]
            total_qte = pos["quantite"] + quantite
            total_cout = pos["quantite"] * pos["prix_moyen"] + quantite * prix_exec
            pos["prix_moyen"] = total_cout / total_qte
            pos["quantite"] = total_qte
        else:
            self.positions[symbole] = {
                "quantite": quantite,
                "prix_moyen": prix_exec,
                "date_entree": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "valeur_actuelle": quantite * prix_exec,
                "pnl_latent": 0,
            }

        self.db.insert_ordre(
            symbole, "ACHAT", quantite, prix, prix_exec,
            frais, slippage * quantite, raison, self.version,
        )
        self.db.set_position(
            symbole, self.positions[symbole]["quantite"],
            self.positions[symbole]["prix_moyen"],
            self.positions[symbole]["date_entree"],
        )
        self.db.log("ACHAT", {
            "symbole": symbole, "quantite": round(quantite, 4),
            "prix": round(prix_exec, 2), "montant": round(montant, 2),
            "frais": round(frais, 2), "raison": raison,
        }, self.version)

        self._update_capital()
        return {
            "ok": True, "symbole": symbole,
            "quantite": round(quantite, 4),
            "prix_exec": round(prix_exec, 2),
            "frais": round(frais, 2),
            "montant_net": round(montant - frais, 2),
        }

    def vendre(self, symbole: str, prix: float, raison: str = "") -> dict:
        pos = self.positions.get(symbole)
        if not pos or pos.get("quantite", 0) <= 0:
            return {"ok": False, "raison": "pas de position"}

        quantite = pos["quantite"]
        slippage = prix * self.config.slippage_pct / 100
        prix_exec = prix - slippage
        montant_brut = quantite * prix_exec
        frais = montant_brut * self.config.frais_par_trade_pct / 100
        montant_net = montant_brut - frais

        pnl = montant_net - (quantite * pos["prix_moyen"])

        self.cash += montant_net
        self.positions[symbole] = {"quantite": 0, "prix_moyen": 0, "date_entree": "", "valeur_actuelle": 0, "pnl_latent": 0}

        self.db.insert_ordre(
            symbole, "VENTE", quantite, prix, prix_exec,
            frais, slippage * quantite, raison, self.version,
        )
        self.db.set_position(symbole, 0, 0, "")
        self.db.log("VENTE", {
            "symbole": symbole, "quantite": round(quantite, 4),
            "prix": round(prix_exec, 2), "montant_net": round(montant_net, 2),
            "pnl": round(pnl, 2), "frais": round(frais, 2), "raison": raison,
        }, self.version)

        self._update_capital()
        return {
            "ok": True, "symbole": symbole,
            "quantite": round(quantite, 4),
            "prix_exec": round(prix_exec, 2),
            "pnl": round(pnl, 2),
            "montant_net": round(montant_net, 2),
        }

    def update_prix(self, prix_actuels: dict[str, float]):
        val_positions = 0
        for sym, pos in self.positions.items():
            if pos.get("quantite", 0) > 0 and sym in prix_actuels:
                prix = prix_actuels[sym]
                pos["valeur_actuelle"] = pos["quantite"] * prix
                pos["pnl_latent"] = pos["valeur_actuelle"] - pos["quantite"] * pos["prix_moyen"]
                val_positions += pos["valeur_actuelle"]

        self.capital = self.cash + val_positions
        self.plus_haut = max(self.plus_haut, self.capital)

    def _update_capital(self):
        val_pos = sum(
            p.get("valeur_actuelle", p.get("quantite", 0) * p.get("prix_moyen", 0))
            for p in self.positions.values()
            if p.get("quantite", 0) > 0
        )
        self.capital = self.cash + val_pos
        self.plus_haut = max(self.plus_haut, self.capital)

    def snapshot(self, benchmark_val: float = 0):
        val_pos = sum(
            p.get("valeur_actuelle", 0) for p in self.positions.values()
            if p.get("quantite", 0) > 0
        )
        nb_pos = sum(1 for p in self.positions.values() if p.get("quantite", 0) > 0)
        dd = (self.capital - self.plus_haut) / self.plus_haut * 100 if self.plus_haut > 0 else 0
        expo = val_pos / self.capital if self.capital > 0 else 0
        pnl_total = self.capital - self.config.capital_initial

        self.db.snapshot_portefeuille(
            self.capital, self.cash, val_pos, nb_pos,
            self.pnl_jour, pnl_total, dd, expo, benchmark_val,
        )
        return {
            "capital": round(self.capital, 2),
            "cash": round(self.cash, 2),
            "valeur_positions": round(val_pos, 2),
            "nb_positions": nb_pos,
            "pnl_total": round(pnl_total, 2),
            "pnl_pct": round(pnl_total / self.config.capital_initial * 100, 2),
            "drawdown": round(dd, 2),
            "exposition": round(expo, 2),
        }

    def reset(self):
        self.capital = self.config.capital_initial
        self.cash = self.config.capital_initial
        self.positions = {}
        self.plus_haut = self.capital
        self.pnl_jour = 0
        self.db.log("RESET", {"capital": self.capital}, self.version)
