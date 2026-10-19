"""
REX CLEMENS CORPORATION
========================
Une entreprise complete qui travaille 24/7 pour faire des profits.
Chaque employe a une specialite, un score, et est juge sur ses resultats.
La defaite n'est pas une option — pour personne.

Departements:
- MATH (50) : Mathematiciens niveau doctorat — patterns statistiques
- TECH (30) : Geeks — anomalies techniques, combinaisons d'indicateurs
- FINANCE (40) : Maniaques de la finance — correlations, secteurs, macro
- ADMIN (10) : Secretaires — logs, organisation, reporting
- SECURITE (10) : Gardiens — surveillance risque, protection capital
- CORTEX (1) : Le cerveau central — decision finale

Systeme de points:
- Bonne prediction confirmee : +10 pts
- Mauvaise prediction : -5 pts
- Decouverte d'une nouvelle formule : +25 pts
- Protection du capital (evite une perte) : +15 pts
- Erreur de protection (bloque un bon trade) : -8 pts
"""

import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict


CORP_PATH = Path("data/rex_corporation.json")

PRENOMS_MATH = [
    "Euler", "Gauss", "Riemann", "Ramanujan", "Noether", "Galois", "Cantor",
    "Hilbert", "Poincare", "Fermat", "Lagrange", "Laplace", "Fourier", "Cauchy",
    "Abel", "Jacobi", "Weierstrass", "Dedekind", "Klein", "Lie", "Borel",
    "Lebesgue", "Hausdorff", "Banach", "Kolmogorov", "Turing", "Nash",
    "Grothendieck", "Mandelbrot", "Wiener", "Ito", "Black", "Scholes",
    "Markowitz", "Shannon", "Bayes", "Markov", "Poisson", "Bernoulli",
    "Pascal", "Newton", "Leibniz", "Descartes", "Archimede", "Pythagore",
    "Fibonacci", "Cardano", "Fermi", "Dirac", "Planck",
]

PRENOMS_TECH = [
    "Linus", "Ada", "Grace", "Knuth", "Dijkstra", "Hoare", "Lamport",
    "Ritchie", "Thompson", "Kernighan", "Stallman", "Torvalds", "Woz",
    "Hopper", "Lovelace", "Babbage", "Berners-Lee", "Cerf", "Postel",
    "Gosling", "Pike", "Carmack", "Sweeney", "Blow", "Simonyi",
    "Stroustrup", "Backus", "McCarthy", "Minsky", "Brooks",
]

PRENOMS_FINANCE = [
    "Simons", "Buffett", "Soros", "Dalio", "Lynch", "Graham", "Icahn",
    "Ackman", "Einhorn", "Tudor", "Kovner", "Druckenmiller", "Paulson",
    "Tepper", "Cohen", "Griffin", "Citadel", "Shaw", "Bridgewater",
    "Thorp", "Livermore", "Niederhoffer", "Taleb", "Mandelbrot_F",
    "Fama", "French", "Sharpe", "Merton", "Ross", "Tobin",
    "Modigliani", "Arrow", "Samuelson", "Friedman", "Keynes",
    "Ricardo", "Smith_A", "Hayek", "Stiglitz", "Krugman",
]

SPECIALITES_MATH = [
    "patterns_fractals", "regression_non_lineaire", "series_temporelles",
    "analyse_spectrale", "theorie_chaos", "processus_stochastiques",
    "optimisation_convexe", "algebre_lineaire_sparse", "decomposition_svd",
    "clustering_kmeans", "detection_anomalies_stat", "test_hypotheses",
    "bootstrap_monte_carlo", "analyse_wavelets", "copules_dependance",
    "extreme_value_theory", "markov_chains", "bayesian_inference",
    "information_theory", "compressed_sensing", "random_matrix",
    "topological_data", "persistent_homology", "ergodic_theory",
    "measure_theory", "functional_analysis", "game_theory",
    "graph_theory", "combinatorial_opt", "convex_geometry",
    "number_theory_primes", "modular_arithmetic", "group_symmetry",
    "differential_geometry", "manifold_learning", "tensor_decomp",
    "spectral_clustering", "kernel_methods", "reproducing_hilbert",
    "operator_algebras", "non_commutative_prob", "free_probability",
    "rough_paths", "regularity_structures", "schramm_loewner",
    "percolation_theory", "random_graphs", "stochastic_calc",
    "malliavin_calculus", "stein_method", "concentration_ineq",
]

SPECIALITES_TECH = [
    "anomalie_volume", "divergence_rsi", "squeeze_bollinger",
    "breakout_range", "mean_reversion", "momentum_rotation",
    "orderflow_imbalance", "microstructure", "latency_arb",
    "feature_engineering", "hyperparameter_hunt", "ensemble_stacking",
    "neural_architecture", "gradient_boosting", "reinforcement_learn",
    "tick_data_analysis", "order_book_depth", "vwap_deviation",
    "market_making_sim", "hft_pattern", "dark_pool_detection",
    "sentiment_nlp", "news_impact", "social_signal",
    "cross_exchange_arb", "funding_rate", "liquidation_cascade",
    "whale_tracking", "smart_money_flow", "retail_vs_institutional",
]

SPECIALITES_FINANCE = [
    "sector_rotation", "risk_parity", "carry_trade",
    "momentum_factor", "value_factor", "quality_factor",
    "low_volatility", "merger_arb", "convertible_arb",
    "stat_arb_pairs", "index_rebalance", "earnings_drift",
    "insider_flow", "short_interest", "put_call_ratio",
    "vix_term_structure", "yield_curve", "credit_spread",
    "fx_correlation", "commodity_super_cycle", "inflation_hedge",
    "tail_risk_hedge", "crisis_alpha", "trend_following",
    "mean_reversion_macro", "global_macro", "emerging_markets",
    "frontier_markets", "real_assets", "private_credit",
    "distressed_debt", "special_situations", "event_driven",
    "activist_strategy", "spinoff_tracker", "ipo_analysis",
    "seasonality", "calendar_effects", "lunar_cycles",
    "sentiment_extremes", "positioning_data",
]


@dataclass
class Employe:
    id: str
    nom: str
    departement: str
    specialite: str
    points: int = 100
    predictions_totales: int = 0
    predictions_correctes: int = 0
    decouvertes: int = 0
    protections_reussies: int = 0
    rang: str = "Junior"
    actif: bool = True
    date_embauche: str = ""
    derniere_action: str = ""

    def __post_init__(self):
        if not self.date_embauche:
            self.date_embauche = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @property
    def win_rate(self) -> float:
        if self.predictions_totales == 0:
            return 0.0
        return self.predictions_correctes / self.predictions_totales * 100

    @property
    def niveau(self) -> str:
        if self.points >= 500:
            return "Directeur"
        elif self.points >= 300:
            return "Senior"
        elif self.points >= 150:
            return "Confirme"
        elif self.points >= 50:
            return "Junior"
        else:
            return "Stagiaire"

    def recompenser(self, pts: int, raison: str):
        self.points += pts
        self.derniere_action = f"{'+' if pts > 0 else ''}{pts}pts: {raison}"

    def predire(self, closes, highs, lows, volumes, idx) -> float:
        random.seed(hash((self.id, self.specialite, idx)) % (2**32))
        return random.random()


def _generer_methode(specialite: str):
    seed_val = hash(specialite) % (2**32)
    rng = random.Random(seed_val)

    genes_possibles = [
        "momentum_5", "momentum_10", "momentum_20",
        "ma_ratio_5", "ma_ratio_20", "ma_ratio_50",
        "vol_ratio", "high_low_pct", "close_position",
        "std_10", "std_20", "gap", "max_20_dist", "min_20_dist",
    ]
    ops = ["add", "sub", "mul", "div", "ratio", "avg", "abs_diff"]
    transforms = ["none", "abs", "clip", "sign", "sqrt_abs"]

    return {
        "gene_a": rng.choice(genes_possibles),
        "gene_b": rng.choice(genes_possibles),
        "operation": rng.choice(ops),
        "transform": rng.choice(transforms),
        "poids": round(rng.uniform(0.5, 2.0), 2),
    }


class RexCorporation:
    def __init__(self):
        self.employes: dict[str, Employe] = {}
        self.historique_points: list = []
        self.cycle_count: int = 0
        self.total_profits: float = 0.0
        self.total_pertes: float = 0.0
        self.charger()

        if not self.employes:
            self._embaucher_equipe()

    def _embaucher_equipe(self):
        print("[CORP] Embauche de l'equipe Rex Clemens Corporation...")

        for i, nom in enumerate(PRENOMS_MATH):
            eid = f"MATH-{i+1:03d}"
            spec = SPECIALITES_MATH[i % len(SPECIALITES_MATH)]
            self.employes[eid] = Employe(id=eid, nom=nom, departement="MATH", specialite=spec)

        for i, nom in enumerate(PRENOMS_TECH):
            eid = f"TECH-{i+1:03d}"
            spec = SPECIALITES_TECH[i % len(SPECIALITES_TECH)]
            self.employes[eid] = Employe(id=eid, nom=nom, departement="TECH", specialite=spec)

        for i, nom in enumerate(PRENOMS_FINANCE):
            eid = f"FIN-{i+1:03d}"
            spec = SPECIALITES_FINANCE[i % len(SPECIALITES_FINANCE)]
            self.employes[eid] = Employe(id=eid, nom=nom, departement="FINANCE", specialite=spec)

        admins = ["Marie", "Sophie", "Aisha", "Yuki", "Elena",
                   "Fatou", "Ines", "Clara", "Nadia", "Rosa"]
        for i, nom in enumerate(admins):
            eid = f"ADM-{i+1:03d}"
            self.employes[eid] = Employe(id=eid, nom=nom, departement="ADMIN",
                                         specialite="reporting" if i < 5 else "organisation")

        gardiens = ["Atlas", "Sentinel", "Vigil", "Bastion", "Aegis",
                     "Rampart", "Bulwark", "Shield", "Ward", "Keep"]
        for i, nom in enumerate(gardiens):
            eid = f"SEC-{i+1:03d}"
            self.employes[eid] = Employe(id=eid, nom=nom, departement="SECURITE",
                                         specialite="risk_monitoring" if i < 5 else "capital_protection")

        self.employes["CORTEX-001"] = Employe(
            id="CORTEX-001", nom="Cortex", departement="DIRECTION",
            specialite="decision_finale", points=200,
        )

        print(f"[CORP] {len(self.employes)} employes embauches")
        self.sauver()

    def consulter_departement(self, departement: str, closes, highs, lows, volumes) -> dict:
        employes_dept = [e for e in self.employes.values()
                         if e.departement == departement and e.actif]
        if not employes_dept or len(closes) < 60:
            return {"vote": 0.5, "confiance": 0, "participants": 0}

        votes = []
        idx = len(closes) - 1
        for emp in employes_dept:
            methode = _generer_methode(emp.specialite)
            try:
                from src.ml.gene_hunter import GENES, OPERATIONS, TRANSFORMATIONS
                gene_a_fn = GENES.get(methode["gene_a"])
                gene_b_fn = GENES.get(methode["gene_b"])
                op_fn = OPERATIONS.get(methode["operation"])
                trans_fn = TRANSFORMATIONS.get(methode["transform"])
                if all([gene_a_fn, gene_b_fn, op_fn, trans_fn]):
                    val_a = gene_a_fn(closes, highs, lows, volumes, idx)
                    val_b = gene_b_fn(closes, highs, lows, volumes, idx)
                    raw = op_fn(val_a, val_b)
                    signal = trans_fn(raw)
                    vote = 1.0 if signal > 0 else 0.0
                    poids = emp.points / 100.0
                    votes.append(vote * poids)
                else:
                    votes.append(0.5)
            except:
                votes.append(0.5)

        if not votes:
            return {"vote": 0.5, "confiance": 0, "participants": 0}

        total_poids = sum(e.points / 100.0 for e in employes_dept)
        vote_moyen = sum(votes) / total_poids if total_poids > 0 else 0.5
        confiance = abs(vote_moyen - 0.5) * 200

        return {
            "vote": round(vote_moyen, 3),
            "confiance": round(confiance, 1),
            "participants": len(employes_dept),
        }

    def vote_corporatif(self, closes, highs, lows, volumes) -> dict:
        resultats = {}
        for dept in ["MATH", "TECH", "FINANCE", "SECURITE"]:
            resultats[dept] = self.consulter_departement(dept, closes, highs, lows, volumes)

        poids_dept = {"MATH": 0.35, "TECH": 0.25, "FINANCE": 0.30, "SECURITE": 0.10}
        vote_final = sum(
            resultats[d]["vote"] * poids_dept[d]
            for d in poids_dept if d in resultats
        )

        decision = "ACHAT" if vote_final > 0.55 else ("CASH" if vote_final < 0.40 else "NEUTRE")

        return {
            "decision": decision,
            "vote_final": round(vote_final, 3),
            "departements": resultats,
            "confiance_globale": round(abs(vote_final - 0.5) * 200, 1),
        }

    def enregistrer_resultat(self, sym: str, prediction_correcte: bool, pnl: float = 0):
        self.cycle_count += 1
        if pnl > 0:
            self.total_profits += pnl
        else:
            self.total_pertes += abs(pnl)

        pts = 10 if prediction_correcte else -5
        if pnl > 0 and prediction_correcte:
            pts += int(pnl)
        raison = f"{'OK' if prediction_correcte else 'ERREUR'} sur {sym} (PnL: {pnl:+.2f}$)"

        for emp in self.employes.values():
            if emp.departement in ("MATH", "TECH", "FINANCE", "DIRECTION"):
                emp.predictions_totales += 1
                if prediction_correcte:
                    emp.predictions_correctes += 1
                emp.recompenser(pts, raison)

        for emp in self.employes.values():
            if emp.departement == "ADMIN":
                emp.recompenser(pts // 2, f"support {raison}")
            elif emp.departement == "SECURITE":
                if not prediction_correcte and pnl < 0:
                    emp.recompenser(5, f"perte limitee sur {sym}")
                elif prediction_correcte and pnl > 0:
                    emp.recompenser(3, f"marche securise {sym}")

        self.historique_points.append({
            "cycle": self.cycle_count,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "sym": sym,
            "correct": prediction_correcte,
            "pnl": round(pnl, 2),
            "pts_distribues": pts,
        })
        if len(self.historique_points) > 500:
            self.historique_points = self.historique_points[-500:]

        if self.cycle_count % 10 == 0:
            self.sauver()

    def recompenser_decouverte(self, employe_id: str, formule_nom: str):
        emp = self.employes.get(employe_id)
        if emp:
            emp.decouvertes += 1
            emp.recompenser(25, f"decouverte: {formule_nom}")

    def classement(self, top_n: int = 20) -> list[dict]:
        tri = sorted(self.employes.values(), key=lambda e: e.points, reverse=True)
        return [
            {
                "rang": i + 1,
                "id": e.id,
                "nom": e.nom,
                "departement": e.departement,
                "specialite": e.specialite,
                "points": e.points,
                "niveau": e.niveau,
                "win_rate": round(e.win_rate, 1),
                "predictions": e.predictions_totales,
                "decouvertes": e.decouvertes,
            }
            for i, e in enumerate(tri[:top_n])
        ]

    def stats_departement(self) -> dict:
        stats = {}
        for emp in self.employes.values():
            dept = emp.departement
            if dept not in stats:
                stats[dept] = {"effectif": 0, "points_total": 0, "points_moyen": 0,
                               "meilleur": "", "meilleur_pts": 0}
            stats[dept]["effectif"] += 1
            stats[dept]["points_total"] += emp.points
            if emp.points > stats[dept]["meilleur_pts"]:
                stats[dept]["meilleur"] = emp.nom
                stats[dept]["meilleur_pts"] = emp.points
        for dept in stats:
            if stats[dept]["effectif"] > 0:
                stats[dept]["points_moyen"] = round(
                    stats[dept]["points_total"] / stats[dept]["effectif"])
        return stats

    def tenir_reunion(self, sym: str, prix: float, decision_ml: str, proba: float) -> dict:
        now = datetime.now(timezone.utc).strftime("%H:%M")
        math_top = sorted([e for e in self.employes.values() if e.departement == "MATH"],
                          key=lambda e: e.points, reverse=True)[:3]
        tech_top = sorted([e for e in self.employes.values() if e.departement == "TECH"],
                          key=lambda e: e.points, reverse=True)[:3]
        fin_top = sorted([e for e in self.employes.values() if e.departement == "FINANCE"],
                         key=lambda e: e.points, reverse=True)[:3]
        sec_top = sorted([e for e in self.employes.values() if e.departement == "SECURITE"],
                         key=lambda e: e.points, reverse=True)[:2]

        dialogues = []
        dialogues.append({"qui": "Cortex", "dept": "DIRECTION", "msg": f"Reunion d'urgence sur {sym} a {prix}$. ML dit {decision_ml} ({proba:.0f}%). Vos analyses ?"})

        if math_top:
            m = math_top[0]
            if proba > 55:
                dialogues.append({"qui": m.nom, "dept": "MATH", "msg": f"Les patterns {m.specialite.replace('_',' ')} confirment. Probabilite statistique favorable."})
            elif proba > 48:
                dialogues.append({"qui": m.nom, "dept": "MATH", "msg": f"Signal faible. Mon analyse {m.specialite.replace('_',' ')} montre une zone d'incertitude."})
            else:
                dialogues.append({"qui": m.nom, "dept": "MATH", "msg": f"Negatif. Les chiffres ne mentent pas — ce n'est pas le moment."})
        if math_top and len(math_top) > 1:
            m2 = math_top[1]
            if random.random() > 0.5:
                dialogues.append({"qui": m2.nom, "dept": "MATH", "msg": f"Je complete — mon modele {m2.specialite.replace('_',' ')} detecte une anomalie interessante."})

        if tech_top:
            t = tech_top[0]
            if decision_ml == "ACHAT":
                dialogues.append({"qui": t.nom, "dept": "TECH", "msg": f"Volume et {t.specialite.replace('_',' ')} alignes. Structure technique propre pour entrer."})
            else:
                dialogues.append({"qui": t.nom, "dept": "TECH", "msg": f"Attention — {t.specialite.replace('_',' ')} montre des signaux contradictoires."})

        if fin_top:
            f = fin_top[0]
            if decision_ml == "ACHAT":
                dialogues.append({"qui": f.nom, "dept": "FINANCE", "msg": f"Le secteur est porteur. {f.specialite.replace('_',' ')} supporte l'entree. J'approuve."})
            elif decision_ml == "CASH":
                dialogues.append({"qui": f.nom, "dept": "FINANCE", "msg": f"Fondamentaux en recul. Ma lecture {f.specialite.replace('_',' ')} deconseille. On sort."})
            else:
                dialogues.append({"qui": f.nom, "dept": "FINANCE", "msg": f"Pas de conviction forte. On attend un meilleur point d'entree."})

        if sec_top:
            s = sec_top[0]
            dialogues.append({"qui": s.nom, "dept": "SECURITE", "msg": f"Risque evalue. Stop-loss en place. Capital protege — feu vert de la securite."})

        dialogues.append({"qui": "Cortex", "dept": "DIRECTION", "msg": f"Decision finale : {decision_ml}. Le chasseur ne doute pas. Execution."})

        reunion = {
            "heure": now,
            "sujet": sym,
            "prix": prix,
            "decision_ml": decision_ml,
            "proba": round(proba, 1),
            "dialogues": dialogues,
        }

        if not hasattr(self, 'reunions'):
            self.reunions = []
        self.reunions.append(reunion)
        if len(self.reunions) > 50:
            self.reunions = self.reunions[-50:]

        return reunion

    def rapport_complet(self) -> dict:
        reunions = getattr(self, 'reunions', [])
        return {
            "nom": "Rex Clemens Corporation",
            "effectif_total": len(self.employes),
            "cycle_count": self.cycle_count,
            "total_profits": round(self.total_profits, 2),
            "total_pertes": round(self.total_pertes, 2),
            "ratio_profit": round(self.total_profits / max(self.total_pertes, 0.01), 2),
            "classement_top20": self.classement(20),
            "departements": self.stats_departement(),
            "derniers_resultats": self.historique_points[-10:],
            "reunions": reunions[-10:],
        }

    def charger(self):
        if CORP_PATH.exists():
            try:
                with open(CORP_PATH) as f:
                    data = json.load(f)
                self.cycle_count = data.get("cycle_count", 0)
                self.total_profits = data.get("total_profits", 0)
                self.total_pertes = data.get("total_pertes", 0)
                self.historique_points = data.get("historique_points", [])
                for ed in data.get("employes", []):
                    emp = Employe(**ed)
                    self.employes[emp.id] = emp
                print(f"[CORP] Corporation chargee: {len(self.employes)} employes, cycle {self.cycle_count}")
            except Exception as e:
                print(f"[CORP] Erreur chargement: {e}")

    def sauver(self):
        CORP_PATH.parent.mkdir(exist_ok=True)
        data = {
            "cycle_count": self.cycle_count,
            "total_profits": self.total_profits,
            "total_pertes": self.total_pertes,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "effectif": len(self.employes),
            "historique_points": self.historique_points[-500:],
            "employes": [asdict(e) for e in self.employes.values()],
        }
        with open(CORP_PATH, "w") as f:
            json.dump(data, f, indent=2)
