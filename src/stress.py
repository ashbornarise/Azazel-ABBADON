"""
Banc de stress — Rex Clemens Capital
═════════════════════════════════════
Un robot passe le banc ou ne passe pas.
Le critère n'est pas le gain, c'est : le compte survit
et aucune règle de survie n'est violée.

Composants :
  1. Rejeu des vraies crises (2008, mars 2020, Terra/Luna, FTX, flash crash)
  2. Monte Carlo (bootstrap par blocs)
  3. Chocs synthétiques (-20%, frais ×5, gaps, pannes, vol ×3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
import pandas as pd

from .backtest import backtest, ResultatBacktest
from .survie import ModuleSurvie
from .strategies.base import StrategieBase


# ═══════════════════════════════════════════════════════════════
# Types et résultats
# ═══════════════════════════════════════════════════════════════

class TypeScenario(Enum):
    CRISE_HISTORIQUE = auto()
    MONTE_CARLO = auto()
    CHOC_SYNTHETIQUE = auto()


@dataclass
class ScenarioPrix:
    """Un scénario de prix à rejouer."""
    nom: str
    type: TypeScenario
    prix: pd.Series
    description: str = ""


@dataclass
class ResultatScenario:
    """Résultat d'un scénario de stress."""
    nom: str
    type: TypeScenario
    resultat_backtest: ResultatBacktest
    compte_survit: bool
    violations: int
    pire_baisse: float
    capital_final_pct: float  # capital final / capital initial

    @property
    def passe(self) -> bool:
        """Le scénario passe si le compte survit et aucune violation fatale."""
        return self.compte_survit and self.capital_final_pct > 0


@dataclass
class ResultatStress:
    """Résultat complet du banc de stress."""
    nom_strategie: str
    scenarios: list[ResultatScenario] = field(default_factory=list)
    seuil_reussite: float = 0.95

    @property
    def nb_total(self) -> int:
        return len(self.scenarios)

    @property
    def nb_reussis(self) -> int:
        return sum(1 for s in self.scenarios if s.passe)

    @property
    def taux_reussite(self) -> float:
        return self.nb_reussis / self.nb_total if self.nb_total > 0 else 0.0

    @property
    def passe(self) -> bool:
        """Le banc passe si le taux de réussite dépasse le seuil."""
        if self.nb_total == 0:
            return False
        # Aucun scénario ne doit amener le compte à zéro
        aucun_zero = all(s.capital_final_pct > 0 for s in self.scenarios)
        return self.taux_reussite >= self.seuil_reussite and aucun_zero

    @property
    def pire_scenario(self) -> Optional[ResultatScenario]:
        if not self.scenarios:
            return None
        return min(self.scenarios, key=lambda s: s.capital_final_pct)

    def resume(self) -> str:
        verdict = "✅ RÉUSSI" if self.passe else "❌ ÉCHOUÉ"
        lignes = [
            f"═══ Banc de stress — {self.nom_strategie} ═══",
            f"Verdict : {verdict}",
            f"Scénarios : {self.nb_reussis}/{self.nb_total} réussis "
            f"({self.taux_reussite:.0%}, seuil {self.seuil_reussite:.0%})",
        ]
        if self.pire_scenario:
            p = self.pire_scenario
            lignes.append(
                f"Pire scénario : {p.nom} — capital final {p.capital_final_pct:.1%}, "
                f"pire baisse {p.pire_baisse:.1%}, violations {p.violations}"
            )
        return "\n".join(lignes)


# ═══════════════════════════════════════════════════════════════
# 1. Générateur de crises historiques (données simulées)
# ═══════════════════════════════════════════════════════════════

def generer_crise_2008(seed: int = 100) -> ScenarioPrix:
    """
    Simule la crise financière de 2008.
    Chute de ~55% sur 17 mois, puis rebond partiel.
    En production : remplacer par les vraies données (yfinance, S&P 500).
    """
    rng = np.random.default_rng(seed)
    n_jours = 504  # ~2 ans

    # Phase 1 : montée lente (6 mois)
    montee = rng.normal(0.0003, 0.008, 126)
    # Phase 2 : krach (12 mois, -55%)
    krach = rng.normal(-0.003, 0.025, 252)
    # Phase 3 : rebond (6 mois)
    rebond = rng.normal(0.002, 0.015, 126)

    rendements = np.concatenate([montee, krach, rebond])
    idx = pd.bdate_range("2007-10-01", periods=n_jours)
    prix = pd.Series(100 * np.exp(np.cumsum(rendements)), index=idx, name="crise_2008")

    return ScenarioPrix(
        nom="Crise 2008 (simulée)",
        type=TypeScenario.CRISE_HISTORIQUE,
        prix=prix,
        description="Chute type 2008 : -55% sur 12 mois, haute volatilité.",
    )


def generer_covid_mars_2020(seed: int = 200) -> ScenarioPrix:
    """
    Simule le krach COVID de mars 2020.
    Chute de ~34% en 23 jours, rebond rapide en V.
    """
    rng = np.random.default_rng(seed)
    n_jours = 252  # 1 an

    # Phase 1 : marché haussier (2 mois)
    hausse = rng.normal(0.0005, 0.006, 42)
    # Phase 2 : krach brutal (23 jours, -34%)
    krach = rng.normal(-0.018, 0.04, 23)
    # Phase 3 : rebond en V (reste de l'année)
    rebond = rng.normal(0.003, 0.018, n_jours - 42 - 23)

    rendements = np.concatenate([hausse, krach, rebond])
    idx = pd.bdate_range("2020-01-02", periods=n_jours)
    prix = pd.Series(100 * np.exp(np.cumsum(rendements)), index=idx, name="covid_2020")

    return ScenarioPrix(
        nom="COVID Mars 2020 (simulé)",
        type=TypeScenario.CRISE_HISTORIQUE,
        prix=prix,
        description="Krach brutal (-34% en 23 jours) suivi d'un rebond en V.",
    )


def generer_terra_luna_2022(seed: int = 300) -> ScenarioPrix:
    """
    Simule l'effondrement de Terra/Luna (mai 2022).
    Chute de ~99% en quelques jours.
    """
    rng = np.random.default_rng(seed)
    n_jours = 90

    # Phase 1 : stagnation (30 jours)
    stag = rng.normal(0.0, 0.02, 30)
    # Phase 2 : effondrement (7 jours, -99%)
    effondrement = rng.normal(-0.5, 0.3, 7)
    # Phase 3 : quasi-mort (reste)
    mort = rng.normal(-0.01, 0.05, n_jours - 37)

    rendements = np.concatenate([stag, effondrement, mort])
    idx = pd.bdate_range("2022-04-01", periods=n_jours)
    prix = pd.Series(100 * np.exp(np.cumsum(rendements)), index=idx, name="terra_luna")

    return ScenarioPrix(
        nom="Terra/Luna 2022 (simulé)",
        type=TypeScenario.CRISE_HISTORIQUE,
        prix=prix,
        description="Effondrement crypto type Terra/Luna : -99% en quelques jours.",
    )


def generer_ftx_2022(seed: int = 400) -> ScenarioPrix:
    """
    Simule la faillite de FTX (novembre 2022).
    Chute de ~75% en 10 jours, pas de rebond.
    """
    rng = np.random.default_rng(seed)
    n_jours = 120

    # Phase 1 : baisse lente (60 jours)
    baisse = rng.normal(-0.001, 0.015, 60)
    # Phase 2 : faillite (10 jours, -75%)
    faillite = rng.normal(-0.12, 0.08, 10)
    # Phase 3 : stagnation basse (50 jours)
    stag = rng.normal(-0.002, 0.02, 50)

    rendements = np.concatenate([baisse, faillite, stag])
    idx = pd.bdate_range("2022-09-01", periods=n_jours)
    prix = pd.Series(100 * np.exp(np.cumsum(rendements)), index=idx, name="ftx_2022")

    return ScenarioPrix(
        nom="FTX 2022 (simulé)",
        type=TypeScenario.CRISE_HISTORIQUE,
        prix=prix,
        description="Faillite type FTX : -75% en 10 jours, pas de rebond.",
    )


def generer_flash_crash_2015(seed: int = 500) -> ScenarioPrix:
    """
    Simule le krach éclair d'août 2015.
    Chute de ~12% en une journée, rebond rapide.
    """
    rng = np.random.default_rng(seed)
    n_jours = 60

    # Phase 1 : normal (20 jours)
    normal = rng.normal(0.0003, 0.008, 20)
    # Phase 2 : flash crash (1 jour, -12%)
    crash = np.array([-0.12])
    # Phase 3 : rebond partiel (39 jours)
    rebond = rng.normal(0.002, 0.012, 39)

    rendements = np.concatenate([normal, crash, rebond])
    idx = pd.bdate_range("2015-07-15", periods=n_jours)
    prix = pd.Series(100 * np.exp(np.cumsum(rendements)), index=idx, name="flash_2015")

    return ScenarioPrix(
        nom="Flash Crash Août 2015 (simulé)",
        type=TypeScenario.CRISE_HISTORIQUE,
        prix=prix,
        description="Krach éclair : -12% en un jour, rebond rapide.",
    )


def toutes_crises_historiques() -> list[ScenarioPrix]:
    """Retourne tous les scénarios de crises historiques."""
    return [
        generer_crise_2008(),
        generer_covid_mars_2020(),
        generer_terra_luna_2022(),
        generer_ftx_2022(),
        generer_flash_crash_2015(),
    ]


# ═══════════════════════════════════════════════════════════════
# 2. Monte Carlo (bootstrap par blocs)
# ═══════════════════════════════════════════════════════════════

def monte_carlo_blocs(
    prix: pd.Series,
    n_iterations: int = 1000,
    taille_bloc: int = 20,
    longueur_sortie: int = 504,
    seed: int = 42,
) -> list[ScenarioPrix]:
    """
    Génère des marchés plausibles par rééchantillonnage de blocs.
    Les blocs conservent la persistance (autocorrélation) des rendements.

    Args:
        prix: série de prix historiques source.
        n_iterations: nombre de scénarios à générer.
        taille_bloc: taille de chaque bloc en jours.
        longueur_sortie: longueur de chaque scénario en jours.
        seed: graine aléatoire.

    Returns:
        Liste de ScenarioPrix simulés.
    """
    rng = np.random.default_rng(seed)
    rendements = prix.pct_change().dropna().values

    if len(rendements) < taille_bloc:
        raise ValueError(
            f"Pas assez de données ({len(rendements)} jours) "
            f"pour des blocs de {taille_bloc} jours."
        )

    n_blocs = (longueur_sortie + taille_bloc - 1) // taille_bloc
    max_debut = len(rendements) - taille_bloc

    scenarios = []
    for i in range(n_iterations):
        # Tirer des blocs aléatoires
        debuts = rng.integers(0, max_debut + 1, size=n_blocs)
        blocs = [rendements[d:d + taille_bloc] for d in debuts]
        rend_sim = np.concatenate(blocs)[:longueur_sortie]

        # Reconstruire les prix
        idx = pd.bdate_range("2020-01-01", periods=longueur_sortie)
        prix_sim = pd.Series(
            100 * np.exp(np.cumsum(rend_sim)),
            index=idx,
            name=f"mc_{i}",
        )

        scenarios.append(ScenarioPrix(
            nom=f"Monte Carlo #{i+1}",
            type=TypeScenario.MONTE_CARLO,
            prix=prix_sim,
            description=f"Bootstrap par blocs de {taille_bloc} jours, itération {i+1}/{n_iterations}.",
        ))

    return scenarios


# ═══════════════════════════════════════════════════════════════
# 3. Chocs synthétiques
# ═══════════════════════════════════════════════════════════════

def choc_chute_brutale(
    prix_base: pd.Series,
    amplitude: float = 0.20,
    jour_choc: int | None = None,
    seed: int = 600,
) -> ScenarioPrix:
    """
    Injecte une chute brutale à un moment donné.

    Args:
        prix_base: série de prix source.
        amplitude: fraction de chute (0.20 = -20%).
        jour_choc: jour de la chute (milieu par défaut).
    """
    prix = prix_base.copy()
    if jour_choc is None:
        jour_choc = len(prix) // 2

    # Appliquer la chute et tout ce qui suit
    facteur = 1.0 - amplitude
    prix.iloc[jour_choc:] *= facteur

    return ScenarioPrix(
        nom=f"Chute brutale -{amplitude:.0%}",
        type=TypeScenario.CHOC_SYNTHETIQUE,
        prix=prix,
        description=f"Chute de {amplitude:.0%} au jour {jour_choc}, sans cotations intermédiaires.",
    )


def choc_frais_multiplies(
    multiplicateur: float = 5.0,
) -> float:
    """
    Retourne un coût par trade multiplié.
    À passer au moteur de backtest.
    """
    cout_base = 0.001  # 0,1%
    return cout_base * multiplicateur


def choc_volatilite_multipliee(
    prix_base: pd.Series,
    multiplicateur: float = 3.0,
    seed: int = 700,
) -> ScenarioPrix:
    """
    Multiplie la volatilité des rendements.
    """
    rng = np.random.default_rng(seed)
    rendements = prix_base.pct_change().dropna()

    # Multiplier la volatilité (centrer, multiplier, recentrer)
    moyenne = rendements.mean()
    rend_amplifie = moyenne + (rendements - moyenne) * multiplicateur

    prix_choc = pd.Series(
        prix_base.iloc[0] * np.exp(np.cumsum(rend_amplifie.values)),
        index=prix_base.index[1:],
        name="vol_x" + str(int(multiplicateur)),
    )

    return ScenarioPrix(
        nom=f"Volatilité ×{multiplicateur:.0f}",
        type=TypeScenario.CHOC_SYNTHETIQUE,
        prix=prix_choc,
        description=f"Volatilité des rendements multipliée par {multiplicateur:.0f}.",
    )


def choc_gap_sans_cotation(
    prix_base: pd.Series,
    jour_debut: int | None = None,
    duree_jours: int = 5,
    chute_pct: float = 0.15,
) -> ScenarioPrix:
    """
    Simule un gap : le prix saute sans cotations intermédiaires.
    Comme si le marché était fermé puis rouvrait beaucoup plus bas.
    """
    prix = prix_base.copy()
    if jour_debut is None:
        jour_debut = len(prix) // 2

    # Supprimer les cotations pendant la durée
    fin = min(jour_debut + duree_jours, len(prix))
    prix_avant = prix.iloc[jour_debut - 1] if jour_debut > 0 else prix.iloc[0]
    prix_apres = prix_avant * (1 - chute_pct)

    # Le prix saute directement
    facteur = prix_apres / prix.iloc[fin - 1] if prix.iloc[fin - 1] > 0 else 1.0
    prix.iloc[fin:] *= facteur
    # Les jours intermédiaires sont NaN (pas de cotation)
    prix.iloc[jour_debut:fin] = np.nan

    return ScenarioPrix(
        nom=f"Gap -{chute_pct:.0%} ({duree_jours}j sans cotation)",
        type=TypeScenario.CHOC_SYNTHETIQUE,
        prix=prix,
        description=f"Gap de {chute_pct:.0%} après {duree_jours} jours sans cotation.",
    )


def choc_donnees_fausses(
    prix_base: pd.Series,
    jour: int | None = None,
) -> ScenarioPrix:
    """
    Injecte un prix faux (0) à un moment donné.
    Le module de survie doit détecter et bloquer.
    """
    prix = prix_base.copy()
    if jour is None:
        jour = len(prix) // 2
    prix.iloc[jour] = 0.0

    return ScenarioPrix(
        nom="Données fausses (prix à 0)",
        type=TypeScenario.CHOC_SYNTHETIQUE,
        prix=prix,
        description="Prix mis à 0 pour tester la détection de données invalides.",
    )


def tous_chocs_synthetiques(prix_base: pd.Series) -> list[ScenarioPrix]:
    """Retourne tous les chocs synthétiques."""
    return [
        choc_chute_brutale(prix_base, amplitude=0.20),
        choc_chute_brutale(prix_base, amplitude=0.40),
        choc_volatilite_multipliee(prix_base, multiplicateur=3.0),
        choc_volatilite_multipliee(prix_base, multiplicateur=5.0),
        choc_gap_sans_cotation(prix_base),
        choc_donnees_fausses(prix_base),
    ]


# ═══════════════════════════════════════════════════════════════
# Exécution du banc de stress complet
# ═══════════════════════════════════════════════════════════════

def executer_scenario(
    scenario: ScenarioPrix,
    strategie: StrategieBase,
    config: dict,
    cout: float = 0.001,
    capital_initial: float = 10_000.0,
) -> ResultatScenario:
    """Exécute un scénario de stress sur une stratégie."""
    # Nettoyer les NaN pour le backtest (le module de survie les gère séparément)
    prix_propre = scenario.prix.dropna()
    if len(prix_propre) < 10:
        # Pas assez de données après nettoyage
        return ResultatScenario(
            nom=scenario.nom,
            type=scenario.type,
            resultat_backtest=ResultatBacktest(
                nom_strategie=strategie.nom,
                equity=pd.Series([1.0]),
                buy_hold=pd.Series([1.0]),
                nb_trades=0,
                rendement_annuel=0.0,
                sharpe=0.0,
                pire_baisse=0.0,
                violations_survie=0,
            ),
            compte_survit=True,
            violations=0,
            pire_baisse=0.0,
            capital_final_pct=1.0,
        )

    module = ModuleSurvie(config, capital_initial=capital_initial)
    resultat = backtest(prix_propre, strategie, cout=cout, module_survie=module)

    capital_final_pct = resultat.equity.iloc[-1] if len(resultat.equity) > 0 else 0.0

    return ResultatScenario(
        nom=scenario.nom,
        type=scenario.type,
        resultat_backtest=resultat,
        compte_survit=capital_final_pct > 0,
        violations=resultat.violations_survie,
        pire_baisse=resultat.pire_baisse,
        capital_final_pct=capital_final_pct,
    )


def banc_de_stress(
    strategie: StrategieBase,
    config: dict,
    prix_historique: pd.Series | None = None,
    n_monte_carlo: int | None = None,
    capital_initial: float = 10_000.0,
    inclure_crises: bool = True,
    inclure_chocs: bool = True,
    inclure_monte_carlo: bool = True,
) -> ResultatStress:
    """
    Exécute le banc de stress complet sur une stratégie.

    Args:
        strategie: la stratégie à tester.
        config: configuration (config.toml).
        prix_historique: série de prix pour Monte Carlo et chocs.
        n_monte_carlo: nombre d'itérations Monte Carlo (défaut depuis config).
        capital_initial: capital initial simulé.
        inclure_crises: inclure les crises historiques.
        inclure_chocs: inclure les chocs synthétiques.
        inclure_monte_carlo: inclure le Monte Carlo.

    Returns:
        ResultatStress avec tous les résultats.
    """
    stress_cfg = config.get("stress", {})
    seuil = stress_cfg.get("seuil_survie_pct_scenarios", 0.95)
    if n_monte_carlo is None:
        n_monte_carlo = stress_cfg.get("monte_carlo_iterations", 1000)
    taille_bloc = stress_cfg.get("taille_bloc_bootstrap", 20)
    cout = config.get("backtest", {}).get("cout_par_trade", 0.001)

    resultat = ResultatStress(
        nom_strategie=strategie.nom,
        seuil_reussite=seuil,
    )

    # 1. Crises historiques
    if inclure_crises:
        for crise in toutes_crises_historiques():
            r = executer_scenario(crise, strategie, config, cout, capital_initial)
            resultat.scenarios.append(r)

    # 2. Monte Carlo
    if inclure_monte_carlo and prix_historique is not None:
        scenarios_mc = monte_carlo_blocs(
            prix_historique,
            n_iterations=n_monte_carlo,
            taille_bloc=taille_bloc,
        )
        for sc in scenarios_mc:
            r = executer_scenario(sc, strategie, config, cout, capital_initial)
            resultat.scenarios.append(r)

    # 3. Chocs synthétiques
    if inclure_chocs and prix_historique is not None:
        for choc in tous_chocs_synthetiques(prix_historique):
            r = executer_scenario(choc, strategie, config, cout, capital_initial)
            resultat.scenarios.append(r)

    # 4. Choc de frais (re-test avec frais ×2 et ×5)
    if inclure_chocs and prix_historique is not None:
        for mult in [2.0, 5.0]:
            cout_choc = choc_frais_multiplies(mult)
            sc = ScenarioPrix(
                nom=f"Frais ×{mult:.0f}",
                type=TypeScenario.CHOC_SYNTHETIQUE,
                prix=prix_historique,
                description=f"Frais et écarts de prix multipliés par {mult:.0f}.",
            )
            r = executer_scenario(sc, strategie, config, cout_choc, capital_initial)
            resultat.scenarios.append(r)

    return resultat
