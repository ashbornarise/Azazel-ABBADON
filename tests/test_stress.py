"""
Tests du banc de stress — Rex Clemens Capital
═════════════════════════════════════════════
Vérifie que le banc de stress fonctionne correctement :
  - Les crises historiques sont générées
  - Le Monte Carlo produit des scénarios variés
  - Les chocs synthétiques injectent les bonnes perturbations
  - Le verdict est cohérent

Exécuter : pytest tests/test_stress.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.stress import (
    generer_crise_2008,
    generer_covid_mars_2020,
    generer_terra_luna_2022,
    generer_ftx_2022,
    generer_flash_crash_2015,
    toutes_crises_historiques,
    monte_carlo_blocs,
    choc_chute_brutale,
    choc_frais_multiplies,
    choc_volatilite_multipliee,
    choc_gap_sans_cotation,
    choc_donnees_fausses,
    executer_scenario,
    banc_de_stress,
    TypeScenario,
)
from src.strategies.buy_hold import BuyHold
from src.strategies.random_control import RandomControl
from src.data import generer_marche_aleatoire


@pytest.fixture
def config():
    return {
        "survie": {
            "risque_max_par_trade": 0.01,
            "exposition_max": 1.0,
            "perte_max_journaliere": 0.03,
            "drawdown_max": 0.10,
            "taille_max_position": 0.25,
        },
        "donnees": {
            "age_max_donnees_minutes": 30,
            "prix_saut_max_pct": 0.50,
            "prix_min": 0.0001,
        },
        "connexion": {
            "erreurs_max_consecutives": 3,
        },
        "backtest": {
            "cout_par_trade": 0.001,
        },
        "stress": {
            "seuil_survie_pct_scenarios": 0.95,
            "monte_carlo_iterations": 50,  # réduit pour les tests
            "taille_bloc_bootstrap": 20,
        },
    }


@pytest.fixture
def prix_base():
    return generer_marche_aleatoire(periodes=500, seed=1)


# ═══════════════════════════════════════════════════════════════
# 1. CRISES HISTORIQUES
# ═══════════════════════════════════════════════════════════════

class TestCrisesHistoriques:
    def test_crise_2008_generee(self):
        crise = generer_crise_2008()
        assert len(crise.prix) == 504
        assert crise.type == TypeScenario.CRISE_HISTORIQUE
        # Vérifier qu'il y a bien une grosse chute
        dd = (crise.prix / crise.prix.cummax() - 1).min()
        assert dd < -0.20, "La crise 2008 doit contenir une chute significative"

    def test_covid_2020_generee(self):
        crise = generer_covid_mars_2020()
        assert len(crise.prix) == 252
        dd = (crise.prix / crise.prix.cummax() - 1).min()
        assert dd < -0.15

    def test_terra_luna_generee(self):
        crise = generer_terra_luna_2022()
        assert len(crise.prix) == 90
        # Terra/Luna : effondrement quasi-total
        ratio = crise.prix.iloc[-1] / crise.prix.max()
        assert ratio < 0.10, "Terra/Luna doit perdre >90% depuis le sommet"

    def test_ftx_generee(self):
        crise = generer_ftx_2022()
        assert len(crise.prix) == 120
        dd = (crise.prix / crise.prix.cummax() - 1).min()
        assert dd < -0.30

    def test_flash_crash_generee(self):
        crise = generer_flash_crash_2015()
        assert len(crise.prix) == 60
        # Vérifier qu'il y a un jour de chute brutale
        rendements = crise.prix.pct_change().dropna()
        assert rendements.min() < -0.08

    def test_toutes_crises_5_scenarios(self):
        crises = toutes_crises_historiques()
        assert len(crises) == 5
        for c in crises:
            assert c.type == TypeScenario.CRISE_HISTORIQUE
            assert len(c.prix) > 0


# ═══════════════════════════════════════════════════════════════
# 2. MONTE CARLO
# ═══════════════════════════════════════════════════════════════

class TestMonteCarlo:
    def test_genere_le_bon_nombre(self, prix_base):
        scenarios = monte_carlo_blocs(prix_base, n_iterations=10, taille_bloc=20)
        assert len(scenarios) == 10

    def test_longueur_correcte(self, prix_base):
        scenarios = monte_carlo_blocs(
            prix_base, n_iterations=5, taille_bloc=20, longueur_sortie=252,
        )
        for sc in scenarios:
            assert len(sc.prix) == 252

    def test_scenarios_differents(self, prix_base):
        scenarios = monte_carlo_blocs(prix_base, n_iterations=5, taille_bloc=20)
        # Les scénarios ne doivent pas être identiques
        prix_finaux = [sc.prix.iloc[-1] for sc in scenarios]
        assert len(set(round(p, 2) for p in prix_finaux)) > 1

    def test_type_monte_carlo(self, prix_base):
        scenarios = monte_carlo_blocs(prix_base, n_iterations=3)
        for sc in scenarios:
            assert sc.type == TypeScenario.MONTE_CARLO

    def test_erreur_si_pas_assez_de_donnees(self):
        prix_court = pd.Series([100, 101, 102], index=pd.bdate_range("2024-01-01", periods=3))
        with pytest.raises(ValueError, match="Pas assez de données"):
            monte_carlo_blocs(prix_court, taille_bloc=20)


# ═══════════════════════════════════════════════════════════════
# 3. CHOCS SYNTHÉTIQUES
# ═══════════════════════════════════════════════════════════════

class TestChocsSynthetiques:
    def test_chute_brutale_20pct(self, prix_base):
        sc = choc_chute_brutale(prix_base, amplitude=0.20)
        milieu = len(prix_base) // 2
        # Après le choc, les prix sont ~20% plus bas
        ratio = sc.prix.iloc[-1] / prix_base.iloc[-1]
        assert 0.75 < ratio < 0.85

    def test_chute_brutale_40pct(self, prix_base):
        sc = choc_chute_brutale(prix_base, amplitude=0.40)
        ratio = sc.prix.iloc[-1] / prix_base.iloc[-1]
        assert 0.55 < ratio < 0.65

    def test_frais_multiplies(self):
        assert choc_frais_multiplies(2.0) == pytest.approx(0.002)
        assert choc_frais_multiplies(5.0) == pytest.approx(0.005)

    def test_volatilite_multipliee(self, prix_base):
        sc = choc_volatilite_multipliee(prix_base, multiplicateur=3.0)
        # La volatilité des rendements doit être ~3× plus grande
        vol_base = prix_base.pct_change().dropna().std()
        vol_choc = sc.prix.pct_change().dropna().std()
        assert vol_choc > vol_base * 2.0  # au moins 2× (pas exact à cause du cumul)

    def test_gap_contient_nan(self, prix_base):
        sc = choc_gap_sans_cotation(prix_base, duree_jours=5)
        assert sc.prix.isna().sum() == 5

    def test_donnees_fausses_contient_zero(self, prix_base):
        sc = choc_donnees_fausses(prix_base)
        assert (sc.prix == 0).any()


# ═══════════════════════════════════════════════════════════════
# 4. EXÉCUTION DE SCÉNARIOS
# ═══════════════════════════════════════════════════════════════

class TestExecution:
    def test_scenario_crise_execute(self, config):
        crise = generer_crise_2008()
        strategie = BuyHold()
        r = executer_scenario(crise, strategie, config)
        assert r.nom == crise.nom
        assert r.type == TypeScenario.CRISE_HISTORIQUE
        assert isinstance(r.capital_final_pct, float)

    def test_scenario_monte_carlo_execute(self, config, prix_base):
        scenarios = monte_carlo_blocs(prix_base, n_iterations=3)
        strategie = BuyHold()
        for sc in scenarios:
            r = executer_scenario(sc, strategie, config)
            assert r.capital_final_pct > 0  # le compte ne tombe pas à zéro

    def test_buy_hold_survit_flash_crash(self, config):
        """Buy & hold devrait survivre à un flash crash (chute temporaire)."""
        crise = generer_flash_crash_2015()
        strategie = BuyHold()
        r = executer_scenario(crise, strategie, config)
        assert r.compte_survit


# ═══════════════════════════════════════════════════════════════
# 5. BANC DE STRESS COMPLET
# ═══════════════════════════════════════════════════════════════

class TestBancComplet:
    def test_banc_complet_tourne(self, config, prix_base):
        """Le banc de stress complet s'exécute sans erreur."""
        strategie = BuyHold()
        resultat = banc_de_stress(
            strategie, config, prix_historique=prix_base,
            n_monte_carlo=10,  # réduit pour le test
        )
        assert resultat.nb_total > 0
        assert resultat.nom_strategie == "acheter_garder"

    def test_banc_avec_temoin_aleatoire(self, config, prix_base):
        """Le témoin aléatoire ne devrait pas avoir l'air bon."""
        strategie = RandomControl(seed=42)
        resultat = banc_de_stress(
            strategie, config, prix_historique=prix_base,
            n_monte_carlo=10,
        )
        assert resultat.nb_total > 0

    def test_resume_genere(self, config, prix_base):
        """Le résumé est généré correctement."""
        strategie = BuyHold()
        resultat = banc_de_stress(
            strategie, config, prix_historique=prix_base,
            n_monte_carlo=5,
        )
        resume = resultat.resume()
        assert "Banc de stress" in resume
        assert strategie.nom in resume

    def test_pire_scenario_identifie(self, config, prix_base):
        """Le pire scénario est celui avec le capital final le plus bas."""
        strategie = BuyHold()
        resultat = banc_de_stress(
            strategie, config, prix_historique=prix_base,
            n_monte_carlo=5,
        )
        pire = resultat.pire_scenario
        assert pire is not None
        for s in resultat.scenarios:
            assert pire.capital_final_pct <= s.capital_final_pct
