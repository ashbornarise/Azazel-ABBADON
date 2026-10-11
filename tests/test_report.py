"""
Tests du module de rapport.

Exécuter : pytest tests/test_report.py -v
"""

import pytest

from src.backtest import backtest
from src.data import generer_marche_aleatoire
from src.report import rapport_backtest, rapport_stress
from src.strategies.buy_hold import BuyHold
from src.strategies.random_control import RandomControl
from src.stress import banc_de_stress


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
        "connexion": {"erreurs_max_consecutives": 3},
        "backtest": {"cout_par_trade": 0.001},
        "stress": {
            "seuil_survie_pct_scenarios": 0.95,
            "monte_carlo_iterations": 5,
            "taille_bloc_bootstrap": 20,
        },
    }


@pytest.fixture
def prix():
    return generer_marche_aleatoire(periodes=500, seed=1)


class TestRapportBacktest:
    def test_rapport_genere(self, prix):
        resultats = [
            backtest(prix, BuyHold()),
            backtest(prix, RandomControl()),
        ]
        rapport = rapport_backtest(resultats)
        assert "Rapport de backtest" in rapport
        assert "acheter_garder" in rapport
        assert "temoin_aleatoire" in rapport

    def test_avertissements_presents(self, prix):
        resultats = [backtest(prix, BuyHold())]
        rapport = rapport_backtest(resultats)
        assert "LIMITES" in rapport
        assert "PASSÉES" in rapport or "passé" in rapport.lower()
        assert "SURAJUSTEMENT" in rapport


class TestRapportStress:
    def test_rapport_stress_genere(self, config, prix):
        resultat = banc_de_stress(
            BuyHold(), config, prix_historique=prix, n_monte_carlo=5,
        )
        rapport = rapport_stress(resultat)
        assert "Banc de stress" in rapport
        assert "VERDICT" in rapport
        assert "LIMITES" in rapport

    def test_rapport_contient_crises(self, config, prix):
        resultat = banc_de_stress(
            BuyHold(), config, prix_historique=prix, n_monte_carlo=3,
        )
        rapport = rapport_stress(resultat)
        assert "Crises historiques" in rapport
        assert "Monte Carlo" in rapport
        assert "Chocs synthétiques" in rapport
