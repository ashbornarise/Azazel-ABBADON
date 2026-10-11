"""
Tests du module de survie — Rex Clemens Capital
════════════════════════════════════════════════
Chaque règle de survie est déclenchée volontairement
et sa réaction est vérifiée.

Exécuter : pytest tests/test_survie.py -v
"""

import datetime as dt

import pytest

from src.survie import (
    ModuleSurvie,
    EtatRobot,
    RaisonBlocage,
    DecisionSurvie,
    charger_config,
)


# ─── Fixture : configuration par défaut ──────────────────────

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
    }


@pytest.fixture
def survie(config):
    return ModuleSurvie(config, capital_initial=10_000.0)


# ═══════════════════════════════════════════════════════════════
# RÈGLE 1 : Risque maximal par trade (1 % du compte)
# ═══════════════════════════════════════════════════════════════

class TestRisqueParTrade:
    def test_exposition_reduite_quand_risque_trop_eleve(self, survie):
        """Si le stop-loss est large, l'exposition doit être réduite."""
        # Stop-loss à 5% → risque max = 1% / 5% = 0.2 d'exposition max
        decision = survie.evaluer(
            exposition_demandee=1.0,
            stop_loss_distance=0.05,
        )
        assert decision.exposition_autorisee <= 0.20 + 1e-6
        assert RaisonBlocage.RISQUE_PAR_TRADE in decision.raisons

    def test_exposition_ok_quand_risque_acceptable(self, survie):
        """Si le stop-loss est serré, pas de réduction pour risque."""
        # Stop-loss à 0.5% → risque max = 1% / 0.5% = 2.0 → clampé à 0.25 par taille max
        decision = survie.evaluer(
            exposition_demandee=0.20,
            stop_loss_distance=0.005,
        )
        assert RaisonBlocage.RISQUE_PAR_TRADE not in decision.raisons

    def test_sans_stop_loss_pas_de_verification_risque(self, survie):
        """Sans stop-loss, la règle de risque par trade ne s'applique pas."""
        decision = survie.evaluer(exposition_demandee=0.20)
        assert RaisonBlocage.RISQUE_PAR_TRADE not in decision.raisons


# ═══════════════════════════════════════════════════════════════
# RÈGLE 2 : Levier interdit (exposition max 1.0)
# ═══════════════════════════════════════════════════════════════

class TestLevierInterdit:
    def test_levier_refuse(self, survie):
        """Une exposition > 1.0 est ramenée à 1.0 max."""
        decision = survie.evaluer(exposition_demandee=2.5)
        assert decision.exposition_autorisee <= 1.0
        assert RaisonBlocage.LEVIER_INTERDIT in decision.raisons

    def test_exposition_negative_refusee(self, survie):
        """Une exposition négative (short) est ramenée à 0."""
        decision = survie.evaluer(exposition_demandee=-0.5)
        assert decision.exposition_autorisee >= 0.0

    def test_exposition_1_acceptee(self, survie):
        """Exposition à 1.0 = pas de levier, acceptée."""
        # Sera réduite à 0.25 par la taille max de position, mais pas par le levier
        decision = survie.evaluer(exposition_demandee=1.0)
        assert RaisonBlocage.LEVIER_INTERDIT not in decision.raisons


# ═══════════════════════════════════════════════════════════════
# RÈGLE 3 : Perte maximale sur une journée (3 % → arrêt)
# ═══════════════════════════════════════════════════════════════

class TestPerteJournaliere:
    def test_arret_si_perte_journaliere_depassee(self, survie):
        """Perte de 3% dans la journée → arrêt jusqu'au lendemain."""
        jour1 = dt.date(2024, 1, 15)
        survie.mettre_a_jour_capital(10_000, jour1)

        # Perte de 3.5% dans la journée
        survie.mettre_a_jour_capital(9_650, jour1)
        decision = survie.evaluer(exposition_demandee=0.5, date=jour1)

        assert decision.etat == EtatRobot.ARRET_JOURNEE
        assert decision.exposition_autorisee == 0.0
        assert RaisonBlocage.PERTE_JOURNALIERE in decision.raisons

    def test_reprise_le_lendemain(self, survie):
        """Le lendemain, le robot reprend."""
        jour1 = dt.date(2024, 1, 15)
        jour2 = dt.date(2024, 1, 16)

        survie.mettre_a_jour_capital(10_000, jour1)
        survie.mettre_a_jour_capital(9_650, jour1)  # -3.5%
        survie.evaluer(exposition_demandee=0.5, date=jour1)  # déclenche l'arrêt

        # Le lendemain
        survie.mettre_a_jour_capital(9_650, jour2)
        decision = survie.evaluer(exposition_demandee=0.2, date=jour2)

        assert decision.etat != EtatRobot.ARRET_JOURNEE
        assert decision.exposition_autorisee > 0.0

    def test_perte_2pct_pas_arret(self, survie):
        """Perte de 2% = sous le seuil, pas d'arrêt."""
        jour = dt.date(2024, 1, 15)
        survie.mettre_a_jour_capital(10_000, jour)
        survie.mettre_a_jour_capital(9_800, jour)  # -2%

        decision = survie.evaluer(exposition_demandee=0.2, date=jour)
        assert decision.etat != EtatRobot.ARRET_JOURNEE


# ═══════════════════════════════════════════════════════════════
# RÈGLE 4 : Drawdown max (10 % → arrêt total)
# ═══════════════════════════════════════════════════════════════

class TestDrawdownMax:
    def test_arret_total_si_drawdown_depasse(self, survie):
        """Drawdown de 10% depuis le plus haut → arrêt total."""
        survie.mettre_a_jour_capital(10_000)
        survie.mettre_a_jour_capital(8_900)  # -11% drawdown

        decision = survie.evaluer(exposition_demandee=0.5)
        assert decision.etat == EtatRobot.ARRET_TOTAL
        assert decision.exposition_autorisee == 0.0
        assert RaisonBlocage.DRAWDOWN_MAX in decision.raisons

    def test_arret_total_persiste(self, survie):
        """L'arrêt total persiste même si le capital remonte."""
        survie.mettre_a_jour_capital(10_000)
        survie.mettre_a_jour_capital(8_900)  # déclenche
        survie.evaluer(exposition_demandee=0.5)

        # Capital remonte mais l'arrêt persiste
        survie.mettre_a_jour_capital(9_500)
        decision = survie.evaluer(exposition_demandee=0.5)
        assert decision.etat == EtatRobot.ARRET_TOTAL

    def test_arret_total_reset_par_clement(self, survie):
        """Seul un reset explicite (par Clément) relance le robot."""
        survie.mettre_a_jour_capital(10_000)
        survie.mettre_a_jour_capital(8_900)
        survie.evaluer(exposition_demandee=0.5)

        survie.reinitialiser(nouveau_capital=5_000)
        decision = survie.evaluer(exposition_demandee=0.2)

        assert decision.etat == EtatRobot.ACTIF
        assert decision.exposition_autorisee > 0.0

    def test_drawdown_5pct_pas_arret(self, survie):
        """Drawdown de 5% = sous le seuil, pas d'arrêt."""
        survie.mettre_a_jour_capital(10_000)
        survie.mettre_a_jour_capital(9_500)  # -5%

        decision = survie.evaluer(exposition_demandee=0.2)
        assert decision.etat != EtatRobot.ARRET_TOTAL


# ═══════════════════════════════════════════════════════════════
# RÈGLE 5 : Taille maximale d'une position (25 % du compte)
# ═══════════════════════════════════════════════════════════════

class TestTaillePosition:
    def test_position_trop_grande_reduite(self, survie):
        """Exposition de 50% réduite à 25%."""
        decision = survie.evaluer(exposition_demandee=0.50)
        assert decision.exposition_autorisee <= 0.25
        assert RaisonBlocage.TAILLE_POSITION in decision.raisons

    def test_position_25pct_ok(self, survie):
        """Exposition de 25% = exactement le seuil, pas de réduction."""
        decision = survie.evaluer(exposition_demandee=0.25)
        assert decision.exposition_autorisee == 0.25
        assert RaisonBlocage.TAILLE_POSITION not in decision.raisons

    def test_position_10pct_ok(self, survie):
        """Exposition de 10% = sous le seuil, pas de réduction."""
        decision = survie.evaluer(exposition_demandee=0.10)
        assert decision.exposition_autorisee == 0.10
        assert RaisonBlocage.TAILLE_POSITION not in decision.raisons


# ═══════════════════════════════════════════════════════════════
# RÈGLE 6 : Données invalides → état liquide
# ═══════════════════════════════════════════════════════════════

class TestDonneesInvalides:
    def test_prix_zero_bloque(self, survie):
        """Prix à 0 → état liquide, aucun trade."""
        decision = survie.evaluer(exposition_demandee=0.2, prix=0.0)
        assert decision.etat == EtatRobot.LIQUIDE
        assert decision.exposition_autorisee == 0.0
        assert RaisonBlocage.DONNEES_INVALIDES in decision.raisons

    def test_prix_negatif_bloque(self, survie):
        """Prix négatif → état liquide."""
        decision = survie.evaluer(exposition_demandee=0.2, prix=-10.0)
        assert decision.etat == EtatRobot.LIQUIDE
        assert decision.exposition_autorisee == 0.0

    def test_saut_impossible_bloque(self, survie):
        """Saut de prix > 50% en un tick → données suspectes → liquide."""
        decision = survie.evaluer(
            exposition_demandee=0.2,
            prix=200.0,
            prix_precedent=100.0,  # +100% = suspect
        )
        assert decision.etat == EtatRobot.LIQUIDE
        assert decision.exposition_autorisee == 0.0

    def test_donnees_perimees_bloque(self, survie):
        """Données de plus de 30 minutes → liquide."""
        maintenant = dt.datetime.now(tz=dt.timezone.utc)
        il_y_a_1h = maintenant - dt.timedelta(hours=1)

        decision = survie.evaluer(
            exposition_demandee=0.2,
            prix=100.0,
            horodatage=il_y_a_1h,
        )
        assert decision.etat == EtatRobot.LIQUIDE
        assert decision.exposition_autorisee == 0.0

    def test_prix_normal_ok(self, survie):
        """Prix normal, pas de blocage."""
        decision = survie.evaluer(
            exposition_demandee=0.2,
            prix=100.0,
            prix_precedent=99.0,
        )
        assert RaisonBlocage.DONNEES_INVALIDES not in decision.raisons


# ═══════════════════════════════════════════════════════════════
# RÈGLE 7 : Connexion perdue → état liquide
# ═══════════════════════════════════════════════════════════════

class TestConnexionPerdue:
    def test_erreurs_consecutives_bloquent(self, survie):
        """3 erreurs consécutives → état liquide."""
        survie.signaler_erreur_connexion()
        survie.signaler_erreur_connexion()
        survie.signaler_erreur_connexion()

        decision = survie.evaluer(exposition_demandee=0.2)
        assert decision.etat == EtatRobot.LIQUIDE
        assert decision.exposition_autorisee == 0.0
        assert RaisonBlocage.CONNEXION_PERDUE in decision.raisons

    def test_connexion_retablie_debloque(self, survie):
        """Après retour de la connexion, le trading reprend."""
        survie.signaler_erreur_connexion()
        survie.signaler_erreur_connexion()
        survie.signaler_erreur_connexion()
        survie.evaluer(exposition_demandee=0.2)  # déclenche LIQUIDE

        survie.signaler_connexion_ok()
        # Remettre l'état actif (le module revient à ACTIF quand pas de raison de blocage)
        survie.etat = EtatRobot.ACTIF
        decision = survie.evaluer(exposition_demandee=0.2)
        assert decision.exposition_autorisee > 0.0

    def test_2_erreurs_pas_bloquant(self, survie):
        """2 erreurs < seuil de 3 → pas de blocage."""
        survie.signaler_erreur_connexion()
        survie.signaler_erreur_connexion()

        decision = survie.evaluer(exposition_demandee=0.2)
        assert RaisonBlocage.CONNEXION_PERDUE not in decision.raisons


# ═══════════════════════════════════════════════════════════════
# ARRÊT D'URGENCE
# ═══════════════════════════════════════════════════════════════

class TestArretUrgence:
    def test_arret_urgence_bloque_tout(self, survie):
        """L'arrêt d'urgence bloque toute exposition."""
        survie.declencher_arret_urgence()
        decision = survie.evaluer(exposition_demandee=0.1)

        assert decision.etat == EtatRobot.ARRET_TOTAL
        assert decision.exposition_autorisee == 0.0
        assert RaisonBlocage.ARRET_URGENCE in decision.raisons

    def test_arret_urgence_prioritaire(self, survie):
        """L'arrêt d'urgence est prioritaire sur tout le reste."""
        survie.declencher_arret_urgence()
        # Même avec capital parfait
        survie.mettre_a_jour_capital(100_000)
        decision = survie.evaluer(exposition_demandee=0.1)
        assert decision.exposition_autorisee == 0.0


# ═══════════════════════════════════════════════════════════════
# COMBINAISONS DE RÈGLES
# ═══════════════════════════════════════════════════════════════

class TestCombinaisons:
    def test_multiple_reductions(self, survie):
        """Levier + taille → l'exposition est réduite par les deux règles."""
        decision = survie.evaluer(exposition_demandee=3.0)
        # Réduit d'abord par levier (→1.0), puis par taille (→0.25)
        assert decision.exposition_autorisee <= 0.25
        assert RaisonBlocage.LEVIER_INTERDIT in decision.raisons
        assert RaisonBlocage.TAILLE_POSITION in decision.raisons

    def test_demande_zero_toujours_ok(self, survie):
        """Demander 0 = cash, toujours accepté."""
        decision = survie.evaluer(exposition_demandee=0.0)
        assert decision.exposition_autorisee == 0.0
        assert not decision.bloque  # pas bloqué car on ne voulait rien

    def test_strategie_ne_contourne_pas_survie(self, survie):
        """Quoi que demande la stratégie, survie a le dernier mot."""
        # Drawdown dépassé → arrêt total, quelle que soit la demande
        survie.mettre_a_jour_capital(10_000)
        survie.mettre_a_jour_capital(8_500)  # -15%
        survie.evaluer(exposition_demandee=0.1)  # déclenche arrêt

        decision = survie.evaluer(exposition_demandee=0.01)
        assert decision.exposition_autorisee == 0.0
        assert decision.etat == EtatRobot.ARRET_TOTAL
