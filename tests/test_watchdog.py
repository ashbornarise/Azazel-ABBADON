"""
Tests du watchdog — Rex Clemens Capital
═══════════════════════════════════════
Vérifie que le watchdog détecte les situations dangereuses
et déclenche les arrêts même indépendamment du robot.

Exécuter : pytest tests/test_watchdog.py -v
"""

import datetime as dt
from pathlib import Path

import pytest

from src.watchdog import Watchdog, JournalWatchdog, CompteSimule


@pytest.fixture
def config():
    return {
        "survie": {
            "drawdown_max": 0.10,
            "perte_max_journaliere": 0.03,
        },
        "watchdog": {
            "intervalle_verification": 1,
            "fichier_arret_urgence": "TEST_ARRET_URGENCE",
        },
    }


@pytest.fixture
def journal(tmp_path):
    return JournalWatchdog(tmp_path / "test_journal.jsonl")


@pytest.fixture
def watchdog(config, journal):
    wd = Watchdog(config, capital_initial=10_000.0, journal=journal)
    # S'assurer que le fichier drapeau de test n'existe pas
    if wd.fichier_arret.exists():
        wd.fichier_arret.unlink()
    yield wd
    # Nettoyage
    if wd.fichier_arret.exists():
        wd.fichier_arret.unlink()


# ═══════════════════════════════════════════════════════════════
# DÉTECTION DE DRAWDOWN
# ═══════════════════════════════════════════════════════════════

class TestDrawdownWatchdog:
    def test_arret_si_drawdown_depasse(self, watchdog):
        """Drawdown ≥ 10% → arrêt d'urgence."""
        evt = watchdog.verifier(8_900)  # -11%
        assert evt.type == "arret_urgence"
        assert watchdog.arret_declenche

    def test_pas_arret_si_drawdown_sous_seuil(self, watchdog):
        """Drawdown de 5% → pas d'arrêt d'urgence (mais alerte perte journalière possible)."""
        # Pour isoler le drawdown, on simule un nouveau jour avec capital déjà abaissé
        jour = dt.date(2024, 1, 15)
        watchdog.verifier(9_500, date=jour)  # fixe le capital de début de journée
        jour2 = dt.date(2024, 1, 16)
        evt = watchdog.verifier(9_500, date=jour2)  # 0% perte journalière, 5% drawdown
        assert evt.type == "verification"
        assert not watchdog.arret_declenche

    def test_drawdown_calcule_depuis_plus_haut(self, watchdog):
        """Le drawdown se calcule depuis le plus haut, pas le capital initial."""
        watchdog.verifier(12_000)  # nouveau plus haut
        evt = watchdog.verifier(10_700)  # -10.8% depuis 12_000
        assert evt.type == "arret_urgence"


# ═══════════════════════════════════════════════════════════════
# DÉTECTION DE PERTE JOURNALIÈRE
# ═══════════════════════════════════════════════════════════════

class TestPerteJournaliereWatchdog:
    def test_alerte_si_perte_journaliere(self, watchdog):
        """Perte journalière ≥ 3% → alerte."""
        jour = dt.date(2024, 1, 15)
        watchdog.verifier(10_000, date=jour)
        evt = watchdog.verifier(9_650, date=jour)  # -3.5%
        assert evt.type == "alerte"

    def test_perte_journaliere_remise_a_zero_le_lendemain(self, watchdog):
        """La perte journalière est recalculée chaque jour."""
        jour1 = dt.date(2024, 1, 15)
        jour2 = dt.date(2024, 1, 16)

        watchdog.verifier(10_000, date=jour1)
        watchdog.verifier(9_700, date=jour1)  # -3% → alerte

        evt = watchdog.verifier(9_700, date=jour2)  # nouveau jour, pas de perte
        assert evt.type == "verification"


# ═══════════════════════════════════════════════════════════════
# FICHIER DRAPEAU (ARRÊT MANUEL)
# ═══════════════════════════════════════════════════════════════

class TestFichierDrapeau:
    def test_arret_si_fichier_present(self, watchdog):
        """Si le fichier drapeau existe → arrêt d'urgence."""
        watchdog.creer_fichier_arret()
        evt = watchdog.verifier(10_000)
        assert evt.type == "arret_urgence"
        assert "ARRÊT MANUEL" in evt.message

    def test_pas_arret_si_pas_de_fichier(self, watchdog):
        """Sans fichier drapeau → pas d'arrêt."""
        evt = watchdog.verifier(10_000)
        assert evt.type != "arret_urgence"

    def test_creer_et_supprimer_fichier(self, watchdog):
        """On peut créer et supprimer le fichier drapeau."""
        watchdog.creer_fichier_arret()
        assert watchdog.fichier_arret.exists()

        watchdog.supprimer_fichier_arret()
        assert not watchdog.fichier_arret.exists()
        assert not watchdog.arret_declenche


# ═══════════════════════════════════════════════════════════════
# CALLBACKS D'ARRÊT
# ═══════════════════════════════════════════════════════════════

class TestCallbacks:
    def test_callback_appele_sur_arret(self, watchdog):
        """Les callbacks sont appelés lors d'un arrêt."""
        appels = []
        watchdog.enregistrer_callback_arret(lambda r: appels.append(r))

        watchdog.verifier(8_900)  # drawdown → arrêt
        assert len(appels) == 1
        assert "Drawdown" in appels[0]

    def test_callback_erreur_ne_casse_pas(self, watchdog):
        """Un callback qui plante ne casse pas le watchdog."""
        def callback_cassé(r):
            raise ValueError("boom")

        watchdog.enregistrer_callback_arret(callback_cassé)
        # Ne doit pas lever d'exception
        watchdog.verifier(8_900)


# ═══════════════════════════════════════════════════════════════
# JOURNAL
# ═══════════════════════════════════════════════════════════════

class TestJournal:
    def test_evenements_persistes(self, watchdog, journal):
        """Les événements sont écrits dans le journal."""
        watchdog.verifier(10_000)
        watchdog.verifier(9_500)

        evenements = journal.lire()
        assert len(evenements) == 2

    def test_journal_vide_au_depart(self, journal):
        """Le journal est vide au départ."""
        evenements = journal.lire()
        assert len(evenements) == 0
