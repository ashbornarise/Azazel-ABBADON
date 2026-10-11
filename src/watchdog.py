"""
Watchdog — Arrêt d'urgence indépendant
══════════════════════════════════════
Processus SÉPARÉ du robot. Peut tout arrêter même si le robot plante.

Principe GMP : l'arrêt doit rester fonctionnel quand la machine
qui commande est tombée en panne.

Déclenchement :
  - Automatique : drawdown ou perte journalière dépassés
  - Manuel : fichier drapeau (ARRET_URGENCE) ou commande
  - Test régulier recommandé
"""

from __future__ import annotations

import json
import logging
import time
import datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Protocol

import tomllib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(levelname)s — %(message)s",
)
log = logging.getLogger("watchdog")


# ─── Interface pour le compte ─────────────────────────────────

class CompteLecture(Protocol):
    """Interface que le watchdog utilise pour lire l'état du compte."""
    def lire_capital(self) -> float: ...
    def lire_positions_ouvertes(self) -> list[dict]: ...


# ─── Compte simulé (pour les tests) ──────────────────────────

@dataclass
class CompteSimule:
    """Compte simulé pour les tests du watchdog."""
    capital: float = 10_000.0
    positions: list[dict] | None = None

    def lire_capital(self) -> float:
        return self.capital

    def lire_positions_ouvertes(self) -> list[dict]:
        return self.positions or []


# ─── Journal du watchdog ──────────────────────────────────────

@dataclass
class EvenementWatchdog:
    """Un événement enregistré par le watchdog."""
    timestamp: str
    type: str          # "verification", "alerte", "arret_urgence"
    capital: float
    drawdown: float
    perte_journaliere: float
    message: str


class JournalWatchdog:
    """Journal persistant des événements du watchdog."""

    def __init__(self, chemin: str | Path = "data/watchdog_journal.jsonl"):
        self.chemin = Path(chemin)
        self.chemin.parent.mkdir(parents=True, exist_ok=True)

    def ecrire(self, evenement: EvenementWatchdog):
        with open(self.chemin, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(evenement), ensure_ascii=False) + "\n")

    def lire(self) -> list[EvenementWatchdog]:
        if not self.chemin.exists():
            return []
        evenements = []
        with open(self.chemin, "r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    d = json.loads(ligne)
                    evenements.append(EvenementWatchdog(**d))
        return evenements


# ─── Watchdog ─────────────────────────────────────────────────

class Watchdog:
    """
    Processus de surveillance indépendant.
    Vérifie périodiquement l'état du compte et peut déclencher
    un arrêt d'urgence si les seuils sont dépassés.
    """

    def __init__(
        self,
        config: dict,
        capital_initial: float = 10_000.0,
        journal: Optional[JournalWatchdog] = None,
    ):
        surv = config.get("survie", {})
        self.drawdown_max = surv.get("drawdown_max", 0.10)
        self.perte_max_journaliere = surv.get("perte_max_journaliere", 0.03)

        wd = config.get("watchdog", {})
        self.intervalle = wd.get("intervalle_verification", 5)
        self.fichier_arret = Path(wd.get("fichier_arret_urgence", "ARRET_URGENCE"))

        self.capital_initial = capital_initial
        self.capital_max = capital_initial
        self.capital_debut_journee = capital_initial
        self.date_courante: Optional[dt.date] = None
        self.arret_declenche = False

        self.journal = journal or JournalWatchdog()
        self._callbacks_arret: list = []

    def enregistrer_callback_arret(self, callback):
        """Enregistre une fonction appelée lors d'un arrêt d'urgence."""
        self._callbacks_arret.append(callback)

    def _notifier_arret(self, raison: str):
        """Notifie tous les callbacks enregistrés."""
        for cb in self._callbacks_arret:
            try:
                cb(raison)
            except Exception as e:
                log.error(f"Erreur dans callback d'arrêt : {e}")

    def verifier(self, capital: float, date: Optional[dt.date] = None) -> EvenementWatchdog:
        """
        Vérifie l'état du compte. Point d'entrée principal.

        Returns:
            EvenementWatchdog décrivant le résultat de la vérification.
        """
        now = dt.datetime.now().isoformat(timespec="seconds")

        # Nouvelle journée
        if date and date != self.date_courante:
            self.capital_debut_journee = self.capital_max if self.date_courante is None else capital
            self.date_courante = date

        # Mettre à jour le high-water mark
        if capital > self.capital_max:
            self.capital_max = capital

        # Calculer les métriques
        drawdown = max(0.0, 1.0 - capital / self.capital_max) if self.capital_max > 0 else 0.0
        perte_j = max(0.0, 1.0 - capital / self.capital_debut_journee) if self.capital_debut_journee > 0 else 0.0

        # ── Vérifier le fichier drapeau d'arrêt manuel ──
        if self.fichier_arret.exists():
            self.arret_declenche = True
            evt = EvenementWatchdog(
                timestamp=now,
                type="arret_urgence",
                capital=capital,
                drawdown=drawdown,
                perte_journaliere=perte_j,
                message=f"ARRÊT MANUEL : fichier drapeau '{self.fichier_arret}' détecté.",
            )
            log.critical(evt.message)
            self.journal.ecrire(evt)
            self._notifier_arret(evt.message)
            return evt

        # ── Vérifier le drawdown ──
        if drawdown >= self.drawdown_max:
            self.arret_declenche = True
            evt = EvenementWatchdog(
                timestamp=now,
                type="arret_urgence",
                capital=capital,
                drawdown=drawdown,
                perte_journaliere=perte_j,
                message=f"ARRÊT : Drawdown {drawdown:.1%} ≥ seuil {self.drawdown_max:.1%}.",
            )
            log.critical(evt.message)
            self.journal.ecrire(evt)
            self._notifier_arret(evt.message)
            return evt

        # ── Vérifier la perte journalière ──
        if perte_j >= self.perte_max_journaliere:
            evt = EvenementWatchdog(
                timestamp=now,
                type="alerte",
                capital=capital,
                drawdown=drawdown,
                perte_journaliere=perte_j,
                message=f"ALERTE : Perte journalière {perte_j:.1%} ≥ seuil {self.perte_max_journaliere:.1%}.",
            )
            log.warning(evt.message)
            self.journal.ecrire(evt)
            self._notifier_arret(evt.message)
            return evt

        # ── Tout va bien ──
        evt = EvenementWatchdog(
            timestamp=now,
            type="verification",
            capital=capital,
            drawdown=drawdown,
            perte_journaliere=perte_j,
            message=f"OK — capital {capital:.2f}, drawdown {drawdown:.1%}, perte jour {perte_j:.1%}.",
        )
        self.journal.ecrire(evt)
        return evt

    def verifier_arret_urgence_fichier(self) -> bool:
        """Vérifie si le fichier drapeau d'arrêt d'urgence existe."""
        return self.fichier_arret.exists()

    def creer_fichier_arret(self):
        """Crée le fichier drapeau d'arrêt d'urgence (commande manuelle)."""
        self.fichier_arret.write_text(
            f"Arrêt d'urgence déclenché à {dt.datetime.now().isoformat()}\n",
            encoding="utf-8",
        )
        log.critical("Fichier d'arrêt d'urgence créé.")

    def supprimer_fichier_arret(self):
        """Supprime le fichier drapeau (reset par Clément)."""
        if self.fichier_arret.exists():
            self.fichier_arret.unlink()
            self.arret_declenche = False
            log.info("Fichier d'arrêt d'urgence supprimé. Robot peut redémarrer.")


# ─── Point d'entrée (processus séparé) ───────────────────────

def main():
    """Point d'entrée pour exécuter le watchdog comme processus séparé."""
    config_path = Path("config.toml")
    if not config_path.exists():
        log.error("config.toml introuvable.")
        return

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    wd = Watchdog(config)
    log.info(f"Watchdog démarré. Intervalle : {wd.intervalle}s. Fichier arrêt : {wd.fichier_arret}")
    log.info(f"Seuils — Drawdown max : {wd.drawdown_max:.0%}, Perte journalière max : {wd.perte_max_journaliere:.0%}")
    log.info("Pour arrêter manuellement : créer le fichier ARRET_URGENCE")

    # En mode réel, le watchdog lirait le capital via l'API du courtier.
    # Ici, il vérifie juste le fichier drapeau en boucle.
    try:
        while True:
            if wd.verifier_arret_urgence_fichier():
                log.critical("ARRÊT D'URGENCE DÉTECTÉ — fichier drapeau présent.")
                break
            time.sleep(wd.intervalle)
    except KeyboardInterrupt:
        log.info("Watchdog arrêté par l'utilisateur.")


if __name__ == "__main__":
    main()
