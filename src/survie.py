"""
Module de survie — Rex Clemens Capital
═══════════════════════════════════════
Gardien du compte. Indépendant de toute stratégie.
Une stratégie propose une exposition cible ; ce module la réduit,
la refuse ou impose zéro. Il ne peut pas être contourné.

Principe : la survie du compte passe avant le gain.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import tomllib
from pathlib import Path


# ─── Types ────────────────────────────────────────────────────

class EtatRobot(Enum):
    """État opérationnel du robot."""
    ACTIF = auto()          # trading normal
    LIQUIDE = auto()        # aucune position, aucun nouveau trade
    ARRET_JOURNEE = auto()  # perte journalière atteinte, arrêt jusqu'au lendemain
    ARRET_TOTAL = auto()    # drawdown max atteint, arrêt définitif (reset par Clément)


class RaisonBlocage(Enum):
    """Pourquoi le module de survie a bloqué ou réduit une opération."""
    AUCUNE = auto()
    RISQUE_PAR_TRADE = auto()
    LEVIER_INTERDIT = auto()
    PERTE_JOURNALIERE = auto()
    DRAWDOWN_MAX = auto()
    TAILLE_POSITION = auto()
    DONNEES_INVALIDES = auto()
    CONNEXION_PERDUE = auto()
    ARRET_URGENCE = auto()


@dataclass
class DecisionSurvie:
    """Résultat de l'évaluation par le module de survie."""
    exposition_autorisee: float    # entre 0.0 et 1.0
    exposition_demandee: float     # ce que la stratégie voulait
    etat: EtatRobot
    raisons: list[RaisonBlocage] = field(default_factory=list)
    details: str = ""

    @property
    def bloque(self) -> bool:
        return self.exposition_autorisee == 0.0 and self.exposition_demandee > 0.0

    @property
    def reduit(self) -> bool:
        return 0.0 < self.exposition_autorisee < self.exposition_demandee


# ─── Configuration ────────────────────────────────────────────

def charger_config(chemin: str | Path = "config.toml") -> dict:
    """Charge la configuration depuis le fichier TOML."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {chemin}")
    with open(chemin, "rb") as f:
        return tomllib.load(f)


# ─── Module de Survie ─────────────────────────────────────────

class ModuleSurvie:
    """
    Gardien du compte. Évalue chaque demande d'exposition et applique
    les règles de survie. Ne peut pas être contourné par une stratégie.
    """

    def __init__(self, config: dict, capital_initial: float = 10_000.0):
        surv = config.get("survie", {})
        self.risque_max_par_trade = surv.get("risque_max_par_trade", 0.01)
        self.exposition_max = surv.get("exposition_max", 1.0)
        self.perte_max_journaliere = surv.get("perte_max_journaliere", 0.03)
        self.drawdown_max = surv.get("drawdown_max", 0.10)
        self.taille_max_position = surv.get("taille_max_position", 0.25)

        donnees = config.get("donnees", {})
        self.age_max_donnees_min = donnees.get("age_max_donnees_minutes", 30)
        self.prix_saut_max_pct = donnees.get("prix_saut_max_pct", 0.50)
        self.prix_min = donnees.get("prix_min", 0.0001)

        conn = config.get("connexion", {})
        self.erreurs_max = conn.get("erreurs_max_consecutives", 3)

        # État du compte
        self.capital_initial = capital_initial
        self.capital = capital_initial
        self.capital_max = capital_initial  # plus haut historique (high-water mark)
        self.capital_debut_journee = capital_initial
        self.date_courante: Optional[dt.date] = None

        # État du robot
        self.etat = EtatRobot.ACTIF
        self.erreurs_consecutives = 0
        self.arret_urgence = False

    # ─── Mise à jour du capital ───────────────────────────────

    def mettre_a_jour_capital(self, nouveau_capital: float, date: Optional[dt.date] = None):
        """Met à jour le capital et vérifie les seuils de survie."""
        if date and date != self.date_courante:
            # Nouvelle journée : réinitialiser le capital de début de journée
            self.capital_debut_journee = self.capital
            self.date_courante = date
            # Sortir de l'arrêt journalier si on était en ARRET_JOURNEE
            if self.etat == EtatRobot.ARRET_JOURNEE:
                self.etat = EtatRobot.ACTIF

        self.capital = nouveau_capital
        if nouveau_capital > self.capital_max:
            self.capital_max = nouveau_capital

    def _perte_journaliere(self) -> float:
        """Perte depuis le début de la journée en fraction."""
        if self.capital_debut_journee <= 0:
            return 0.0
        return max(0.0, 1.0 - self.capital / self.capital_debut_journee)

    def _drawdown(self) -> float:
        """Drawdown depuis le plus haut historique en fraction."""
        if self.capital_max <= 0:
            return 0.0
        return max(0.0, 1.0 - self.capital / self.capital_max)

    # ─── Vérification des données ─────────────────────────────

    def verifier_donnees(
        self,
        prix: float,
        prix_precedent: Optional[float] = None,
        horodatage: Optional[dt.datetime] = None,
    ) -> list[RaisonBlocage]:
        """Vérifie la qualité des données. Retourne les raisons de blocage."""
        raisons = []

        # Prix nul, négatif ou quasi-nul
        if prix <= self.prix_min:
            raisons.append(RaisonBlocage.DONNEES_INVALIDES)

        # Saut de prix impossible
        if prix_precedent is not None and prix_precedent > self.prix_min:
            saut = abs(prix - prix_precedent) / prix_precedent
            if saut > self.prix_saut_max_pct:
                raisons.append(RaisonBlocage.DONNEES_INVALIDES)

        # Données périmées
        if horodatage is not None:
            age = (dt.datetime.now(tz=horodatage.tzinfo) - horodatage).total_seconds() / 60
            if age > self.age_max_donnees_min:
                raisons.append(RaisonBlocage.DONNEES_INVALIDES)

        return raisons

    # ─── Signalement d'erreurs de connexion ───────────────────

    def signaler_erreur_connexion(self):
        """Incrémente le compteur d'erreurs consécutives."""
        self.erreurs_consecutives += 1

    def signaler_connexion_ok(self):
        """Réinitialise le compteur d'erreurs."""
        self.erreurs_consecutives = 0

    # ─── Arrêt d'urgence ──────────────────────────────────────

    def declencher_arret_urgence(self):
        """Déclenche l'arrêt d'urgence (fichier drapeau ou commande manuelle)."""
        self.arret_urgence = True
        self.etat = EtatRobot.ARRET_TOTAL

    # ─── Remise à zéro (uniquement par Clément) ───────────────

    def reinitialiser(self, nouveau_capital: Optional[float] = None):
        """
        Réinitialise le robot après un arrêt total.
        Appel réservé à Clément — ne peut pas être automatisé.
        """
        self.etat = EtatRobot.ACTIF
        self.arret_urgence = False
        self.erreurs_consecutives = 0
        if nouveau_capital is not None:
            self.capital = nouveau_capital
            self.capital_max = nouveau_capital
            self.capital_debut_journee = nouveau_capital

    # ─── Évaluation principale ────────────────────────────────

    def evaluer(
        self,
        exposition_demandee: float,
        prix: Optional[float] = None,
        prix_precedent: Optional[float] = None,
        horodatage: Optional[dt.datetime] = None,
        date: Optional[dt.date] = None,
        stop_loss_distance: Optional[float] = None,
    ) -> DecisionSurvie:
        """
        Évalue une demande d'exposition et applique toutes les règles de survie.

        Args:
            exposition_demandee: entre 0.0 (cash) et 1.0 (tout investi).
                                 Valeurs > 1.0 = levier, refusé.
            prix: prix actuel de l'actif (pour vérification qualité).
            prix_precedent: prix précédent (pour détecter les sauts).
            horodatage: heure du dernier prix (pour détecter données périmées).
            date: date courante (pour gestion perte journalière).
            stop_loss_distance: distance du stop-loss en fraction du prix,
                                pour calculer le risque par trade.

        Returns:
            DecisionSurvie avec l'exposition autorisée et les raisons.
        """
        raisons: list[RaisonBlocage] = []
        expo = max(0.0, exposition_demandee)  # jamais négatif (pas de short)

        # ── Arrêt d'urgence ──
        if self.arret_urgence:
            return DecisionSurvie(
                exposition_autorisee=0.0,
                exposition_demandee=exposition_demandee,
                etat=EtatRobot.ARRET_TOTAL,
                raisons=[RaisonBlocage.ARRET_URGENCE],
                details="Arrêt d'urgence déclenché.",
            )

        # ── Arrêt total (drawdown max) ──
        if self.etat == EtatRobot.ARRET_TOTAL:
            return DecisionSurvie(
                exposition_autorisee=0.0,
                exposition_demandee=exposition_demandee,
                etat=EtatRobot.ARRET_TOTAL,
                raisons=[RaisonBlocage.DRAWDOWN_MAX],
                details="Arrêt total : drawdown max dépassé. Reset par Clément requis.",
            )

        # ── Mise à jour du capital si date fournie ──
        if date is not None:
            self.mettre_a_jour_capital(self.capital, date)

        # ── Vérifier le drawdown ──
        dd = self._drawdown()
        if dd >= self.drawdown_max:
            self.etat = EtatRobot.ARRET_TOTAL
            raisons.append(RaisonBlocage.DRAWDOWN_MAX)
            return DecisionSurvie(
                exposition_autorisee=0.0,
                exposition_demandee=exposition_demandee,
                etat=EtatRobot.ARRET_TOTAL,
                raisons=raisons,
                details=f"Drawdown {dd:.1%} ≥ seuil {self.drawdown_max:.1%}. Arrêt total.",
            )

        # ── Vérifier la perte journalière ──
        pj = self._perte_journaliere()
        if pj >= self.perte_max_journaliere:
            self.etat = EtatRobot.ARRET_JOURNEE
            raisons.append(RaisonBlocage.PERTE_JOURNALIERE)
            return DecisionSurvie(
                exposition_autorisee=0.0,
                exposition_demandee=exposition_demandee,
                etat=EtatRobot.ARRET_JOURNEE,
                raisons=raisons,
                details=f"Perte journalière {pj:.1%} ≥ seuil {self.perte_max_journaliere:.1%}. Arrêt jusqu'à demain.",
            )

        # ── Arrêt journée encore actif ──
        if self.etat == EtatRobot.ARRET_JOURNEE:
            raisons.append(RaisonBlocage.PERTE_JOURNALIERE)
            return DecisionSurvie(
                exposition_autorisee=0.0,
                exposition_demandee=exposition_demandee,
                etat=EtatRobot.ARRET_JOURNEE,
                raisons=raisons,
                details="Arrêt journalier encore actif.",
            )

        # ── Vérification des données ──
        if prix is not None:
            raisons_donnees = self.verifier_donnees(prix, prix_precedent, horodatage)
            if raisons_donnees:
                self.etat = EtatRobot.LIQUIDE
                return DecisionSurvie(
                    exposition_autorisee=0.0,
                    exposition_demandee=exposition_demandee,
                    etat=EtatRobot.LIQUIDE,
                    raisons=raisons_donnees,
                    details="Données invalides : passage à l'état liquide.",
                )

        # ── Vérification de la connexion ──
        if self.erreurs_consecutives >= self.erreurs_max:
            raisons.append(RaisonBlocage.CONNEXION_PERDUE)
            self.etat = EtatRobot.LIQUIDE
            return DecisionSurvie(
                exposition_autorisee=0.0,
                exposition_demandee=exposition_demandee,
                etat=EtatRobot.LIQUIDE,
                raisons=raisons,
                details=f"Connexion perdue ({self.erreurs_consecutives} erreurs consécutives).",
            )

        # ── Réductions (on ne bloque pas, on réduit) ──

        # Levier interdit
        if expo > self.exposition_max:
            expo = self.exposition_max
            raisons.append(RaisonBlocage.LEVIER_INTERDIT)

        # Taille maximale de position
        if expo > self.taille_max_position:
            expo = self.taille_max_position
            raisons.append(RaisonBlocage.TAILLE_POSITION)

        # Risque par trade (si un stop-loss est fourni)
        if stop_loss_distance is not None and stop_loss_distance > 0 and self.capital > 0:
            # risque_max = risque_max_par_trade * capital
            # exposition * stop_loss_distance * capital = risque
            # → exposition_max_risque = risque_max_par_trade / stop_loss_distance
            expo_risque = self.risque_max_par_trade / stop_loss_distance
            if expo > expo_risque:
                expo = expo_risque
                raisons.append(RaisonBlocage.RISQUE_PAR_TRADE)

        # Si on était en état liquide mais les données sont OK maintenant, revenir actif
        if self.etat == EtatRobot.LIQUIDE and not raisons:
            self.etat = EtatRobot.ACTIF

        return DecisionSurvie(
            exposition_autorisee=round(expo, 6),
            exposition_demandee=exposition_demandee,
            etat=self.etat if self.etat != EtatRobot.LIQUIDE else EtatRobot.ACTIF,
            raisons=raisons,
            details="" if not raisons else "Exposition réduite par les règles de survie.",
        )
