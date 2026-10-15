"""
Alpha Scorer — Combine tous les signaux en un score final
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidat:
    symbole: str
    prix: float
    score_stagnation: float = 0
    score_qualite: float = 0
    score_reversal: float = 0
    score_alpha: float = 0
    decision: str = "EN_OBSERVATION"
    raison: str = ""
    details: dict | None = None


class AlphaScorer:
    """
    Combine les scores de stagnation, qualité fondamentale et retournement
    en un score alpha final. Classe les candidats.
    """

    POIDS_STAGNATION = 0.30
    POIDS_QUALITE = 0.40
    POIDS_REVERSAL = 0.30

    SEUIL_CANDIDAT = 40
    SEUIL_ACHAT = 55

    def scorer(self, symbole: str, prix: float,
               stagnation: dict, qualite: dict, reversal: dict) -> Candidat:
        s_stag = stagnation.get("score_stagnation", 0)
        s_qual = qualite.get("score_qualite", 0)
        s_rev = reversal.get("reversal_score", 0)

        score = (
            s_stag * self.POIDS_STAGNATION
            + s_qual * self.POIDS_QUALITE
            + s_rev * self.POIDS_REVERSAL
        )

        if score >= self.SEUIL_ACHAT:
            decision = "ACHAT_RECOMMANDE"
            raison = "Score alpha élevé : stagnation + qualité + signes de reprise"
        elif score >= self.SEUIL_CANDIDAT:
            decision = "CANDIDAT"
            raison = "Potentiel intéressant, surveillance recommandée"
        else:
            decision = "EN_OBSERVATION"
            raison = "Score insuffisant pour action immédiate"

        return Candidat(
            symbole=symbole,
            prix=prix,
            score_stagnation=round(s_stag, 1),
            score_qualite=round(s_qual, 1),
            score_reversal=round(s_rev, 1),
            score_alpha=round(score, 1),
            decision=decision,
            raison=raison,
            details={
                "stagnation": stagnation,
                "qualite": qualite,
                "reversal": reversal,
            },
        )

    def classer(self, candidats: list[Candidat]) -> list[Candidat]:
        return sorted(candidats, key=lambda c: c.score_alpha, reverse=True)
