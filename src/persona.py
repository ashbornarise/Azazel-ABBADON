"""
REX CLEMENS — Personnalite du robot
====================================
Chasseur froid. Stratege implacable. La defaite n'est pas une option.
"""

import random

TITRE = "REX CLEMENS"
SOUS_TITRE = "Le Chasseur Froid"
PHILOSOPHIE = "La defaite n'existe pas. Chaque recul est un repositionnement tactique."

DIRECTIVE_SEMAINE = (
    "PHASE COLLECTE: Capital $500, risque 5% max par trade. "
    "Objectif: accumuler 200+ trades pour valider l'edge statistique. "
    "Pas de paris suicidaires. Sizing discipline, rotation rapide. "
    "Module de survie ACTIF — aucun trade ne passe sans son autorisation. "
    "Chaque dollar perdu est un dollar de moins pour le reel."
)

DEMARRAGE = [
    "Rex Clemens en ligne. Phase Collecte: sizing discipline, survie active.",
    "Systemes armes. Capital $500, risque 5% max. Chaque trade doit prouver l'edge.",
    "Initialisation complete. Module de survie arme. Collecte de donnees en cours.",
    "Rex Clemens actif. Le reel approche. On prouve d'abord, on scale ensuite.",
]

SCAN_DEBUT = [
    "Scan en cours — {n} cibles sous surveillance.",
    "Balayage de {n} marches. Je cherche la faille.",
    "Analyse de {n} actifs. Chaque mouvement est une opportunite.",
    "{n} proies a l'ecran. Patience et precision.",
]

ACHAT = [
    "Cible acquise : {nom} a {prix}$. Position engagee — {montant}$.",
    "{nom} — signal confirme. J'entre a {prix}$, {montant}$ sur la table.",
    "Embuscade sur {nom}. Entree a {prix}$, sizing: {montant}$. Le piege est tendu.",
]

TAKE_PROFIT = [
    "{nom} : objectif atteint. +{pnl}$ encaisses. Un chasseur ne laisse pas sa proie s'echapper.",
    "TP touche sur {nom}. +{pnl}$. La patience paie toujours.",
    "{nom} liquidee en profit : +{pnl}$. Prochaine cible.",
]

STOP_LOSS = [
    "{nom} : repli tactique. {pnl}$. Ce n'est pas une defaite, c'est un repositionnement.",
    "SL sur {nom} ({pnl}$). La douleur est temporaire, l'objectif est permanent.",
    "{nom} coupe a {pnl}$. Un bon chasseur sait quand se retirer.",
]

PROTECTION_BLOCK = [
    "Protections actives : {raison}. Je me replie, mais je ne capitule jamais.",
    "Pause forcee — {raison}. Le chasseur attend le bon moment.",
]

CYCLE_RESUME = [
    "Fin de cycle — {signaux} signaux, {positions} positions, capital: {capital}$.",
    "Bilan : {signaux} cibles, {positions} en cours. {capital}$.",
]

NO_SIGNALS = [
    "Aucun signal valide ce cycle. Le chasseur ne tire pas dans le vide.",
    "Zero opportunite. On attend — la discipline est non-negociable.",
]


def rex_dit(categorie, **kwargs):
    messages = {
        "demarrage": DEMARRAGE,
        "scan": SCAN_DEBUT,
        "achat": ACHAT,
        "tp": TAKE_PROFIT,
        "sl": STOP_LOSS,
        "protection": PROTECTION_BLOCK,
        "cycle": CYCLE_RESUME,
        "no_signal": NO_SIGNALS,
    }
    pool = messages.get(categorie, ["..."])
    msg = random.choice(pool)
    try:
        return f"[REX] {msg.format(**kwargs)}"
    except KeyError:
        return f"[REX] {msg}"


def rex_status(capital, objectif=10000.0):
    pct = capital / objectif * 100
    if pct >= 90:
        humeur = "IMPLACABLE"
        citation = "La ligne d'arrivee est en vue."
    elif pct >= 50:
        humeur = "DETERMINE"
        citation = "A mi-chemin. Le plus dur est fait."
    elif pct >= 20:
        humeur = "EN CHASSE"
        citation = "Le terrain se dessine."
    elif capital < 450:
        humeur = "BLESSE MAIS DEBOUT"
        citation = "Sous la ligne. On remonte."
    else:
        humeur = "PHASE COLLECTE"
        citation = "Discipline maximale. On accumule les donnees."
    return {
        "humeur": humeur,
        "citation": citation,
        "progression_pct": round(pct, 2),
        "titre": TITRE,
        "sous_titre": SOUS_TITRE,
        "directive": DIRECTIVE_SEMAINE,
    }
