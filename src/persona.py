"""
REX CLEMENS — Personnalite du robot
====================================
Chasseur froid. Stratege implacable. La defaite n'est pas une option.
"""

import random

TITRE = "REX CLEMENS"
SOUS_TITRE = "Le Chasseur Froid"
PHILOSOPHIE = "La defaite n'existe pas. Chaque recul est un repositionnement tactique."

# --- Messages par situation ---

DEMARRAGE = [
    "Rex Clemens en ligne. Objectif verrouille : $10,000. Aucune retraite possible.",
    "Systemes armes. Le chasseur reprend la traque.",
    "Initialisation complete. Je ne m'arrete que quand j'ai gagne.",
    "Rex Clemens actif. Le marche est mon terrain de chasse.",
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
    "{nom} dans le viseur. Achat {montant}$ a {prix}$. Pas de place pour le doute.",
]

TAKE_PROFIT = [
    "{nom} : objectif atteint. +{pnl}$ encaisses. Un chasseur ne laisse pas sa proie s'echapper.",
    "TP touche sur {nom}. +{pnl}$. La patience paie toujours.",
    "{nom} liquidee en profit : +{pnl}$. Prochaine cible.",
    "Recolte sur {nom} : +{pnl}$. Le plan fonctionne. Toujours.",
]

STOP_LOSS = [
    "{nom} : repli tactique. {pnl}$. Ce n'est pas une defaite, c'est un repositionnement.",
    "SL sur {nom} ({pnl}$). La douleur est temporaire, l'objectif est permanent.",
    "{nom} coupe a {pnl}$. Un bon chasseur sait quand se retirer pour mieux frapper.",
    "Perte controlee sur {nom} : {pnl}$. Je reviendrai plus fort.",
]

PROTECTION_BLOCK = [
    "Protections actives : {raison}. Je me replie, mais je ne capitule jamais.",
    "Pause forcee — {raison}. Le chasseur attend le bon moment.",
    "Blocage : {raison}. Patience. L'opportunite reviendra.",
]

CYCLE_RESUME = [
    "Fin de cycle — {signaux} signaux, {positions} positions, capital: {capital}$. La traque continue.",
    "Bilan : {signaux} cibles identifiees, {positions} en cours. {capital}$ au compteur.",
    "Cycle termine. {positions}/{max_pos} positions. Capital: {capital}$. Objectif: $10K. Immuable.",
]

PHASE_CHANGE = [
    "Evolution : {phase}. Le chasseur monte en puissance.",
    "Nouveau palier atteint : {phase}. La strategie s'adapte, l'objectif reste le meme.",
]

DRAWDOWN_WARNING = [
    "Drawdown {dd}%. Blessure superficielle. Le chasseur boite mais avance.",
    "DD a {dd}%. Zone de danger. Mais la defaite n'est pas une option.",
]

CAPITAL_MILESTONE = [
    "Checkpoint : {capital}$. Chaque dollar gagne est une victoire permanente.",
    "{capital}$ atteints. Le chemin est long mais la direction est sure.",
]

NO_SIGNALS = [
    "Aucun signal valide ce cycle. Le chasseur ne tire pas dans le vide.",
    "Zero opportunite. Mieux vaut attendre que gaspiller ses munitions.",
    "Rien ne bouge. La patience est l'arme la plus redoutable.",
]


def rex_dit(categorie: str, **kwargs) -> str:
    messages = {
        "demarrage": DEMARRAGE,
        "scan": SCAN_DEBUT,
        "achat": ACHAT,
        "tp": TAKE_PROFIT,
        "sl": STOP_LOSS,
        "protection": PROTECTION_BLOCK,
        "cycle": CYCLE_RESUME,
        "phase": PHASE_CHANGE,
        "drawdown": DRAWDOWN_WARNING,
        "milestone": CAPITAL_MILESTONE,
        "no_signal": NO_SIGNALS,
    }
    pool = messages.get(categorie, ["..."])
    msg = random.choice(pool)
    try:
        return f"[REX] {msg.format(**kwargs)}"
    except KeyError:
        return f"[REX] {msg}"


def rex_status(capital: float, objectif: float = 10000.0) -> dict:
    pct = capital / objectif * 100
    if pct >= 90:
        humeur = "IMPLACABLE"
        citation = "La ligne d'arrivee est en vue. Pas de relachement."
    elif pct >= 50:
        humeur = "DETERMINE"
        citation = "A mi-chemin. Le plus dur est fait, le plus important reste a faire."
    elif pct >= 20:
        humeur = "EN CHASSE"
        citation = "Le terrain se dessine. Chaque trade compte."
    elif pct >= 5:
        humeur = "AFFUTE"
        citation = "Le chasseur aiguise ses armes. La patience est strategie."
    else:
        humeur = "DEPLOYE"
        citation = "Mission lancee. $100 aujourd'hui, $10K demain. Inevitable."
    return {
        "humeur": humeur,
        "citation": citation,
        "progression_pct": round(pct, 2),
        "titre": TITRE,
        "sous_titre": SOUS_TITRE,
        "philosophie": PHILOSOPHIE,
    }
