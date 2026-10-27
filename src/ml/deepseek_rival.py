"""
Rex Clemens — DeepSeek Le Stratege (V2 OPTIMISE)
==================================================
Duel ML vs DeepSeek sur les signaux d'achat.
OPTIMISATION TOKENS: max 10 appels/heure, cache 30min, prompts courts.
Budget cible: ~$1.50/mois (50 appels/jour max).
"""

import json
import os
import time
from datetime import datetime, timezone

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"

MAX_CALLS_PER_HOUR = 10
CACHE_TTL_SECONDS = 1800
MAX_CALLS_PER_DAY = 50

_call_timestamps = []
_cache = {}
_daily_count = 0
_daily_reset_date = None

SCORE_BOARD = {
    "ml_wins": 0,
    "deepseek_wins": 0,
    "draws": 0,
    "total_duels": 0,
    "deepseek_pnl": 0.0,
    "ml_pnl": 0.0,
}


def _rate_limited():
    global _daily_count, _daily_reset_date
    now = datetime.now(timezone.utc)
    today = now.date()
    if _daily_reset_date != today:
        _daily_reset_date = today
        _daily_count = 0
    if _daily_count >= MAX_CALLS_PER_DAY:
        return True
    cutoff = time.time() - 3600
    active = [ts for ts in _call_timestamps if ts > cutoff]
    _call_timestamps.clear()
    _call_timestamps.extend(active)
    return len(active) >= MAX_CALLS_PER_HOUR


def _get_cached(sym):
    if sym in _cache:
        entry = _cache[sym]
        if time.time() - entry["ts"] < CACHE_TTL_SECONDS:
            return entry["result"]
    return None


def _set_cache(sym, result):
    _cache[sym] = {"ts": time.time(), "result": result}


def interroger_deepseek(sym, nom, prix, signal_ml, features_resume):
    """
    Interroge DeepSeek pour un avis ACHAT/CASH.
    Retourne un dict avec la decision et le raisonnement.
    """
    global _daily_count

    cached = _get_cached(sym)
    if cached:
        return cached

    if not HAS_REQUESTS or not DEEPSEEK_API_KEY:
        return {"decision": "NEUTRE", "raison": "API non configuree", "cached": False}

    if _rate_limited():
        return {"decision": "NEUTRE", "raison": f"Rate limit ({_daily_count}/{MAX_CALLS_PER_DAY}/jour)", "cached": False}

    prompt = (
        f"{nom} ({sym}) @ ${prix:.2f}. ML dit: {signal_ml}. "
        f"Donnees: {features_resume}. "
        f"Reponds ACHAT ou CASH en 1 mot, puis 1 phrase de raison. "
        f"Pas de NEUTRE — un strategue indecis est un strategue mort."
    )

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": "Tu es un analyste trading. Reponds ACHAT ou CASH, puis 1 phrase."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 60,
                "temperature": 0.3,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        _call_timestamps.append(time.time())
        _daily_count += 1

        first_word = content.split()[0].upper().rstrip(".:,!") if content else "NEUTRE"
        if first_word == "ACHAT":
            decision = "ACHAT"
        elif first_word in ("CASH", "VENTE", "VENDRE", "NON"):
            decision = "CASH"
        else:
            decision = "CASH"

        result = {
            "decision": decision,
            "raison": content[:120],
            "cached": False,
            "tokens_used": data.get("usage", {}).get("total_tokens", 0),
            "daily_calls": _daily_count,
        }
        _set_cache(sym, result)
        return result

    except Exception as e:
        return {"decision": "NEUTRE", "raison": f"Erreur: {str(e)[:60]}", "cached": False}


def enregistrer_resultat_duel(ml_decision, deepseek_decision, pnl_reel):
    """Enregistre le resultat d'un duel pour le scoreboard."""
    SCORE_BOARD["total_duels"] += 1
    if pnl_reel > 0:
        if ml_decision == "ACHAT":
            SCORE_BOARD["ml_wins"] += 1
            SCORE_BOARD["ml_pnl"] += pnl_reel
        if deepseek_decision == "ACHAT":
            SCORE_BOARD["deepseek_wins"] += 1
            SCORE_BOARD["deepseek_pnl"] += pnl_reel
    elif pnl_reel < 0:
        if ml_decision == "ACHAT":
            SCORE_BOARD["ml_pnl"] += pnl_reel
        if deepseek_decision == "ACHAT":
            SCORE_BOARD["deepseek_pnl"] += pnl_reel


def get_stats():
    return {
        **SCORE_BOARD,
        "daily_calls": _daily_count,
        "max_daily": MAX_CALLS_PER_DAY,
        "cache_size": len(_cache),
        "hourly_calls": len([ts for ts in _call_timestamps if ts > time.time() - 3600]),
    }
