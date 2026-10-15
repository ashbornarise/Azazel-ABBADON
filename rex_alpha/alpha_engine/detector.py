"""
Détecteurs de situations — Rex Alpha Engine

A. Détecteur de stagnation/baisse : identifie les actions délaissées
B. Détecteur de retournement : cherche les signes de reprise
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class StagnationDetector:
    """Identifie les actions en baisse prolongée ou stagnation."""

    def analyser(self, prix: pd.Series, volumes: pd.Series | None = None) -> dict:
        if len(prix) < 252:
            return {"eligible": False, "raison": "historique insuffisant"}

        dernier = float(prix.iloc[-1])
        ret = prix.pct_change()

        # Rendements multi-horizons
        ret_1m = float((dernier / prix.iloc[-21] - 1) * 100) if len(prix) > 21 else 0
        ret_3m = float((dernier / prix.iloc[-63] - 1) * 100) if len(prix) > 63 else 0
        ret_6m = float((dernier / prix.iloc[-126] - 1) * 100) if len(prix) > 126 else 0
        ret_1y = float((dernier / prix.iloc[-252] - 1) * 100) if len(prix) > 252 else 0

        # Distance au plus haut 52 semaines
        haut_52 = float(prix.iloc[-252:].max())
        dist_haut = float((dernier / haut_52 - 1) * 100)

        # Volatilité
        vol_20 = float(ret.iloc[-20:].std() * np.sqrt(252) * 100)
        vol_60 = float(ret.iloc[-60:].std() * np.sqrt(252) * 100) if len(ret) > 60 else vol_20

        # Durée de consolidation (jours sous la MA50)
        ma50 = prix.rolling(50).mean()
        sous_ma50 = int((prix.iloc[-60:] < ma50.iloc[-60:]).sum()) if len(prix) > 60 else 0

        # Volume relatif
        vol_ratio = 1.0
        if volumes is not None and len(volumes) > 20:
            vol_recent = float(volumes.iloc[-5:].mean())
            vol_normal = float(volumes.iloc[-60:-5].mean()) if len(volumes) > 65 else float(volumes.mean())
            vol_ratio = round(vol_recent / max(vol_normal, 1), 2)

        # Score de stagnation (0 = pas de stagnation, 100 = forte baisse prolongée)
        score = 0.0
        if dist_haut < -10:
            score += min(abs(dist_haut) * 0.5, 25)
        if ret_3m < -5:
            score += min(abs(ret_3m) * 0.3, 20)
        if ret_6m < -10:
            score += min(abs(ret_6m) * 0.2, 15)
        if sous_ma50 > 30:
            score += min(sous_ma50 * 0.3, 15)
        if vol_20 > 30:
            score += min((vol_20 - 30) * 0.3, 10)
        score = min(score, 100)

        en_stagnation = score >= 25

        return {
            "eligible": en_stagnation,
            "score_stagnation": round(score, 1),
            "ret_1m": round(ret_1m, 2),
            "ret_3m": round(ret_3m, 2),
            "ret_6m": round(ret_6m, 2),
            "ret_1y": round(ret_1y, 2),
            "dist_haut_52s": round(dist_haut, 2),
            "vol_20j": round(vol_20, 1),
            "vol_60j": round(vol_60, 1),
            "jours_sous_ma50": sous_ma50,
            "volume_ratio": vol_ratio,
        }


class ReversalDetector:
    """Détecte les signes de retournement potentiel."""

    def analyser(self, prix: pd.Series, volumes: pd.Series | None = None) -> dict:
        if len(prix) < 60:
            return {"reversal_score": 0, "signes": []}

        ret = prix.pct_change()
        signes = []
        score = 0.0

        # 1. Stabilisation du prix (volatilité en baisse)
        vol_10 = float(ret.iloc[-10:].std())
        vol_30 = float(ret.iloc[-30:].std())
        if vol_30 > 0 and vol_10 / vol_30 < 0.8:
            signes.append("volatilite_en_baisse")
            score += 15

        # 2. Prix au-dessus de la MA courte (reprise de tendance)
        ma10 = prix.rolling(10).mean()
        ma20 = prix.rolling(20).mean()
        if float(prix.iloc[-1]) > float(ma10.iloc[-1]):
            signes.append("prix_au_dessus_ma10")
            score += 10
        if float(ma10.iloc[-1]) > float(ma20.iloc[-1]):
            signes.append("ma10_croise_ma20")
            score += 15

        # 3. RSI en zone de rebond (sort de survendu)
        delta = ret.copy()
        gain = delta.clip(lower=0).rolling(14).mean()
        perte = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / perte.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1]) if not rsi.iloc[-1] != rsi.iloc[-1] else 50
        if 30 < rsi_val < 50:
            signes.append("rsi_zone_rebond")
            score += 10
        elif rsi_val < 30:
            signes.append("rsi_survendu")
            score += 5

        # 4. Volume en hausse (intérêt croissant)
        if volumes is not None and len(volumes) > 20:
            vol_5 = float(volumes.iloc[-5:].mean())
            vol_20 = float(volumes.iloc[-20:].mean())
            if vol_20 > 0 and vol_5 / vol_20 > 1.2:
                signes.append("volume_en_hausse")
                score += 15

        # 5. Rendement récent positif après baisse
        ret_5 = float(prix.iloc[-1] / prix.iloc[-5] - 1) if len(prix) > 5 else 0
        ret_20 = float(prix.iloc[-1] / prix.iloc[-20] - 1) if len(prix) > 20 else 0
        if ret_5 > 0.02 and ret_20 < 0:
            signes.append("rebond_court_terme")
            score += 15

        # 6. Force relative vs marché (amélioration)
        if len(prix) > 60:
            force_20 = float(prix.iloc[-1] / prix.iloc[-20] - 1)
            force_60 = float(prix.iloc[-1] / prix.iloc[-60] - 1)
            if force_20 > force_60:
                signes.append("force_relative_amelioration")
                score += 10

        score = min(score, 100)

        return {
            "reversal_score": round(score, 1),
            "signes": signes,
            "nb_signes": len(signes),
            "rsi": round(rsi_val, 1),
            "ret_5j": round(ret_5 * 100, 2) if ret_5 else 0,
            "ret_20j": round(ret_20 * 100, 2) if ret_20 else 0,
        }
