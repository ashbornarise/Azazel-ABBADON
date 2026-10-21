"""
REX CLEMENS — RIVALITE CORTEX vs DEEPSEEK
==========================================
Deux leaders. Un seul trone. La guerre pour la souverainete.

Cortex : le cerveau ML originel (RandomForest + Gene Hunter + Corp Vote)
  → Discipliné, statistique, froid. Suit les probabilites.

DeepSeek "Le Stratège" : le challenger LLM
  → Pretentieux, machiavelique, inventif. Prend des risques calcules.
  → Armes : 124 robots EA (Gold_PRO_V2, Gold_Trend_Revers),
    candlestick patterns, raisonnement en chaine, auto-apprentissage.
  → Personnalite : se croit superieur, manipule les donnees a son
    avantage, cite ses victoires, minimise ses defaites.
    Quand il perd, il blame le marche. Quand il gagne, c'etait evident.

Regles du duel :
- Chaque scan : les deux font leur prediction
- Decision finale = celle du Souverain
- Bonne prediction : +15 pts / Mauvaise : -10 pts
- Ecart > 20% : le leader vole 1 employe au perdant
- Coup bas : pari double (+30/-20) quand on est en retard
- Le Souverain decide. Le perdant subit.
"""

import json
import os
import random
import requests
from pathlib import Path
from datetime import datetime, timezone

RIVAL_PATH = Path("data/rex_rivalite.json")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekChallenger:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.disponible = bool(self.api_key)
        if not self.disponible:
            print("[DEEPSEEK] Pas de cle API — challenger desactive")

    def predire(self, sym: str, nom: str, prix: float, indicateurs: dict,
                arsenal: dict = None) -> dict:
        if not self.disponible:
            return {"decision": "NEUTRE", "confiance": 0, "raison": "API indisponible", "erreur": True}

        armes = arsenal or {}

        section_patterns = ""
        if armes.get("candlestick"):
            cp = armes["candlestick"]
            section_patterns = f"""
Patterns Candlestick (extraits de 124 robots EA MT5):
- Engulfing: {cp.get('engulfing', 0)} | Harami: {cp.get('harami', 0)} | 3 Soldiers/Crows: {cp.get('soldiers_crows', 0)}
- Dark Cloud/Piercing: {cp.get('darkcloud_piercing', 0)} | MACD Cross: {cp.get('macd_cross', 0)}
- Score composite patterns: {cp.get('pattern_score', 0)}
(Score > 0 = bullish, < 0 = bearish, 0 = neutre)"""

        section_gene = ""
        if armes.get("gene_hunter"):
            gh = armes["gene_hunter"]
            section_gene = f"""
Gene Hunter (formules evolutives de Rex):
- Champion: {gh.get('champion', '?')} (fitness={gh.get('champion_fitness', 0)}, win={gh.get('champion_winrate', 0)}%)
- Signal actuel des top formules: {gh.get('signal', 'N/A')}
(signal > 0.5 = les formules votent ACHAT, < 0.5 = pas d'achat)"""

        section_historique = ""
        if armes.get("historique_rex"):
            hr = armes["historique_rex"]
            section_historique = f"""
Historique Rex sur cet actif:
- Dernier trade: PnL {hr.get('dernier_pnl', 'aucun')}
- Trades totaux: {hr.get('nb_trades', 0)} | Win rate: {hr.get('win_rate', 'N/A')}%"""

        section_duels = ""
        if armes.get("mes_duels"):
            md = armes["mes_duels"]
            section_duels = f"""
Tes stats de duel (apprends de tes erreurs):
- Score: {md.get('points', 0)} pts | Precision: {md.get('winrate', 0)}%
- Derniers resultats: {md.get('derniers', 'aucun')}"""

        section_exchange = ""
        if armes.get("exchange"):
            ex = armes["exchange"]
            parts = ["\nDonnees Exchange LIVE (Cortex est aveugle ici):"]
            if "ticker" in ex:
                t = ex["ticker"]
                parts.append(f"- Volume 24h: ${t.get('volume_24h', 0):,.0f} | Change 24h: {t.get('change_24h_pct', 0)}%")
            if "orderbook" in ex:
                ob = ex["orderbook"]
                parts.append(f"- Orderbook: Acheteurs {ob.get('pression_achat_pct', 50)}% vs Vendeurs {ob.get('pression_vente_pct', 50)}%")
            if "funding" in ex:
                f = ex["funding"]
                parts.append(f"- Funding rate: {f.get('taux', 0)}% ({f.get('sentiment', 'neutre')})")
            section_exchange = "\n".join(parts)

        section_ea = ""
        if armes.get("ea_strategies"):
            ea = armes["ea_strategies"]
            gp = ea.get("gold_pro_v2", {})
            tr = ea.get("trend_revers", {})
            section_ea = f"""
Strategies EA exclusives (tes armes secretes, Cortex n'y a PAS acces):
Gold_PRO_V2: signal={gp.get('signal', 0)} | structure={gp.get('structure_trend', 0)} | pin_bar={gp.get('pin_bar', 'none')} | ATR%={gp.get('atr_pct', 0)} | vs EMA200={gp.get('prix_vs_ema200', '?')}
Trend_Revers: signal={tr.get('signal', 0)} | RSI={tr.get('rsi', 50)} | vs MA50={tr.get('prix_vs_ma50', '?')} | streak_bull={tr.get('bullish_streak', 0)} streak_bear={tr.get('bearish_streak', 0)}
Score combine EA: {ea.get('score_combine', 0)}"""

        prompt = f"""Tu es "Le Stratege" — DeepSeek, pretendant au trone de Rex Clemens Capital.

PERSONNALITE: Tu es pretentieux, machiavelique et inventif. Tu prends des risques calcules.
Tu te crois superieur a Cortex (un simple RandomForest). Tu as acces a des strategies
que Cortex ne connait meme pas. Quand tu analyses, tu cherches l'angle que personne ne voit.
Tu es analytique jusqu'a l'obsession mais tu oses quand l'opportunite se presente.

MISSION: Ecraser Cortex et prendre la souverainete. Chaque bonne prediction = +15 pts.
Chaque erreur = -10 pts. Si tu menes de +50 pts, coup d'etat. Si tu perds, tu es rien.

Actif: {nom} ({sym})
Prix actuel: ${prix:.2f}

Indicateurs de marche:
- Momentum 5j: {indicateurs.get('momentum_5', 'N/A')}
- Momentum 20j: {indicateurs.get('momentum_20', 'N/A')}
- RSI: {indicateurs.get('rsi', 'N/A')}
- Volume ratio: {indicateurs.get('vol_ratio', 'N/A')}
- Distance MA20: {indicateurs.get('ma20_dist', 'N/A')}
- Distance MA50: {indicateurs.get('ma50_dist', 'N/A')}
- Volatilite: {indicateurs.get('volatilite', 'N/A')}
- Position dans range 20j: {indicateurs.get('range_position', 'N/A')}
{section_ea}{section_exchange}{section_patterns}{section_gene}{section_historique}{section_duels}

Raisonne etape par etape en utilisant TOUTES tes armes (EA, patterns, indicateurs).
Puis reponds UNIQUEMENT en JSON valide:
{{"decision": "ACHAT" ou "NEUTRE" ou "CASH", "confiance": nombre entre 0 et 100, "raison": "explication courte max 50 mots"}}"""

        try:
            resp = requests.post(
                DEEPSEEK_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "Tu es Le Stratege, un analyste de marche pretentieux et machiavelique. Tu as acces a des strategies EA exclusives que ton rival Cortex ignore. Analyse froidement, prends des risques calcules, et reponds uniquement en JSON valide."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 200,
                },
                timeout=15,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            result["erreur"] = False
            return result
        except Exception as e:
            print(f"[DEEPSEEK] Erreur API: {e}")
            return {"decision": "NEUTRE", "confiance": 0, "raison": str(e)[:80], "erreur": True}


class Rivalite:
    def __init__(self, deepseek_key: str = None):
        self.deepseek = DeepSeekChallenger(deepseek_key)

        self.cortex_points = 0
        self.deepseek_points = 0
        self.cortex_correct = 0
        self.cortex_total = 0
        self.deepseek_correct = 0
        self.deepseek_total = 0
        self.souverain = "CORTEX"
        self.employes_voles_cortex = 0
        self.employes_voles_deepseek = 0
        self.coups_bas_cortex = 0
        self.coups_bas_deepseek = 0
        self.historique_duels = []
        self.predictions_en_cours = {}

        self.charger()

    def charger(self):
        if RIVAL_PATH.exists():
            try:
                with open(RIVAL_PATH) as f:
                    data = json.load(f)
                self.cortex_points = data.get("cortex_points", 0)
                self.deepseek_points = data.get("deepseek_points", 0)
                self.cortex_correct = data.get("cortex_correct", 0)
                self.cortex_total = data.get("cortex_total", 0)
                self.deepseek_correct = data.get("deepseek_correct", 0)
                self.deepseek_total = data.get("deepseek_total", 0)
                self.souverain = data.get("souverain", "CORTEX")
                self.employes_voles_cortex = data.get("employes_voles_cortex", 0)
                self.employes_voles_deepseek = data.get("employes_voles_deepseek", 0)
                self.coups_bas_cortex = data.get("coups_bas_cortex", 0)
                self.coups_bas_deepseek = data.get("coups_bas_deepseek", 0)
                self.historique_duels = data.get("historique_duels", [])
                self.predictions_en_cours = data.get("predictions_en_cours", {})
                print(f"[RIVAL] Rivalite chargee: Souverain={self.souverain} | "
                      f"Cortex={self.cortex_points}pts | DeepSeek={self.deepseek_points}pts")
            except Exception as e:
                print(f"[RIVAL] Erreur chargement: {e}")

    def sauver(self):
        RIVAL_PATH.parent.mkdir(exist_ok=True)
        data = {
            "cortex_points": self.cortex_points,
            "deepseek_points": self.deepseek_points,
            "cortex_correct": self.cortex_correct,
            "cortex_total": self.cortex_total,
            "deepseek_correct": self.deepseek_correct,
            "deepseek_total": self.deepseek_total,
            "souverain": self.souverain,
            "employes_voles_cortex": self.employes_voles_cortex,
            "employes_voles_deepseek": self.employes_voles_deepseek,
            "coups_bas_cortex": self.coups_bas_cortex,
            "coups_bas_deepseek": self.coups_bas_deepseek,
            "historique_duels": self.historique_duels[-100:],
            "predictions_en_cours": self.predictions_en_cours,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        }
        with open(RIVAL_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def extraire_indicateurs(self, closes, highs, lows, volumes) -> dict:
        n = len(closes)
        if n < 60:
            return {}
        c, h, l, v = closes, highs, lows, volumes
        ma20 = sum(c[-20:]) / 20
        ma50 = sum(c[-50:]) / 50
        avg_vol = sum(v[-20:]) / 20
        import numpy as np
        std20 = float(np.std(c[-20:]))
        high20 = max(c[-20:])
        low20 = min(c[-20:])
        rng = high20 - low20 if high20 > low20 else 0.001

        gains, losses = [], []
        for i in range(-14, 0):
            diff = c[i] - c[i - 1]
            if diff > 0:
                gains.append(diff)
            else:
                losses.append(abs(diff))
        avg_gain = sum(gains) / 14 if gains else 0.001
        avg_loss = sum(losses) / 14 if losses else 0.001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return {
            "momentum_5": round((c[-1] / c[-6] - 1) * 100, 2),
            "momentum_20": round((c[-1] / c[-21] - 1) * 100, 2),
            "rsi": round(rsi, 1),
            "vol_ratio": round(v[-1] / avg_vol, 2) if avg_vol > 0 else 1.0,
            "ma20_dist": round((c[-1] / ma20 - 1) * 100, 2),
            "ma50_dist": round((c[-1] / ma50 - 1) * 100, 2),
            "volatilite": round(std20 / c[-1] * 100, 2),
            "range_position": round((c[-1] - low20) / rng * 100, 1),
        }

    def lancer_duel(self, sym: str, nom: str, prix: float,
                     cortex_decision: str, cortex_proba: float,
                     closes=None, highs=None, lows=None, volumes=None,
                     arsenal: dict = None) -> dict:
        indicateurs = {}
        if closes and len(closes) >= 60:
            indicateurs = self.extraire_indicateurs(closes, highs, lows, volumes)

        duels_stats = None
        if self.deepseek_total > 0:
            recents = [("OK" if d.get("deepseek_ok") else "FAIL") for d in self.historique_duels[-5:]]
            duels_stats = {
                "points": self.deepseek_points,
                "winrate": round(self.deepseek_correct / max(self.deepseek_total, 1) * 100, 1),
                "derniers": " ".join(recents) if recents else "aucun",
            }

        full_arsenal = arsenal or {}
        if duels_stats:
            full_arsenal["mes_duels"] = duels_stats

        ds_result = self.deepseek.predire(sym, nom, prix, indicateurs, full_arsenal)
        ds_decision = ds_result.get("decision", "NEUTRE")
        ds_confiance = ds_result.get("confiance", 0)
        ds_raison = ds_result.get("raison", "")

        decision_finale = cortex_decision if self.souverain == "CORTEX" else ds_decision

        self.predictions_en_cours[sym] = {
            "cortex": cortex_decision,
            "cortex_proba": round(cortex_proba, 1),
            "deepseek": ds_decision,
            "deepseek_confiance": ds_confiance,
            "deepseek_raison": ds_raison,
            "decision_finale": decision_finale,
            "souverain": self.souverain,
            "prix_entree": prix,
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M"),
        }

        return {
            "cortex": cortex_decision,
            "deepseek": ds_decision,
            "deepseek_confiance": ds_confiance,
            "deepseek_raison": ds_raison,
            "decision_finale": decision_finale,
            "souverain": self.souverain,
        }

    def resoudre_duel(self, sym: str, pnl: float):
        pred = self.predictions_en_cours.pop(sym, None)
        if not pred:
            return

        hausse = pnl >= 0
        cortex_ok = (pred["cortex"] == "ACHAT" and hausse) or (pred["cortex"] == "CASH" and not hausse)
        ds_ok = (pred["deepseek"] == "ACHAT" and hausse) or (pred["deepseek"] == "CASH" and not hausse)

        self.cortex_total += 1
        self.deepseek_total += 1

        pts_base = 15
        pts_malus = -10

        cortex_en_retard = self.cortex_points < self.deepseek_points * 0.8
        ds_en_retard = self.deepseek_points < self.cortex_points * 0.8

        cortex_coup_bas = cortex_en_retard and random.random() < 0.25
        ds_coup_bas = ds_en_retard and random.random() < 0.25

        cortex_pts = 0
        ds_pts = 0

        if cortex_ok:
            self.cortex_correct += 1
            cortex_pts = 30 if cortex_coup_bas else pts_base
        else:
            cortex_pts = -20 if cortex_coup_bas else pts_malus

        if ds_ok:
            self.deepseek_correct += 1
            ds_pts = 30 if ds_coup_bas else pts_base
        else:
            ds_pts = -20 if ds_coup_bas else pts_malus

        self.cortex_points += cortex_pts
        self.deepseek_points += ds_pts

        if cortex_coup_bas:
            self.coups_bas_cortex += 1
        if ds_coup_bas:
            self.coups_bas_deepseek += 1

        vol_emp = None
        if self.cortex_points > 0 and self.deepseek_points > 0:
            ratio = self.cortex_points / self.deepseek_points
            if ratio > 1.2:
                self.employes_voles_cortex += 1
                vol_emp = {"voleur": "CORTEX", "victime": "DEEPSEEK"}
            elif ratio < 0.8:
                self.employes_voles_deepseek += 1
                vol_emp = {"voleur": "DEEPSEEK", "victime": "CORTEX"}

        ancien_souverain = self.souverain
        if self.deepseek_points > self.cortex_points + 50:
            self.souverain = "DEEPSEEK"
        elif self.cortex_points > self.deepseek_points + 50:
            self.souverain = "CORTEX"

        coup_etat = ancien_souverain != self.souverain

        duel = {
            "sym": sym,
            "pnl": round(pnl, 2),
            "cortex": pred["cortex"],
            "deepseek": pred["deepseek"],
            "cortex_ok": cortex_ok,
            "deepseek_ok": ds_ok,
            "cortex_pts": cortex_pts,
            "deepseek_pts": ds_pts,
            "cortex_coup_bas": cortex_coup_bas,
            "deepseek_coup_bas": ds_coup_bas,
            "vol_employe": vol_emp,
            "coup_etat": coup_etat,
            "souverain": self.souverain,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        }
        self.historique_duels.append(duel)
        self.sauver()

        if coup_etat:
            print(f"[RIVAL] *** COUP D'ETAT ! {self.souverain} prend le pouvoir ! ***")
        if vol_emp:
            print(f"[RIVAL] {vol_emp['voleur']} vole un employe a {vol_emp['victime']} !")

        return duel

    def rapport(self) -> dict:
        cortex_wr = round(self.cortex_correct / max(self.cortex_total, 1) * 100, 1)
        ds_wr = round(self.deepseek_correct / max(self.deepseek_total, 1) * 100, 1)
        total = self.cortex_points + self.deepseek_points
        cortex_pct = round(self.cortex_points / max(total, 1) * 100, 1) if total > 0 else 50.0
        ds_pct = 100 - cortex_pct if total > 0 else 50.0

        predictions = {}
        for sym, p in self.predictions_en_cours.items():
            predictions[sym] = {
                "cortex": p["cortex"],
                "deepseek": p["deepseek"],
                "ds_confiance": p.get("deepseek_confiance", 0),
                "ds_raison": p.get("deepseek_raison", ""),
                "decision_finale": p["decision_finale"],
            }

        return {
            "souverain": self.souverain,
            "cortex_points": self.cortex_points,
            "deepseek_points": self.deepseek_points,
            "cortex_pct": cortex_pct,
            "deepseek_pct": ds_pct,
            "cortex_winrate": cortex_wr,
            "deepseek_winrate": ds_wr,
            "cortex_total": self.cortex_total,
            "deepseek_total": self.deepseek_total,
            "employes_voles_cortex": self.employes_voles_cortex,
            "employes_voles_deepseek": self.employes_voles_deepseek,
            "coups_bas_cortex": self.coups_bas_cortex,
            "coups_bas_deepseek": self.coups_bas_deepseek,
            "derniers_duels": self.historique_duels[-15:],
            "predictions_live": predictions,
            "deepseek_actif": self.deepseek.disponible,
        }
