"""
Rex Clemens Capital — Trading temps reel V2
=============================================
Multi-classifier ML (RF+LightGBM+XGBoost+GBM) avec 31 features + Optuna hyperopt.
Protections: MaxDrawdown, StoplossGuard, CooldownPeriod, FeeFilter.
Seuils dynamiques (mean +/- N*std) au lieu de seuils fixes.
Trade toutes les 5 minutes, prix toutes les 30 secondes.

Usage : python run_realtime.py
Dashboard : http://localhost:8777
"""

import http.server
import json
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.ml.features import calculer_features, calculer_features_v2
from src.ml.protections import GestionnaireProtections
from src.ml.multi_classifier import benchmark_classifieurs, seuil_dynamique
from src.ml.hyperopt import optimiser_tous, construire_classifieurs_optimises
from src.strategies.ml_strategy import MLStrategy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

ETAT_PATH = Path("data/etat_realtime.json")
ML_PATH = Path("data/etat_ml.json")
HISTORY_PATH = Path("data/trade_history.json")
PORT = 8777
CAPITAL_INIT = 100.0
TRADE_INTERVAL = 300  # 5 minutes

ACTIFS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "MSFT": "Microsoft",
    "SOL-USD": "Solana",
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


etat = {
    "capital": CAPITAL_INIT,
    "capital_initial": CAPITAL_INIT,
    "cash": CAPITAL_INIT,
    "valeur_positions": 0.0,
    "pnl": 0.0,
    "pnl_pct": 0.0,
    "drawdown": 0.0,
    "plus_haut": CAPITAL_INIT,
    "nb_trades": 0,
    "trades_gagnants": 0,
    "trades_perdants": 0,
    "gain_total": 0.0,
    "perte_totale": 0.0,
    "positions": {},
    "signaux": {},
    "prix_historique": {},
    "equity": [],
    "timestamps_equity": [],
    "trades": [],
    "derniere_analyse": "-",
    "prochaine_analyse": "-",
    "timestamp": "-",
    "actif": True,
}

ml_etat = {
    "actifs": {},
    "timestamp": "-",
    "nb_features": 31,
    "modele": "Multi-Classifier (RF+LightGBM+XGBoost+GBM)",
    "arbres": 100,
    "walk_forward": True,
    "taille_train": 252,
    "pas_reentrain": 21,
    "features_noms": [
        "ret_1j", "ret_5j", "ret_20j",
        "vol_10j", "vol_20j",
        "ma_ratio_5_20", "ma_ratio_20_60",
        "rsi_14", "dist_haut_20j",
        "amplitude_5j", "skew_20j",
        "macd_hist", "stoch_k", "stoch_d",
        "bb_position", "atr_14", "trend_strength",
        "ret_60j", "volume_ratio", "volume_trend",
        "kurt_20j", "pct_hausse_20j", "dist_bas_20j",
        "adx_14", "bb_width", "jour_semaine", "heure",
        "pattern_score", "engulfing", "macd_cross", "cci_20",
    ],
    "protections": {
        "max_drawdown": "-5%",
        "stoploss_guard": "3 pertes consecutives",
        "cooldown": "300s par actif",
        "fee_filter": "gain > 2x frais",
    },
    "seuils_dynamiques": {},
    "benchmark_classifieurs": {},
    "hyperopt": {},
}

protections = GestionnaireProtections(
    max_drawdown_pct=-5.0,
    pertes_max_consecutives=3,
    cooldown_seconds=300,
    fee_pct=0.1,
)

probas_historiques = {}

hyperopt_cache = {}
hyperopt_derniere_run = {}
HYPEROPT_INTERVAL = 3600

ohlcv_cache = {}
lock = threading.Lock()


def charger_historique():
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def sauver_historique(trades):
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(trades, f, cls=NumpyEncoder)


def fetch_ohlcv(sym, period="5d", interval="5m"):
    now = time.time()
    key = f"{sym}_{period}_{interval}"
    cached = ohlcv_cache.get(key)
    if cached and now - cached["ts"] < 120:
        return cached["data"]
    try:
        t = yf.Ticker(sym)
        df = t.history(period=period, interval=interval)
        if df.empty:
            return cached["data"] if cached else []
        candles = []
        for idx, row in df.iterrows():
            ts_str = idx.strftime("%Y-%m-%d %H:%M") if hasattr(idx, "strftime") else str(idx)[:16]
            candles.append({
                "t": ts_str,
                "o": round(float(row["Open"]), 4),
                "h": round(float(row["High"]), 4),
                "l": round(float(row["Low"]), 4),
                "c": round(float(row["Close"]), 4),
                "v": int(row.get("Volume", 0)),
            })
        ohlcv_cache[key] = {"ts": now, "data": candles}
        return candles
    except Exception as e:
        print(f"[OHLCV] {sym}: {e}")
        return cached["data"] if cached else []


def sauver_ml():
    ML_PATH.parent.mkdir(exist_ok=True)
    with open(ML_PATH, "w") as f:
        json.dump(ml_etat, f, cls=NumpyEncoder)


def sauver():
    ETAT_PATH.parent.mkdir(exist_ok=True)
    with open(ETAT_PATH, "w") as f:
        json.dump(etat, f, cls=NumpyEncoder)


def calculer_valeur_totale():
    val_pos = 0.0
    for sym, pos in etat["positions"].items():
        if pos.get("en_position") and sym in etat["signaux"]:
            prix = etat["signaux"][sym].get("prix", 0)
            val_pos += pos["quantite"] * prix
    etat["valeur_positions"] = round(val_pos, 2)
    total = etat["cash"] + val_pos
    etat["capital"] = round(total, 2)
    etat["pnl"] = round(total - CAPITAL_INIT, 2)
    etat["pnl_pct"] = round((total / CAPITAL_INIT - 1) * 100, 2)
    etat["plus_haut"] = max(etat["plus_haut"], total)
    if etat["plus_haut"] > 0:
        etat["drawdown"] = round((total - etat["plus_haut"]) / etat["plus_haut"] * 100, 2)


def fetch_prix():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for sym, nom in ACTIFS.items():
        try:
            t = yf.Ticker(sym)
            h = t.history(period="1d", interval="1m")
            if h.empty:
                h = t.history(period="5d", interval="5m")
            if h.empty:
                continue
            prix = float(h["Close"].iloc[-1])

            if sym not in etat["signaux"]:
                etat["signaux"][sym] = {"nom": nom, "decision": "ATTENTE", "exposition": 0, "prix": prix}
            etat["signaux"][sym]["prix"] = round(prix, 2)

            if sym not in etat["prix_historique"]:
                etat["prix_historique"][sym] = []
            etat["prix_historique"][sym].append([now, round(prix, 2)])
            if len(etat["prix_historique"][sym]) > 300:
                etat["prix_historique"][sym] = etat["prix_historique"][sym][-300:]
        except Exception as e:
            print(f"  [PRIX] {sym}: {e}")

    with lock:
        calculer_valeur_totale()
        etat["timestamp"] = now
        ts = datetime.now(timezone.utc).strftime("%H:%M")
        etat["equity"].append(round(etat["capital"], 2))
        etat["timestamps_equity"].append(ts)
        if len(etat["equity"]) > 300:
            etat["equity"] = etat["equity"][-300:]
            etat["timestamps_equity"] = etat["timestamps_equity"][-300:]
        sauver()


def analyser_ml_details(sym, nom, hist_df, expo, decision):
    try:
        prix_serie = hist_df["Close"].dropna()
        volume = hist_df.get("Volume")
        high = hist_df.get("High")
        low = hist_df.get("Low")
        open_ = hist_df.get("Open")

        features = calculer_features_v2(prix_serie, volume=volume, high=high, low=low, open_=open_)
        masque = features.notna().all(axis=1)
        feat_valid = features[masque]
        if len(feat_valid) < 300:
            return

        ret_demain = prix_serie.pct_change().shift(-1)
        cible = (ret_demain > 0).astype(float)

        train_end = len(feat_valid) - 1
        val_size = min(63, train_end // 5)
        train_start = max(0, train_end - 252 - val_size)
        train_stop = train_end - val_size

        X_train = feat_valid.iloc[train_start:train_stop]
        y_train = cible.loc[X_train.index].dropna()
        X_train = X_train.loc[y_train.index]

        X_val = feat_valid.iloc[train_stop:train_end]
        y_val = cible.loc[X_val.index].dropna()
        X_val = X_val.loc[y_val.index]

        if len(X_train) < 100 or len(X_val) < 10:
            return

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_last_s = scaler.transform(feat_valid.iloc[[-1]])
        feat_names = list(features.columns)

        now_ts = time.time()
        clf_custom = None
        if sym not in hyperopt_derniere_run or (now_ts - hyperopt_derniere_run[sym]) > HYPEROPT_INTERVAL:
            try:
                print(f"  [HYPEROPT] {nom}: optimisation Optuna (30 trials)...")
                h_results = optimiser_tous(
                    X_train_s, y_train.values, X_val_s, y_val.values,
                    n_trials=30, seed=42,
                )
                if h_results:
                    hyperopt_cache[sym] = h_results
                    hyperopt_derniere_run[sym] = now_ts
                    clf_custom = construire_classifieurs_optimises(h_results, seed=42)
                    best_h = min(h_results.values(), key=lambda r: r.log_loss_val)
                    print(f"  [HYPEROPT] {nom}: meilleur={best_h.nom} LogLoss={best_h.log_loss_val} Acc={best_h.accuracy_val}%")
                    ml_etat.setdefault("hyperopt", {})[sym] = {
                        r.nom: {"params": r.meilleurs_params, "log_loss": r.log_loss_val, "accuracy": r.accuracy_val}
                        for r in h_results.values()
                    }
            except Exception as e:
                print(f"  [HYPEROPT] {nom}: {e}")
        elif sym in hyperopt_cache:
            clf_custom = construire_classifieurs_optimises(hyperopt_cache[sym], seed=42)

        meilleur, tous_resultats = benchmark_classifieurs(
            X_train_s, y_train.values,
            X_val_s, y_val.values,
            X_last_s, feat_names,
            classifieurs_custom=clf_custom,
        )
        if meilleur is None:
            return

        proba_hausse = meilleur.proba_hausse
        proba_baisse = round(100 - proba_hausse, 1)

        if sym not in probas_historiques:
            probas_historiques[sym] = []
        probas_historiques[sym].append(proba_hausse)
        if len(probas_historiques[sym]) > 100:
            probas_historiques[sym] = probas_historiques[sym][-100:]

        seuil_a, seuil_v = seuil_dynamique(
            np.array(probas_historiques[sym]), n_std=1.0,
        )
        ml_etat["seuils_dynamiques"][sym] = {
            "achat": seuil_a, "vente": seuil_v,
        }

        last_row = feat_valid.iloc[-1]
        feat_vals = {c: round(float(last_row[c]), 4) for c in features.columns}

        bench_resume = {}
        for rnom, r in tous_resultats.items():
            bench_resume[rnom] = {
                "accuracy": r.accuracy_val,
                "log_loss": r.log_loss_val,
                "proba_hausse": r.proba_hausse,
            }
        ml_etat["benchmark_classifieurs"][sym] = bench_resume

        n_recent = min(30, len(feat_valid))
        recent_feat = feat_valid.iloc[-n_recent:]
        recent_s = scaler.transform(recent_feat)

        from sklearn.ensemble import RandomForestClassifier
        rf_quick = RandomForestClassifier(
            n_estimators=50, max_depth=5, min_samples_leaf=20,
            random_state=42, n_jobs=1,
        )
        rf_quick.fit(X_train_s, y_train.values)
        recent_proba = rf_quick.predict_proba(recent_s)
        hist_expo = [round(float(p[1]) * 100, 1) if len(p) == 2 else 50.0 for p in recent_proba]
        hist_dates = [d.strftime("%m-%d") for d in recent_feat.index]

        ml_etat["actifs"][sym] = {
            "nom": nom,
            "decision": decision,
            "exposition": round(expo * 100, 1),
            "proba_hausse": proba_hausse,
            "proba_baisse": proba_baisse,
            "meilleur_modele": meilleur.nom,
            "accuracy_val": meilleur.accuracy_val,
            "log_loss": meilleur.log_loss_val,
            "jours_entrainement": len(X_train),
            "jours_validation": len(X_val),
            "feature_importance": meilleur.feature_importance,
            "feature_valeurs": feat_vals,
            "historique_expo": hist_expo,
            "historique_dates": hist_dates,
            "prix": round(float(prix_serie.iloc[-1]), 2),
            "seuil_achat": seuil_a,
            "seuil_vente": seuil_v,
            "nb_classifieurs": len(tous_resultats),
        }
    except Exception as e:
        print(f"  [ML-DETAIL] {nom}: {e}")


def analyser_et_trader():
    now = datetime.now(timezone.utc)
    etat["derniere_analyse"] = now.strftime("%Y-%m-%d %H:%M")
    etat["prochaine_analyse"] = (now + timedelta(seconds=TRADE_INTERVAL)).strftime("%Y-%m-%d %H:%M")

    print(f"\n{'='*50}")
    print(f"[TRADE] Analyse a {now.strftime('%H:%M:%S')} UTC")
    print(f"  Cash: {etat['cash']:.2f}$ | Capital: {etat['capital']:.2f}$")

    peut, raison_block = protections.peut_trader("", etat.get("drawdown", 0.0))
    if not peut:
        print(f"  PROTECTIONS ACTIVES: {raison_block}")
        etat["protection_active"] = raison_block
        with lock:
            sauver()
        return
    etat["protection_active"] = ""

    for sym, nom in ACTIFS.items():
        try:
            peut_sym, raison_sym = protections.peut_trader(sym, etat.get("drawdown", 0.0))
            if not peut_sym:
                print(f"  {nom}: BLOQUE - {raison_sym}")
                continue

            t = yf.Ticker(sym)
            hist = t.history(period="2y", interval="1d")
            if hist.empty or len(hist) < 300:
                print(f"  {nom}: pas assez de donnees ({len(hist)} jours)")
                continue

            prix_serie = hist["Close"].dropna()
            prix_actuel = float(prix_serie.iloc[-1])

            ml = MLStrategy(taille_min_train=252, pas_reentrain=21, n_arbres=50, seed=42)
            signal = ml.calculer_exposition(prix_serie)
            expo = float(signal.iloc[-1])

            seuil_a = ml_etat["seuils_dynamiques"].get(sym, {}).get("achat", 55.0)
            seuil_v = ml_etat["seuils_dynamiques"].get(sym, {}).get("vente", 40.0)
            proba_pct = expo * 100

            if proba_pct >= seuil_a:
                decision = "ACHAT"
            elif proba_pct <= seuil_v:
                decision = "CASH"
            else:
                decision = "NEUTRE"

            etat["signaux"][sym] = {
                "nom": nom,
                "decision": decision,
                "exposition": round(expo, 3),
                "prix": round(prix_actuel, 2),
                "seuil_achat": seuil_a,
                "seuil_vente": seuil_v,
            }

            analyser_ml_details(sym, nom, hist, expo, decision)
            sauver_ml()

            pos = etat["positions"].get(sym, {})

            if decision == "ACHAT" and not pos.get("en_position"):
                montant = etat["cash"] * 0.20
                if montant >= 1:
                    qte = montant / prix_actuel
                    etat["positions"][sym] = {
                        "en_position": True,
                        "prix_entree": prix_actuel,
                        "quantite": qte,
                        "montant_entree": round(montant, 2),
                        "date": now.strftime("%Y-%m-%d %H:%M"),
                    }
                    etat["cash"] -= montant
                    etat["nb_trades"] += 1
                    etat["trades"].append({
                        "date": now.strftime("%m-%d %H:%M"),
                        "sym": sym, "nom": nom,
                        "action": "ACHAT",
                        "prix": round(prix_actuel, 2),
                        "montant": round(montant, 2),
                        "capital": round(etat["cash"], 2),
                    })
                    protections.apres_trade(sym, 0.0)
                    print(f"  {nom}: ACHAT {montant:.2f}$ a {prix_actuel:.2f} (proba {proba_pct:.0f}% >= seuil {seuil_a}%)")
                else:
                    print(f"  {nom}: ACHAT souhaite mais cash insuffisant ({etat['cash']:.2f}$)")

            elif decision == "CASH" and pos.get("en_position"):
                valeur = pos["quantite"] * prix_actuel
                pnl = valeur - pos["montant_entree"]

                rentable, raison_fee = protections.fee_filter.est_rentable(pnl, pos["montant_entree"])
                if not rentable and pnl > 0:
                    print(f"  {nom}: VENTE annulee - {raison_fee}")
                    continue

                etat["cash"] += valeur
                if pnl >= 0:
                    etat["trades_gagnants"] += 1
                    etat["gain_total"] += pnl
                else:
                    etat["trades_perdants"] += 1
                    etat["perte_totale"] += abs(pnl)
                etat["positions"][sym] = {"en_position": False}
                etat["trades"].append({
                    "date": now.strftime("%m-%d %H:%M"),
                    "sym": sym, "nom": nom,
                    "action": "VENTE",
                    "prix": round(prix_actuel, 2),
                    "montant": round(valeur, 2),
                    "pnl": round(pnl, 2),
                    "capital": round(etat["cash"], 2),
                })
                protections.apres_trade(sym, pnl)
                print(f"  {nom}: VENTE {valeur:.2f}$ (PnL: {pnl:+.2f}$)")

            else:
                print(f"  {nom}: {decision} | {prix_actuel:.2f} | proba {proba_pct:.0f}% (seuils: A>{seuil_a}% V<{seuil_v}%)")

        except Exception as e:
            print(f"  {nom}: ERREUR - {e}")

    with lock:
        calculer_valeur_totale()
        ml_etat["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
        sauver()
        sauver_ml()
        sauver_historique(etat["trades"])

    total_pos = sum(1 for p in etat["positions"].values() if p.get("en_position"))
    print(f"  Positions ouvertes: {total_pos} | Capital: {etat['capital']:.2f}$ | PnL: {etat['pnl']:+.2f}$")


def loop_prix():
    while True:
        try:
            fetch_prix()
        except Exception as e:
            print(f"[PRIX] Erreur globale: {e}")
        time.sleep(30)


def loop_trade():
    analyser_et_trader()
    while True:
        time.sleep(TRADE_INTERVAL)
        try:
            analyser_et_trader()
        except Exception as e:
            print(f"[TRADE] Erreur globale: {e}")


class Handler(http.server.SimpleHTTPRequestHandler):
    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data.encode() if isinstance(data, str) else json.dumps(data, cls=NumpyEncoder).encode())

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/api/etat":
            with lock:
                try:
                    data = ETAT_PATH.read_text()
                except FileNotFoundError:
                    data = "{}"
            self._json_response(data)
            return

        if path == "/api/ml":
            with lock:
                try:
                    data = ML_PATH.read_text()
                except FileNotFoundError:
                    data = "{}"
            self._json_response(data)
            return

        if path == "/api/ohlcv":
            sym = params.get("sym", ["BTC-USD"])[0]
            period = params.get("period", ["5d"])[0]
            interval = params.get("interval", ["5m"])[0]
            candles = fetch_ohlcv(sym, period, interval)
            self._json_response({"sym": sym, "candles": candles})
            return

        if path == "/api/history":
            trades = charger_historique()
            page = int(params.get("page", ["1"])[0])
            per_page = int(params.get("per_page", ["50"])[0])
            sym_filter = params.get("sym", [None])[0]
            if sym_filter:
                trades = [t for t in trades if t.get("sym") == sym_filter]
            total = len(trades)
            trades_rev = list(reversed(trades))
            start = (page - 1) * per_page
            end = start + per_page
            self._json_response({
                "trades": trades_rev[start:end],
                "total": total,
                "page": page,
                "pages": (total + per_page - 1) // per_page if total > 0 else 1,
            })
            return

        if path == "/ml":
            self.path = "/dashboard_ml.html"
            super().do_GET()
            return

        if path == "/old":
            self.path = "/dashboard_realtime.html"
            super().do_GET()
            return

        if path == "/" or path == "":
            self.path = "/dashboard_mega.html"
        super().do_GET()

    def log_message(self, fmt, *args):
        pass


def main():
    ETAT_PATH.parent.mkdir(exist_ok=True)
    saved_trades = charger_historique()
    if saved_trades:
        etat["trades"] = saved_trades
        etat["nb_trades"] = len(saved_trades)
        print(f"[HISTORY] {len(saved_trades)} trades charges depuis l'historique")
    sauver()

    server = http.server.HTTPServer(("", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[SERVEUR] Dashboard sur http://localhost:{PORT}")

    threading.Thread(target=loop_prix, daemon=True).start()
    print("[PRIX] Rafraichissement toutes les 30 secondes")

    print(f"[TRADE] Intervalle: {TRADE_INTERVAL}s ({TRADE_INTERVAL//60} min)")
    print("[TRADE] Premiere analyse en cours...")
    print()

    try:
        loop_trade()
    except KeyboardInterrupt:
        print("\n[SERVEUR] Arret.")
        server.shutdown()


if __name__ == "__main__":
    main()
