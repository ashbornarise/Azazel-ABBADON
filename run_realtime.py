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
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=ResourceWarning)

import numpy as np
import pandas as pd
import yfinance as yf

from src.ml.features import calculer_features, calculer_features_v2
from src.ml.protections import (
    GestionnaireProtections, calculer_atr, calculer_sl_structurel,
    calculer_tp_structurel, position_sizing_atr, trailing_paliers,
)
from src.ml.multi_classifier import benchmark_classifieurs, seuil_dynamique
from src.ml.hyperopt import optimiser_tous, construire_classifieurs_optimises
from src.ml.market_intel import MarketIntelligence
from src.ml.deep_model import DeepEnsemble
from src.ml.social_intel import SocialIntelligence
from src.strategies.ml_strategy import MLStrategy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

ETAT_PATH = Path("data/etat_realtime.json")
ML_PATH = Path("data/etat_ml.json")
HISTORY_PATH = Path("data/trade_history.json")
PORT = int(os.environ.get("PORT", 8777))
CAPITAL_INIT = 100.0
TRADE_INTERVAL = 120  # 2 minutes — frequence augmentee pour capter plus de mouvements

MARCHES_PRIORITAIRES = {
    "BTC-USD": 1.5, "ETH-USD": 1.2, "SOL-USD": 1.2, "BNB-USD": 1.1,
    "NVDA": 1.3, "TSLA": 1.2, "AAPL": 1.1, "MSFT": 1.1,
    "GC=F": 1.3, "CL=F": 1.1,
}
BASE_ALLOC = 0.15
MAX_ALLOC = 0.30

ACTIFS = {
    # === CRYPTO TOP 50 ===
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
    "BNB-USD": "BNB", "XRP-USD": "XRP", "ADA-USD": "Cardano",
    "AVAX-USD": "Avalanche", "DOGE-USD": "Dogecoin", "DOT-USD": "Polkadot",
    "LINK-USD": "Chainlink", "ATOM-USD": "Cosmos",
    "LTC-USD": "Litecoin", "NEAR-USD": "Near",
    "OP-USD": "Optimism", "ARB-USD": "Arbitrum",
    "FIL-USD": "Filecoin", "ALGO-USD": "Algorand", "VET-USD": "VeChain",
    "ICP-USD": "Internet Computer", "HBAR-USD": "Hedera",
    "AAVE-USD": "Aave", "MKR-USD": "Maker",
    "INJ-USD": "Injective", "TIA-USD": "Celestia",
    "RENDER-USD": "Render", "FET-USD": "Fetch.ai", "PEPE24478-USD": "Pepe",
    "WIF-USD": "Dogwifhat", "BONK-USD": "Bonk", "FLOKI-USD": "Floki",
    "SEI-USD": "Sei",
    "RUNE-USD": "THORChain", "PENDLE-USD": "Pendle", "JUP-USD": "Jupiter",
    "WLD-USD": "Worldcoin", "ONDO-USD": "Ondo", "JASMY-USD": "Jasmy",
    "KAS-USD": "Kaspa",
    # === US TECH MEGA (30) ===
    "MSFT": "Microsoft", "AAPL": "Apple", "NVDA": "Nvidia",
    "TSLA": "Tesla", "AMZN": "Amazon", "META": "Meta",
    "GOOGL": "Google", "AMD": "AMD", "NFLX": "Netflix",
    "COIN": "Coinbase", "CRM": "Salesforce", "ORCL": "Oracle",
    "UBER": "Uber", "SHOP": "Shopify",
    "PLTR": "Palantir", "SNOW": "Snowflake", "MU": "Micron",
    "INTC": "Intel", "PYPL": "PayPal", "ADBE": "Adobe",
    "AVGO": "Broadcom", "QCOM": "Qualcomm", "TXN": "Texas Instruments",
    "PANW": "Palo Alto", "CRWD": "CrowdStrike", "DDOG": "Datadog",
    "ZS": "Zscaler", "NET": "Cloudflare", "MRVL": "Marvell",
    "ARM": "ARM Holdings",
    # === US FINANCE (20) ===
    "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "V": "Visa", "MA": "Mastercard", "BRK-B": "Berkshire",
    "MS": "Morgan Stanley", "WFC": "Wells Fargo", "C": "Citigroup",
    "SCHW": "Schwab", "AXP": "American Express", "BLK": "BlackRock",
    "ICE": "Intercontinental", "CME": "CME Group", "SPGI": "S&P Global",
    "MCO": "Moody's", "AON": "Aon", "TFC": "Truist",
    "USB": "US Bancorp", "PNC": "PNC Financial",
    # === US SANTE (20) ===
    "JNJ": "J&J", "UNH": "UnitedHealth", "PFE": "Pfizer",
    "ABBV": "AbbVie", "LLY": "Eli Lilly", "MRK": "Merck",
    "TMO": "Thermo Fisher", "ABT": "Abbott", "DHR": "Danaher",
    "BMY": "Bristol-Myers", "AMGN": "Amgen", "GILD": "Gilead",
    "ISRG": "Intuitive Surg.", "VRTX": "Vertex", "REGN": "Regeneron",
    "MDT": "Medtronic", "SYK": "Stryker", "BSX": "Boston Scientific",
    "EW": "Edwards Life", "ZTS": "Zoetis",
    # === US CONSO & INDUSTRIE (25) ===
    "DIS": "Disney", "KO": "Coca-Cola", "PEP": "PepsiCo",
    "WMT": "Walmart", "COST": "Costco", "HD": "Home Depot",
    "MCD": "McDonald's", "SBUX": "Starbucks", "NKE": "Nike",
    "LOW": "Lowe's", "TGT": "Target", "TJX": "TJ Maxx",
    "PG": "Procter & Gamble", "CL": "Colgate", "EL": "Estee Lauder",
    "BA": "Boeing", "CAT": "Caterpillar", "GE": "GE Aerospace",
    "HON": "Honeywell", "UPS": "UPS", "FDX": "FedEx",
    "RTX": "RTX", "LMT": "Lockheed Martin", "DE": "Deere",
    "MMM": "3M",
    # === US ENERGY & UTILITIES (10) ===
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "SLB": "Schlumberger", "EOG": "EOG Resources", "MPC": "Marathon Petro",
    "NEE": "NextEra Energy", "DUK": "Duke Energy", "SO": "Southern Co",
    "D": "Dominion Energy",
    # === EUROPE (20) ===
    "ASML": "ASML", "SAP": "SAP", "NVO": "Novo Nordisk",
    "TTE": "TotalEnergies", "SNY": "Sanofi", "SHEL": "Shell",
    "UL": "Unilever", "AZN": "AstraZeneca", "DEO": "Diageo",
    "SPOT": "Spotify", "RACE": "Ferrari", "LOGI": "Logitech",
    "STM": "STMicro", "SIE.DE": "Siemens", "ALV.DE": "Allianz",
    "BAS.DE": "BASF", "BMW.DE": "BMW", "VOW3.DE": "Volkswagen",
    "MC.PA": "LVMH", "OR.PA": "L'Oreal",
    # === ASIE & EMERGENTS (15) ===
    "TSM": "TSMC", "BABA": "Alibaba", "JD": "JD.com",
    "BIDU": "Baidu", "PDD": "PDD Holdings", "NIO": "NIO",
    "XPEV": "XPeng", "LI": "Li Auto", "GRAB": "Grab",
    "SE": "Sea Limited", "MELI": "MercadoLibre", "NU": "Nu Holdings",
    "INFY": "Infosys", "WIT": "Wipro", "HDB": "HDFC Bank",
    # === ETFs MAJEURS (15) ===
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "IWM": "Russell 2000 ETF",
    "DIA": "Dow Jones ETF", "GLD": "Gold ETF", "SLV": "Silver ETF",
    "USO": "Oil ETF", "TLT": "Bonds 20Y ETF", "XLF": "Finance ETF",
    "XLK": "Tech ETF", "XLE": "Energy ETF", "XLV": "Health ETF",
    "ARKK": "ARK Innovation", "SOXX": "Semicon ETF", "KWEB": "China Tech ETF",
    # === FOREX (25) ===
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD", "USDCHF=X": "USD/CHF",
    "NZDUSD=X": "NZD/USD", "EURJPY=X": "EUR/JPY", "GBPJPY=X": "GBP/JPY",
    "EURGBP=X": "EUR/GBP", "EURCAD=X": "EUR/CAD", "AUDJPY=X": "AUD/JPY",
    "USDMXN=X": "USD/MXN", "USDZAR=X": "USD/ZAR", "EURCHF=X": "EUR/CHF",
    "USDTRY=X": "USD/TRY", "USDSGD=X": "USD/SGD", "USDHKD=X": "USD/HKD",
    "USDNOK=X": "USD/NOK", "USDSEK=X": "USD/SEK", "USDPLN=X": "USD/PLN",
    "GBPCHF=X": "GBP/CHF", "AUDNZD=X": "AUD/NZD", "CADJPY=X": "CAD/JPY",
    "CHFJPY=X": "CHF/JPY",
    # === COMMODITIES (12) ===
    "GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil",
    "NG=F": "Natural Gas", "PL=F": "Platinum", "PA=F": "Palladium",
    "HG=F": "Copper", "ZC=F": "Corn", "ZW=F": "Wheat",
    "ZS=F": "Soybean", "KC=F": "Coffee", "CC=F": "Cocoa",
    # === INDICES MONDIAUX (15) ===
    "^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^DJI": "Dow Jones",
    "^RUT": "Russell 2000", "^VIX": "VIX",
    "^FTSE": "FTSE 100", "^GDAXI": "DAX", "^FCHI": "CAC 40",
    "^N225": "Nikkei", "^HSI": "Hang Seng", "^STOXX50E": "Euro Stoxx 50",
    "^BVSP": "Bovespa", "^GSPTSE": "TSX Canada",
    "^NSEI": "Nifty 50", "^KS11": "KOSPI",
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
    "modele": "Deep Ensemble (LSTM+GRU+RF+LightGBM+XGBoost+GBM)",
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
        "circuit_breaker": "-8% DD, recovery 95%",
        "stoploss_guard": "3 pertes consecutives",
        "cooldown": "300s par actif",
        "fee_filter": "gain > 2x frais",
        "atr_sizing": "2% risk / (ATR * 1.5)",
        "sl_structurel": "swing low 15 bougies",
        "trailing_paliers": "avance 1.75x risk",
        "take_profit": "swing high ou +8%",
        "daily_limits": "perte -3$ / gain +15$ jour",
        "equity_tasks": "coupe pire si DD>-6%, ferme tout si +20%",
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

market_intel = MarketIntelligence(ACTIFS)
social_intel = SocialIntelligence()
deep_models = {}
DEEP_RETRAIN_INTERVAL = 1800

cortex_logs = []
MAX_CORTEX_LOGS = 200

def cortex_log(level, module, message):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = {"ts": ts, "level": level, "module": module, "msg": message}
    cortex_logs.append(entry)
    if len(cortex_logs) > MAX_CORTEX_LOGS:
        del cortex_logs[:len(cortex_logs) - MAX_CORTEX_LOGS]

ohlcv_cache = {}
lock = threading.Lock()


def charger_historique():
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def charger_etat():
    if ETAT_PATH.exists():
        try:
            with open(ETAT_PATH) as f:
                saved = json.load(f)
            keys_to_restore = [
                "capital", "capital_initial", "cash", "valeur_positions",
                "pnl", "pnl_pct", "drawdown", "plus_haut",
                "nb_trades", "trades_gagnants", "trades_perdants",
                "gain_total", "perte_totale",
                "positions", "equity", "timestamps_equity",
            ]
            for k in keys_to_restore:
                if k in saved:
                    etat[k] = saved[k]
            print(f"[STATE] Etat restaure: ${etat['capital']:.2f} capital, {sum(1 for p in etat['positions'].values() if p.get('en_position'))} positions actives")
            return True
        except Exception as e:
            print(f"[STATE] Erreur chargement etat: {e}")
    return False


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


def verifier_tp_trailing(sym, nom, prix):
    pos = etat["positions"].get(sym, {})
    if not pos.get("en_position"):
        return
    tp = pos.get("take_profit")
    sl = pos.get("stop_loss")
    entry = pos.get("prix_entree", 0)
    if not entry:
        return
    pnl_pct = (prix - entry) / entry * 100
    if pos.get("pic_prix"):
        pos["pic_prix"] = max(pos["pic_prix"], prix)
    else:
        pos["pic_prix"] = prix
    if pnl_pct >= 2.0 and not pos.get("trailing_active"):
        pos["trailing_active"] = True
        pos["stop_loss"] = round(entry * 1.005, 4)
        cortex_log("ML", "TRAILING", f"{nom}: Trailing actif, SL releve a breakeven+0.5% ({pos['stop_loss']:.2f})")
    if pos.get("trailing_active") and pos["pic_prix"] > 0:
        sl_initial = pos.get("sl_initial", entry * 0.95)
        nouveau_sl = trailing_paliers(entry, sl_initial, prix, palier_ratio=1.75)
        trail_sl_pct = round(pos["pic_prix"] * 0.97, 4)
        nouveau_sl = max(nouveau_sl, trail_sl_pct)
        if nouveau_sl > pos.get("stop_loss", 0):
            pos["stop_loss"] = nouveau_sl
    if tp and prix >= tp:
        now = datetime.now(timezone.utc)
        valeur = pos["quantite"] * prix
        pnl = valeur - pos["montant_entree"]
        etat["cash"] += valeur
        etat["trades_gagnants"] += 1
        etat["gain_total"] += pnl
        etat["positions"][sym] = {"en_position": False}
        etat["trades"].append({
            "date": now.strftime("%m-%d %H:%M"),
            "sym": sym, "nom": nom,
            "action": "TAKE-PROFIT",
            "prix": round(prix, 2),
            "montant": round(valeur, 2),
            "pnl": round(pnl, 2),
            "capital": round(etat["cash"], 2),
        })
        protections.apres_trade(sym, pnl)
        cortex_log("BUY", "RISK", f"{nom} ({sym}): TAKE-PROFIT @ {prix:.2f} | PnL: +{pnl:.2f}$ | +{pnl_pct:.1f}%")
        print(f"  [TAKE-PROFIT] {nom}: +{pnl:.2f}$ a {prix:.2f}")
    if sl and prix <= sl and not (tp and prix >= tp):
        now = datetime.now(timezone.utc)
        valeur = pos["quantite"] * prix
        pnl = valeur - pos["montant_entree"]
        etat["cash"] += valeur
        if pnl >= 0:
            etat["trades_gagnants"] += 1
            etat["gain_total"] += pnl
        else:
            etat["trades_perdants"] += 1
            etat["perte_totale"] += abs(pnl)
        raison = "Trailing SL" if pos.get("trailing_active") else "Stop-Loss -5%"
        etat["positions"][sym] = {"en_position": False}
        etat["trades"].append({
            "date": now.strftime("%m-%d %H:%M"),
            "sym": sym, "nom": nom,
            "action": "STOP-LOSS",
            "prix": round(prix, 2),
            "montant": round(valeur, 2),
            "pnl": round(pnl, 2),
            "capital": round(etat["cash"], 2),
            "raison": raison,
        })
        protections.apres_trade(sym, pnl)
        cortex_log("STOP", "RISK", f"{nom} ({sym}): {raison} @ {prix:.2f} | PnL: {pnl:+.2f}$")
        print(f"  [{raison.upper()}] {nom}: {pnl:+.2f}$ a {prix:.2f}")


def executer_stop_loss(sym, nom, prix):
    pos = etat["positions"].get(sym, {})
    if not pos.get("en_position"):
        return
    protections.stop_technique.mettre_a_jour_prix(sym, prix)
    stop, raison = protections.stop_technique.verifier_stop(sym, prix)
    if not stop:
        return
    now = datetime.now(timezone.utc)
    valeur = pos["quantite"] * prix
    pnl = valeur - pos["montant_entree"]
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
        "action": "STOP-LOSS",
        "prix": round(prix, 2),
        "montant": round(valeur, 2),
        "pnl": round(pnl, 2),
        "capital": round(etat["cash"], 2),
        "raison": raison,
    })
    protections.apres_trade(sym, pnl)
    protections.stop_technique.fermer_position(sym)
    cortex_log("STOP", "RISK", f"{nom} ({sym}): STOP-LOSS @ {prix:.2f} | PnL: {pnl:+.2f}$ | {raison}")
    print(f"  [STOP-LOSS] {nom}: VENTE FORCEE a {prix:.2f} | PnL: {pnl:+.2f}$ | {raison}")


def _fetch_un_prix(sym, nom):
    try:
        t = yf.Ticker(sym)
        h = t.history(period="1d", interval="1m")
        if h.empty:
            h = t.history(period="5d", interval="5m")
        if h.empty:
            return None
        return (sym, nom, float(h["Close"].iloc[-1]))
    except Exception as e:
        return None


def fetch_prix():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = {executor.submit(_fetch_un_prix, sym, nom): (sym, nom) for sym, nom in ACTIFS.items()}
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            sym, nom, prix = result

            if sym not in etat["signaux"]:
                etat["signaux"][sym] = {"nom": nom, "decision": "ATTENTE", "exposition": 0, "prix": prix}
            etat["signaux"][sym]["prix"] = round(prix, 2)

            if sym not in etat["prix_historique"]:
                etat["prix_historique"][sym] = []
            etat["prix_historique"][sym].append([now, round(prix, 2)])
            if len(etat["prix_historique"][sym]) > 300:
                etat["prix_historique"][sym] = etat["prix_historique"][sym][-300:]

            executer_stop_loss(sym, nom, prix)
            verifier_tp_trailing(sym, nom, prix)

    with lock:
        calculer_valeur_totale()
        etat["timestamp"] = now
        etat["nb_actifs"] = len(ACTIFS)
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
                    cortex_log("OPT", "HYPEROPT", f"{nom}: Best={best_h.nom} LL={best_h.log_loss_val} Acc={best_h.accuracy_val}%")
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

    cortex_log("INFO", "CORTEX", f"Cycle d'analyse demarre | Cash: {etat['cash']:.2f}$ | Capital: {etat['capital']:.2f}$")
    print(f"\n{'='*50}")
    print(f"[TRADE] Analyse a {now.strftime('%H:%M:%S')} UTC")
    print(f"  Cash: {etat['cash']:.2f}$ | Capital: {etat['capital']:.2f}$")

    peut, raison_block = protections.peut_trader(
        "", etat.get("drawdown", 0.0),
        capital=etat.get("capital", 0.0), plus_haut=etat.get("plus_haut", 0.0))
    if not peut:
        print(f"  PROTECTIONS ACTIVES: {raison_block}")
        cortex_log("WARN", "PROTECT", raison_block)
        etat["protection_active"] = raison_block
        with lock:
            sauver()
        return
    etat["protection_active"] = ""

    eq_action = protections.equity_tasks.evaluer(
        etat.get("drawdown", 0.0), etat.get("pnl_pct", 0.0),
        etat["positions"], etat["signaux"])
    if eq_action == "__FERMER_TOUT__":
        cortex_log("BUY", "EQUITY", f"Profit >= {protections.equity_tasks.profit_ferme_tout}% — fermeture de toutes les positions!")
        for sym_close, pos_close in list(etat["positions"].items()):
            if pos_close.get("en_position"):
                sig = etat["signaux"].get(sym_close, {})
                px = sig.get("prix", pos_close.get("prix_entree", 0))
                val = pos_close["quantite"] * px
                pnl_close = val - pos_close["montant_entree"]
                etat["cash"] += val
                if pnl_close >= 0:
                    etat["trades_gagnants"] += 1
                    etat["gain_total"] += pnl_close
                else:
                    etat["trades_perdants"] += 1
                    etat["perte_totale"] += abs(pnl_close)
                etat["positions"][sym_close] = {"en_position": False}
                etat["trades"].append({"date": now.strftime("%m-%d %H:%M"), "sym": sym_close,
                    "nom": ACTIFS.get(sym_close, sym_close), "action": "TAKE-PROFIT",
                    "prix": round(px, 2), "montant": round(val, 2),
                    "pnl": round(pnl_close, 2), "capital": round(etat["cash"], 2)})
                protections.apres_trade(sym_close, pnl_close)
        with lock:
            calculer_valeur_totale()
            sauver()
        return
    elif eq_action and eq_action in etat["positions"]:
        pos_pire = etat["positions"][eq_action]
        if pos_pire.get("en_position"):
            sig = etat["signaux"].get(eq_action, {})
            px = sig.get("prix", pos_pire.get("prix_entree", 0))
            val = pos_pire["quantite"] * px
            pnl_close = val - pos_pire["montant_entree"]
            etat["cash"] += val
            etat["trades_perdants"] += 1
            etat["perte_totale"] += abs(pnl_close)
            etat["positions"][eq_action] = {"en_position": False}
            etat["trades"].append({"date": now.strftime("%m-%d %H:%M"), "sym": eq_action,
                "nom": ACTIFS.get(eq_action, eq_action), "action": "STOP-LOSS",
                "prix": round(px, 2), "montant": round(val, 2),
                "pnl": round(pnl_close, 2), "capital": round(etat["cash"], 2),
                "raison": "EquityTask: coupe pire perdant"})
            protections.apres_trade(eq_action, pnl_close)
            cortex_log("STOP", "EQUITY", f"Coupe {eq_action} (pire perdant) DD={etat.get('drawdown', 0):.1f}%")

    def _download_hist(sym):
        try:
            return (sym, yf.Ticker(sym).history(period="2y", interval="1d"))
        except:
            return (sym, pd.DataFrame())

    print(f"  Telechargement parallele de {len(ACTIFS)} actifs...")
    hist_cache = {}
    with ThreadPoolExecutor(max_workers=40) as executor:
        for sym, hist in executor.map(lambda s: _download_hist(s), ACTIFS.keys()):
            if not hist.empty and len(hist) >= 300:
                hist_cache[sym] = hist
    print(f"  {len(hist_cache)}/{len(ACTIFS)} actifs avec donnees suffisantes")

    for sym, nom in ACTIFS.items():
        try:
            peut_sym, raison_sym = protections.peut_trader(
                sym, etat.get("drawdown", 0.0),
                capital=etat.get("capital", 0.0), plus_haut=etat.get("plus_haut", 0.0))
            if not peut_sym:
                continue

            hist = hist_cache.get(sym)
            if hist is None:
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
                growth_ratio = etat["capital"] / CAPITAL_INIT
                if growth_ratio >= 1.0:
                    kelly = min(1.0 + (growth_ratio - 1.0) * 0.5, 2.0)
                else:
                    kelly = max(0.5, growth_ratio)
                intel_boost = market_intel.get_boost_dynamique(sym) if market_intel.last_scan else 1.0
                social_boost = social_intel.get_social_boost(sym) if social_intel.last_scan else 1.0
                boost = MARCHES_PRIORITAIRES.get(sym, 1.0) * intel_boost * social_boost * kelly

                candles = ohlcv_cache.get(f"{sym}_5d_5m", {}).get("data", [])
                highs = [c["h"] for c in candles] if candles else []
                lows = [c["l"] for c in candles] if candles else []
                closes = [c["c"] for c in candles] if candles else []

                atr = calculer_atr(highs, lows, closes, period=14)
                if atr > 0:
                    montant = position_sizing_atr(
                        etat["cash"], risk_pct=2.0 * boost, prix_entree=prix_actuel,
                        atr=atr, multiplier=1.5)
                else:
                    alloc = min(BASE_ALLOC * boost, MAX_ALLOC)
                    montant = etat["cash"] * alloc

                sl_struct = calculer_sl_structurel(lows, n=15) if lows else 0
                tp_struct = calculer_tp_structurel(highs, n=15) if highs else 0

                sl_prix = sl_struct if sl_struct > 0 and sl_struct < prix_actuel * 0.98 else round(prix_actuel * 0.95, 4)
                tp_prix = tp_struct if tp_struct > prix_actuel * 1.02 else round(prix_actuel * 1.08, 4)

                if montant >= 1:
                    qte = montant / prix_actuel
                    etat["positions"][sym] = {
                        "en_position": True,
                        "prix_entree": prix_actuel,
                        "quantite": qte,
                        "montant_entree": round(montant, 2),
                        "date": now.strftime("%Y-%m-%d %H:%M"),
                        "stop_loss": round(sl_prix, 4),
                        "take_profit": round(tp_prix, 4),
                        "trailing_active": False,
                        "pic_prix": prix_actuel,
                        "atr": round(atr, 6),
                        "sl_initial": round(sl_prix, 4),
                    }
                    etat["cash"] -= montant
                    etat["nb_trades"] += 1
                    protections.stop_technique.ouvrir_position(sym, prix_actuel)
                    etat["trades"].append({
                        "date": now.strftime("%m-%d %H:%M"),
                        "sym": sym, "nom": nom,
                        "action": "ACHAT",
                        "prix": round(prix_actuel, 2),
                        "montant": round(montant, 2),
                        "capital": round(etat["cash"], 2),
                    })
                    protections.apres_trade(sym, 0.0)
                    sl_pct = abs((sl_prix - prix_actuel) / prix_actuel * 100)
                    cortex_log("BUY", "TRADE", f"{nom} ({sym}): ACHAT {montant:.2f}$ @ {prix_actuel:.2f} | ATR={atr:.4f} SL={sl_prix:.2f}(-{sl_pct:.1f}%) TP={tp_prix:.2f} | K={kelly:.2f}")
                    print(f"  {nom}: ACHAT {montant:.2f}$ a {prix_actuel:.2f} (ATR sizing, SL struct={sl_prix:.2f}, TP={tp_prix:.2f})")
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
                protections.stop_technique.fermer_position(sym)
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
                cortex_log("SELL", "TRADE", f"{nom} ({sym}): VENTE {valeur:.2f}$ | PnL: {pnl:+.2f}$ | Proba {proba_pct:.0f}% <= {seuil_v}%")
                print(f"  {nom}: VENTE {valeur:.2f}$ (PnL: {pnl:+.2f}$)")

            else:
                cortex_log("HOLD", "ANALYSE", f"{nom}: {decision} @ {prix_actuel:.2f} | Proba {proba_pct:.0f}% (A>{seuil_a}% V<{seuil_v}%)")
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


def loop_intel():
    time.sleep(10)
    while True:
        try:
            cortex_log("INFO", "INTEL", "Scan momentum marches en cours...")
            results = market_intel.scanner_momentum()
            top = market_intel.top_movers
            top_name = top[0]["nom"] if top else "N/A"
            top_mom = top[0]["momentum"] if top else 0
            cortex_log("INFO", "INTEL", f"Scan OK: {len(results)} actifs | Top={top_name} ({top_mom:+.1f}) | Secteurs={market_intel.sector_momentum}")
            opps = market_intel.get_top_opportunities(5)
            if opps:
                noms = ", ".join(o["nom"] for o in opps)
                cortex_log("ML", "INTEL", f"Opportunites: {noms}")
        except Exception as e:
            cortex_log("ERR", "INTEL", f"Erreur scan: {e}")
        time.sleep(180)


def loop_social():
    time.sleep(30)
    while True:
        try:
            cortex_log("INFO", "SOCIAL", "Scan Reddit/Twitter en cours...")
            sentiments = social_intel.scan_reddit()
            trending = social_intel.trending
            if trending:
                top3 = ", ".join(f"{s[0]}({s[1]['label']})" for s in trending[:3])
                cortex_log("ML", "SOCIAL", f"Trending: {top3} | {len(sentiments)} tickers detectes sur {social_intel.to_dict()['total_posts']} posts")
            else:
                cortex_log("INFO", "SOCIAL", "Aucun ticker detecte dans les posts")
        except Exception as e:
            cortex_log("ERR", "SOCIAL", f"Erreur scan social: {e}")
        time.sleep(300)


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

        if path == "/api/terminal":
            self._json_response({"logs": list(cortex_logs)})
            return

        if path == "/api/stats":
            with lock:
                positions = etat.get("positions", {})
                actives = {s: p for s, p in positions.items() if p.get("en_position")}
                stats = {
                    "capital": etat.get("capital", CAPITAL_INIT),
                    "cash": etat.get("cash", CAPITAL_INIT),
                    "pnl": etat.get("pnl", 0),
                    "pnl_pct": etat.get("pnl_pct", 0),
                    "drawdown": etat.get("drawdown", 0),
                    "nb_trades": etat.get("nb_trades", 0),
                    "trades_gagnants": etat.get("trades_gagnants", 0),
                    "trades_perdants": etat.get("trades_perdants", 0),
                    "gain_total": etat.get("gain_total", 0),
                    "perte_totale": etat.get("perte_totale", 0),
                    "win_rate": round(etat["trades_gagnants"] / max(1, etat["nb_trades"]) * 100, 1),
                    "positions_actives": len(actives),
                    "equity": etat.get("equity", []),
                    "timestamps_equity": etat.get("timestamps_equity", []),
                    "signaux": etat.get("signaux", {}),
                    "positions": positions,
                    "secteurs": {},
                }
                crypto = forex = actions = indices = 0
                for sym in actives:
                    if sym.endswith("-USD") or sym.endswith("-EUR"):
                        crypto += 1
                    elif "=" in sym:
                        forex += 1
                    elif sym.startswith("^"):
                        indices += 1
                    else:
                        actions += 1
                stats["secteurs"] = {"Crypto": crypto, "Forex": forex, "Actions": actions, "Indices": indices}
            self._json_response(stats)
            return

        if path == "/api/protections":
            with lock:
                p = protections
                dd = etat.get("drawdown", 0)
                cap = etat.get("capital", CAPITAL_INIT)
                ph = etat.get("plus_haut", CAPITAL_INIT)
                cb_bloque, cb_raison = p.circuit_breaker.est_bloque(
                    drawdown_pct=dd, capital=cap, plus_haut=ph)
                dl_bloque, dl_raison = p.daily_limits.est_bloque()
                sg_bloque, sg_raison = p.stoploss.est_bloque()
                sf_bloque, sf_raison = p.session_filter.est_bloque()
                pnl_jour = sum(p.daily_limits._trades_jour) if p.daily_limits._trades_jour else 0
                data = {
                    "circuit_breaker": {"actif": True, "bloque": cb_bloque, "raison": cb_raison,
                                        "tripped": p.circuit_breaker._tripped},
                    "daily_limits": {"actif": True, "bloque": dl_bloque, "raison": dl_raison,
                                     "pnl_jour": round(pnl_jour, 2),
                                     "max_perte": p.daily_limits.max_perte_jour,
                                     "max_gain": p.daily_limits.max_gain_jour},
                    "stoploss_guard": {"actif": True, "bloque": sg_bloque, "raison": sg_raison,
                                       "pertes_consec": p.stoploss._pertes_consecutives},
                    "session_filter": {"actif": p.session_filter.actif, "bloque": sf_bloque, "raison": sf_raison},
                    "atr_sizing": {"actif": True, "risk_pct": 2.0, "multiplier": 1.5},
                    "trailing_paliers": {"actif": True, "palier_ratio": 1.75},
                    "drawdown": round(dd, 2),
                    "capital": round(cap, 2),
                    "plus_haut": round(ph, 2),
                }
            self._json_response(data)
            return

        if path == "/api/intel":
            data = market_intel.to_dict()
            data["opportunities"] = market_intel.get_top_opportunities(10)
            self._json_response(data)
            return

        if path == "/api/social":
            data = social_intel.to_dict()
            self._json_response(data)
            return

        if path == "/api/deep":
            breakdown = {}
            for sym, model in deep_models.items():
                if model.trained:
                    nom = ACTIFS.get(sym, sym)
                    prices = etat.get("prix_historique", {}).get(sym, [])
                    if len(prices) > 60:
                        price_vals = [p[1] for p in prices]
                        pred = model.predict(price_vals)
                        bd = model.get_model_breakdown(price_vals)
                        breakdown[sym] = {"nom": nom, "prediction": pred, "models": bd}
            self._json_response({"models": breakdown, "total_trained": len(deep_models)})
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

        if path == "/old" or path == "/v1":
            self.path = "/dashboard_mega.html"
            super().do_GET()
            return

        if path == "/" or path == "" or path == "/mt" or path == "/terminal":
            self.path = "/dashboard_mt.html"
        super().do_GET()

    def log_message(self, fmt, *args):
        pass


def main():
    ETAT_PATH.parent.mkdir(exist_ok=True)
    charger_etat()
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

    threading.Thread(target=loop_intel, daemon=True).start()
    print("[INTEL] Market Intelligence: scan toutes les 3 minutes")

    threading.Thread(target=loop_social, daemon=True).start()
    print("[SOCIAL] Reddit/Twitter sentiment: scan toutes les 5 minutes")

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
