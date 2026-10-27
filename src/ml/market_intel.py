"""
Rex Clemens — Market Intelligence Engine
==========================================
Scrape en continu les donnees web pour detecter :
- Top movers du jour (momentum)
- Volume anormal (breakout signals)
- Tendance sectorielle
- Score de momentum composite

Utilise yfinance (pas d'API key necessaire).
"""

import threading
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import yfinance as yf


class MarketIntelligence:
    def __init__(self, symbols_dict):
        self.symbols = symbols_dict
        self.intel = {}
        self.top_movers = []
        self.sector_momentum = {}
        self.last_scan = None
        self.scan_interval = 180

    def scanner_momentum(self):
        results = {}

        def _scan_one(sym, nom):
            try:
                t = yf.Ticker(sym)
                h = t.history(period="5d", interval="1h")
                if h.empty or len(h) < 10:
                    return None

                closes = h["Close"].values
                volumes = h["Volume"].values

                chg_1h = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 else 0
                chg_4h = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
                chg_24h = (closes[-1] - closes[-24]) / closes[-24] * 100 if len(closes) >= 24 else 0

                vol_avg = np.mean(volumes[-24:]) if len(volumes) >= 24 else np.mean(volumes)
                vol_now = np.mean(volumes[-3:]) if len(volumes) >= 3 else volumes[-1]
                vol_ratio = vol_now / max(vol_avg, 1)

                ema_fast = self._ema(closes, 9)
                ema_slow = self._ema(closes, 21)
                trend = 1 if ema_fast > ema_slow else -1

                rsi = self._rsi(closes, 14)

                momentum_score = (
                    chg_1h * 0.15 +
                    chg_4h * 0.25 +
                    chg_24h * 0.30 +
                    (vol_ratio - 1) * 10 * 0.15 +
                    (rsi - 50) * 0.15
                )

                return {
                    "sym": sym, "nom": nom,
                    "prix": round(float(closes[-1]), 4),
                    "chg_1h": round(chg_1h, 2),
                    "chg_4h": round(chg_4h, 2),
                    "chg_24h": round(chg_24h, 2),
                    "vol_ratio": round(vol_ratio, 2),
                    "trend": trend,
                    "rsi": round(rsi, 1),
                    "momentum": round(momentum_score, 2),
                }
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=15) as ex:
            futures = {ex.submit(_scan_one, s, n): s for s, n in self.symbols.items()}
            for f in as_completed(futures):
                r = f.result()
                if r:
                    results[r["sym"]] = r

        self.intel = results
        self.top_movers = sorted(
            results.values(),
            key=lambda x: abs(x["momentum"]),
            reverse=True
        )[:20]

        sectors = {"crypto": [], "stock": [], "forex": [], "commodity": [], "index": []}
        for sym, data in results.items():
            if sym.endswith("-USD") or sym.endswith("-EUR"):
                sectors["crypto"].append(data["momentum"])
            elif "=" in sym:
                sectors["forex"].append(data["momentum"])
            elif sym.startswith("^"):
                sectors["index"].append(data["momentum"])
            elif sym.endswith("=F"):
                sectors["commodity"].append(data["momentum"])
            else:
                sectors["stock"].append(data["momentum"])

        self.sector_momentum = {
            k: round(np.mean(v), 2) if v else 0
            for k, v in sectors.items()
        }

        self.last_scan = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return results

    def get_boost_dynamique(self, sym):
        data = self.intel.get(sym)
        if not data:
            return 1.0
        m = data["momentum"]
        vr = data["vol_ratio"]
        boost = 1.0
        if m > 5:
            boost += 0.3
        elif m > 2:
            boost += 0.15
        elif m < -5:
            boost -= 0.2
        if vr > 2.0:
            boost += 0.2
        elif vr > 1.5:
            boost += 0.1
        return round(max(0.5, min(2.0, boost)), 2)

    def get_top_opportunities(self, n=10):
        return [
            m for m in self.top_movers
            if m["momentum"] > 0 and m["vol_ratio"] > 1.0 and m["rsi"] < 75
        ][:n]

    def to_dict(self):
        return {
            "last_scan": self.last_scan,
            "top_movers": self.top_movers[:10],
            "sector_momentum": self.sector_momentum,
            "total_scanned": len(self.intel),
        }

    @staticmethod
    def _ema(data, period):
        if len(data) < period:
            return float(data[-1]) if len(data) > 0 else 0
        k = 2 / (period + 1)
        ema = float(data[0])
        for v in data[1:]:
            ema = float(v) * k + ema * (1 - k)
        return ema

    @staticmethod
    def _rsi(data, period=14):
        if len(data) < period + 1:
            return 50.0
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - 100 / (1 + rs), 1)
