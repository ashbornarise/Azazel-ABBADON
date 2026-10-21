"""
REX CLEMENS — Exchange Intelligence (ccxt)
===========================================
Arme EXCLUSIVE de DeepSeek "Le Stratege".
Donnees d'exchange que yfinance ne fournit pas :
- Funding rates (crypto futures)
- Orderbook depth (pression achat/vente)
- Open interest (positions ouvertes)
- Ticker 24h (volume, VWAP, high/low)

Cortex n'a PAS acces a ces donnees. Avantage asymetrique.
Aucune cle API requise — donnees publiques uniquement.
"""

import time
from typing import Optional

try:
    import ccxt
    CCXT_DISPONIBLE = True
except ImportError:
    CCXT_DISPONIBLE = False


SYMBOLES_CRYPTO = {
    "BTC-USD": "BTC/USDT",
    "ETH-USD": "ETH/USDT",
    "SOL-USD": "SOL/USDT",
    "DOGE-USD": "DOGE/USDT",
    "XRP-USD": "XRP/USDT",
    "ADA-USD": "ADA/USDT",
    "AVAX-USD": "AVAX/USDT",
    "LINK-USD": "LINK/USDT",
    "DOT-USD": "DOT/USDT",
    "MATIC-USD": "MATIC/USDT",
    "SHIB-USD": "SHIB/USDT",
}


class ExchangeIntel:
    def __init__(self):
        self.disponible = CCXT_DISPONIBLE
        self.exchange = None
        self.cache = {}
        self.cache_ttl = 120

        if self.disponible:
            try:
                self.exchange = ccxt.binance({"enableRateLimit": True})
                print("[EXCHANGE] ccxt Binance connecte — donnees publiques")
            except Exception as e:
                print(f"[EXCHANGE] Erreur init ccxt: {e}")
                self.disponible = False
        else:
            print("[EXCHANGE] ccxt non installe — exchange intel desactive")

    def _cache_valide(self, key: str) -> bool:
        if key not in self.cache:
            return False
        return (time.time() - self.cache[key]["ts"]) < self.cache_ttl

    def analyser_crypto(self, sym_yf: str) -> Optional[dict]:
        """Analyse complete d'un actif crypto via exchange public data."""
        if not self.disponible or not self.exchange:
            return None

        sym_ccxt = SYMBOLES_CRYPTO.get(sym_yf)
        if not sym_ccxt:
            return None

        cache_key = f"crypto_{sym_yf}"
        if self._cache_valide(cache_key):
            return self.cache[cache_key]["data"]

        result = {}

        try:
            ticker = self.exchange.fetch_ticker(sym_ccxt)
            result["ticker"] = {
                "prix": ticker.get("last", 0),
                "volume_24h": round(ticker.get("quoteVolume", 0), 0),
                "change_24h_pct": round(ticker.get("percentage", 0), 2),
                "high_24h": ticker.get("high", 0),
                "low_24h": ticker.get("low", 0),
                "vwap": ticker.get("vwap", 0),
            }
        except Exception:
            pass

        try:
            book = self.exchange.fetch_order_book(sym_ccxt, limit=10)
            total_bids = sum(b[1] for b in book.get("bids", []))
            total_asks = sum(a[1] for a in book.get("asks", []))
            total = total_bids + total_asks
            result["orderbook"] = {
                "pression_achat_pct": round(total_bids / total * 100, 1) if total > 0 else 50,
                "pression_vente_pct": round(total_asks / total * 100, 1) if total > 0 else 50,
                "ratio_bid_ask": round(total_bids / max(total_asks, 1e-10), 2),
                "spread_pct": round((book["asks"][0][0] - book["bids"][0][0]) / book["bids"][0][0] * 100, 4) if book.get("asks") and book.get("bids") else 0,
            }
        except Exception:
            pass

        try:
            funding = self.exchange.fetch_funding_rate(sym_ccxt)
            result["funding"] = {
                "taux": round(funding.get("fundingRate", 0) * 100, 4),
                "sentiment": "longs dominant" if funding.get("fundingRate", 0) > 0.01 else (
                    "shorts dominant" if funding.get("fundingRate", 0) < -0.01 else "neutre"
                ),
            }
        except Exception:
            pass

        if result:
            self.cache[cache_key] = {"data": result, "ts": time.time()}

        return result if result else None

    def rapport_pour_deepseek(self, sym_yf: str) -> str:
        """Genere le texte a injecter dans le prompt DeepSeek."""
        data = self.analyser_crypto(sym_yf)
        if not data:
            return ""

        sections = []
        sections.append("\nDonnees Exchange LIVE (tes yeux sur le marche, Cortex est aveugle ici):")

        if "ticker" in data:
            t = data["ticker"]
            sections.append(f"- Volume 24h: ${t['volume_24h']:,.0f} | Change 24h: {t['change_24h_pct']}%")
            sections.append(f"- High/Low 24h: ${t['high_24h']:.2f} / ${t['low_24h']:.2f}")

        if "orderbook" in data:
            ob = data["orderbook"]
            sections.append(f"- Orderbook: Acheteurs {ob['pression_achat_pct']}% vs Vendeurs {ob['pression_vente_pct']}% | Ratio {ob['ratio_bid_ask']}")

        if "funding" in data:
            f = data["funding"]
            sections.append(f"- Funding rate: {f['taux']}% ({f['sentiment']})")

        return "\n".join(sections)
