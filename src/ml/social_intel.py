"""
Rex Clemens — Social Intelligence Engine
==========================================
Scrape Reddit + Twitter/X pour sentiment de marche.
Pas de cle API necessaire — endpoints publics uniquement.

Subreddits : r/cryptocurrency, r/wallstreetbets, r/stocks, r/forex, r/options
Twitter/X : via Nitter RSS (public, pas d'auth)
Google Trends : pytrends (pas d'auth)
"""

import re
import time
import json
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import URLError

REDDIT_SUBS = [
    "cryptocurrency", "wallstreetbets", "stocks", "forex",
    "options", "CryptoMarkets", "investing", "daytrading",
    "bitcoinmarkets", "ethtrader",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

SENTIMENT_WORDS = {
    "bullish": 2, "moon": 2, "rocket": 2, "pump": 1.5, "buy": 1, "long": 1,
    "breakout": 1.5, "rally": 1.5, "surge": 1.5, "soar": 1.5, "gain": 1,
    "green": 1, "ATH": 2, "undervalued": 1.5, "accumulate": 1, "hodl": 1.5,
    "bearish": -2, "crash": -2, "dump": -1.5, "sell": -1, "short": -1,
    "plunge": -1.5, "drop": -1, "red": -1, "overvalued": -1.5, "bubble": -1.5,
    "rug": -2, "scam": -2, "fear": -1.5, "panic": -1.5, "loss": -1,
}

TICKER_PATTERN = re.compile(r'\$([A-Z]{2,5})\b')
CRYPTO_ALIASES = {
    "bitcoin": "BTC-USD", "btc": "BTC-USD", "ethereum": "ETH-USD", "eth": "ETH-USD",
    "solana": "SOL-USD", "sol": "SOL-USD", "doge": "DOGE-USD", "dogecoin": "DOGE-USD",
    "xrp": "XRP-USD", "ripple": "XRP-USD", "cardano": "ADA-USD", "ada": "ADA-USD",
    "avalanche": "AVAX-USD", "avax": "AVAX-USD", "bnb": "BNB-USD",
    "polkadot": "DOT-USD", "dot": "DOT-USD", "chainlink": "LINK-USD", "link": "LINK-USD",
    "gold": "GC=F", "silver": "SI=F", "oil": "CL=F", "crude": "CL=F",
    "nvidia": "NVDA", "nvda": "NVDA", "tesla": "TSLA", "tsla": "TSLA",
    "apple": "AAPL", "aapl": "AAPL", "microsoft": "MSFT", "msft": "MSFT",
    "amazon": "AMZN", "amzn": "AMZN", "meta": "META", "google": "GOOGL",
    "palantir": "PLTR", "pltr": "PLTR", "amd": "AMD",
}

STOCK_TICKERS = {
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX",
    "COIN", "PLTR", "SHOP", "SNOW", "MU", "INTC", "PYPL", "SQ", "UBER",
    "JPM", "BAC", "GS", "V", "MA", "DIS", "KO", "PEP", "JNJ", "PFE",
    "LLY", "ASML", "SAP", "NVO", "SPOT", "CRM", "ORCL",
}


class SocialIntelligence:
    def __init__(self):
        self.mentions = {}
        self.sentiment_scores = {}
        self.trending = []
        self.reddit_data = []
        self.last_scan = None
        self.scan_interval = 300

    def _fetch_reddit_sub(self, sub):
        posts = []
        try:
            url = f"https://www.reddit.com/r/{sub}/hot/.rss?limit=25"
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=15) as resp:
                raw = resp.read().decode()
            import xml.etree.ElementTree as ET
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(raw)
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                content_el = entry.find("atom:content", ns)
                link_el = entry.find("atom:link", ns)
                title = title_el.text if title_el is not None and title_el.text else ""
                content = ""
                if content_el is not None and content_el.text:
                    text = re.sub(r"<[^>]+>", " ", content_el.text)
                    content = text[:500]
                href = link_el.get("href", "") if link_el is not None else ""
                posts.append({
                    "sub": sub,
                    "title": title,
                    "text": content,
                    "score": 1,
                    "comments": 0,
                    "created": 0,
                    "url": href,
                })
        except Exception:
            pass
        if not posts:
            try:
                url = f"https://old.reddit.com/r/{sub}/hot.json?limit=25"
                req = Request(url, headers=HEADERS)
                with urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())
                for child in data.get("data", {}).get("children", []):
                    post = child.get("data", {})
                    posts.append({
                        "sub": sub,
                        "title": post.get("title", ""),
                        "text": post.get("selftext", "")[:500],
                        "score": post.get("score", 0),
                        "comments": post.get("num_comments", 0),
                        "created": post.get("created_utc", 0),
                        "url": f"https://reddit.com{post.get('permalink', '')}",
                    })
            except Exception:
                pass
        return posts

    def _extract_tickers(self, text):
        text_lower = text.lower()
        found = set()
        for alias, sym in CRYPTO_ALIASES.items():
            if alias in text_lower:
                found.add(sym)
        for match in TICKER_PATTERN.findall(text):
            if match in STOCK_TICKERS:
                found.add(match)
        return found

    def _calc_sentiment(self, text):
        text_lower = text.lower()
        score = 0
        hits = 0
        for word, weight in SENTIMENT_WORDS.items():
            count = text_lower.count(word)
            if count:
                score += weight * count
                hits += count
        return round(score, 2), hits

    def scan_reddit(self):
        all_posts = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(self._fetch_reddit_sub, sub): sub for sub in REDDIT_SUBS}
            for f in as_completed(futures):
                posts = f.result()
                all_posts.extend(posts)

        mentions = {}
        sentiments = {}

        for post in all_posts:
            full_text = post["title"] + " " + post["text"]
            tickers = self._extract_tickers(full_text)
            sentiment, hits = self._calc_sentiment(full_text)
            engagement = post["score"] + post["comments"] * 2

            for sym in tickers:
                if sym not in mentions:
                    mentions[sym] = {"count": 0, "sentiment": 0, "engagement": 0, "posts": []}
                mentions[sym]["count"] += 1
                mentions[sym]["sentiment"] += sentiment
                mentions[sym]["engagement"] += engagement
                if len(mentions[sym]["posts"]) < 3:
                    mentions[sym]["posts"].append({
                        "title": post["title"][:80],
                        "sub": post["sub"],
                        "score": post["score"],
                        "sentiment": sentiment,
                    })

        for sym, data in mentions.items():
            avg_sent = data["sentiment"] / max(data["count"], 1)
            sentiments[sym] = {
                "mentions": data["count"],
                "sentiment": round(avg_sent, 2),
                "engagement": data["engagement"],
                "label": "BULLISH" if avg_sent > 1 else "BEARISH" if avg_sent < -1 else "NEUTRAL",
                "top_posts": data["posts"],
            }

        self.mentions = mentions
        self.sentiment_scores = sentiments
        self.reddit_data = all_posts[:50]

        self.trending = sorted(
            sentiments.items(),
            key=lambda x: x[1]["mentions"] * (1 + abs(x[1]["sentiment"])),
            reverse=True,
        )[:20]

        self.last_scan = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return sentiments

    def get_social_boost(self, sym):
        data = self.sentiment_scores.get(sym)
        if not data:
            return 1.0
        s = data["sentiment"]
        m = data["mentions"]
        boost = 1.0
        if s > 2 and m >= 3:
            boost = 1.3
        elif s > 1 and m >= 2:
            boost = 1.15
        elif s < -2 and m >= 3:
            boost = 0.7
        elif s < -1 and m >= 2:
            boost = 0.85
        return round(boost, 2)

    def to_dict(self):
        trending_list = []
        for sym, data in self.trending:
            trending_list.append({
                "sym": sym,
                "mentions": data["mentions"],
                "sentiment": data["sentiment"],
                "label": data["label"],
                "engagement": data["engagement"],
                "top_posts": data.get("top_posts", []),
            })
        return {
            "last_scan": self.last_scan,
            "trending": trending_list,
            "total_posts": len(self.reddit_data),
            "total_tickers": len(self.sentiment_scores),
            "subreddits": REDDIT_SUBS,
        }
