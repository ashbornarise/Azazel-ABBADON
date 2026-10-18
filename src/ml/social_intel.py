"""
Rex Clemens — Social Intelligence Engine V2
=============================================
Sources gratuites (aucune cle API) :
1. Reddit RSS : r/cryptocurrency, r/wallstreetbets, r/stocks, r/forex...
2. News RSS : Yahoo Finance, MarketWatch, CoinDesk, Investing.com, Google News
3. Fear & Greed Index : alternative.me (crypto) + CNN (marche global)
4. Sentiment lexical enrichi sur titres + contenus

Scan toutes les 5 minutes. Score composite = Reddit + News + Fear & Greed.
"""

import re
import time
import json
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import URLError

REDDIT_SUBS = [
    "cryptocurrency", "wallstreetbets", "stocks", "forex",
    "options", "CryptoMarkets", "investing", "daytrading",
    "bitcoinmarkets", "ethtrader",
]

NEWS_RSS_FEEDS = [
    {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rssindex", "type": "general"},
    {"name": "Yahoo Finance Markets", "url": "https://finance.yahoo.com/rss/topfinstories", "type": "general"},
    {"name": "MarketWatch Top", "url": "https://feeds.marketwatch.com/marketwatch/topstories", "type": "general"},
    {"name": "MarketWatch Markets", "url": "https://feeds.marketwatch.com/marketwatch/marketpulse", "type": "general"},
    {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "type": "crypto"},
    {"name": "CoinTelegraph", "url": "https://cointelegraph.com/rss", "type": "crypto"},
    {"name": "Investing.com News", "url": "https://www.investing.com/rss/news.rss", "type": "general"},
    {"name": "Investing.com Forex", "url": "https://www.investing.com/rss/news_14.rss", "type": "forex"},
    {"name": "Investing.com Crypto", "url": "https://www.investing.com/rss/news_301.rss", "type": "crypto"},
    {"name": "Google News Business", "url": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en", "type": "general"},
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/xml, application/rss+xml, application/atom+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

SENTIMENT_WORDS = {
    # Bullish
    "bullish": 2, "moon": 2, "rocket": 2, "pump": 1.5, "buy": 1, "long": 1,
    "breakout": 1.5, "rally": 1.5, "surge": 1.5, "soar": 1.5, "gain": 1,
    "green": 1, "ATH": 2, "undervalued": 1.5, "accumulate": 1, "hodl": 1.5,
    "upside": 1, "upgrade": 1.5, "outperform": 1.5, "beat": 1, "record": 1,
    "recovery": 1, "boom": 1.5, "growth": 1, "strong": 1, "bullrun": 2,
    # Bearish
    "bearish": -2, "crash": -2, "dump": -1.5, "sell": -1, "short": -1,
    "plunge": -1.5, "drop": -1, "red": -1, "overvalued": -1.5, "bubble": -1.5,
    "rug": -2, "scam": -2, "fear": -1.5, "panic": -1.5, "loss": -1,
    "downgrade": -1.5, "underperform": -1.5, "miss": -1, "warning": -1,
    "recession": -2, "crisis": -2, "layoff": -1, "weak": -1, "correction": -1,
    "bankruptcy": -2, "default": -1.5, "inflation": -1, "tariff": -1,
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
    "sp500": "^GSPC", "s&p": "^GSPC", "nasdaq": "^IXIC", "dow": "^DJI",
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
        self.news_data = []
        self.news_headlines = []
        self.fear_greed = {"crypto": None, "market": None}
        self.last_scan = None
        self.last_news_scan = None
        self.scan_interval = 300

    # ── Reddit ──

    def _fetch_reddit_sub(self, sub):
        posts = []
        try:
            url = f"https://www.reddit.com/r/{sub}/hot/.rss?limit=25"
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=15) as resp:
                raw = resp.read().decode()
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
                    "sub": sub, "title": title, "text": content,
                    "score": 1, "comments": 0, "created": 0, "url": href,
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

    # ── News RSS ──

    def _fetch_rss_feed(self, feed):
        articles = []
        try:
            req = Request(feed["url"], headers=HEADERS)
            with urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            root = ET.fromstring(raw)
            items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
            for item in items[:20]:
                title = ""
                desc = ""
                for tag in ["title", "{http://www.w3.org/2005/Atom}title"]:
                    el = item.find(tag)
                    if el is not None and el.text:
                        title = el.text.strip()
                        break
                for tag in ["description", "summary", "{http://www.w3.org/2005/Atom}summary"]:
                    el = item.find(tag)
                    if el is not None and el.text:
                        desc = re.sub(r"<[^>]+>", " ", el.text)[:300].strip()
                        break
                if title:
                    articles.append({
                        "source": feed["name"],
                        "type": feed["type"],
                        "title": title,
                        "desc": desc,
                    })
        except Exception:
            pass
        return articles

    def scan_news(self):
        all_articles = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(self._fetch_rss_feed, f): f for f in NEWS_RSS_FEEDS}
            for f in as_completed(futures):
                articles = f.result()
                all_articles.extend(articles)

        news_mentions = {}
        for art in all_articles:
            full_text = art["title"] + " " + art["desc"]
            tickers = self._extract_tickers(full_text)
            sentiment, hits = self._calc_sentiment(full_text)
            for sym in tickers:
                if sym not in news_mentions:
                    news_mentions[sym] = {"count": 0, "sentiment": 0, "headlines": []}
                news_mentions[sym]["count"] += 1
                news_mentions[sym]["sentiment"] += sentiment
                if len(news_mentions[sym]["headlines"]) < 3:
                    news_mentions[sym]["headlines"].append({
                        "title": art["title"][:100],
                        "source": art["source"],
                        "sentiment": sentiment,
                    })

        for sym, data in news_mentions.items():
            if sym not in self.sentiment_scores:
                self.sentiment_scores[sym] = {
                    "mentions": 0, "sentiment": 0, "engagement": 0,
                    "label": "NEUTRAL", "top_posts": [],
                }
            sc = self.sentiment_scores[sym]
            sc["news_mentions"] = data["count"]
            sc["news_sentiment"] = round(data["sentiment"] / max(data["count"], 1), 2)
            sc["news_headlines"] = data["headlines"]
            old_s = sc["sentiment"]
            combined = (old_s * 0.6 + sc["news_sentiment"] * 0.4) if sc["mentions"] > 0 else sc["news_sentiment"]
            sc["sentiment"] = round(combined, 2)
            sc["label"] = "BULLISH" if combined > 1 else "BEARISH" if combined < -1 else "NEUTRAL"

        self.news_data = all_articles
        self.news_headlines = sorted(all_articles, key=lambda a: abs(self._calc_sentiment(a["title"])[0]), reverse=True)[:15]
        self.last_news_scan = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return news_mentions

    # ── Fear & Greed Index ──

    def scan_fear_greed(self):
        # Crypto Fear & Greed (alternative.me — free, no auth)
        try:
            req = Request("https://api.alternative.me/fng/?limit=1&format=json", headers=HEADERS)
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            item = data.get("data", [{}])[0]
            self.fear_greed["crypto"] = {
                "value": int(item.get("value", 50)),
                "label": item.get("value_classification", "Neutral"),
                "timestamp": item.get("timestamp", ""),
            }
        except Exception:
            pass

        # CNN Fear & Greed (market global)
        try:
            req = Request("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", headers={
                **HEADERS, "Referer": "https://edition.cnn.com/",
            })
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            score = data.get("fear_and_greed", {}).get("score", None)
            rating = data.get("fear_and_greed", {}).get("rating", "Neutral")
            if score is not None:
                self.fear_greed["market"] = {
                    "value": round(score),
                    "label": rating,
                }
        except Exception:
            pass

        return self.fear_greed

    # ── Shared utils ──

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

    # ── Main scan ──

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

    def scan_all(self):
        self.scan_reddit()
        self.scan_news()
        self.scan_fear_greed()
        self._recompute_trending()
        return self.sentiment_scores

    def _recompute_trending(self):
        self.trending = sorted(
            self.sentiment_scores.items(),
            key=lambda x: (x[1].get("mentions", 0) + x[1].get("news_mentions", 0)) * (1 + abs(x[1]["sentiment"])),
            reverse=True,
        )[:20]

    # ── Boost for trading ──

    def get_social_boost(self, sym):
        data = self.sentiment_scores.get(sym)
        fg = self.fear_greed
        boost = 1.0

        # Reddit + News sentiment
        if data:
            s = data["sentiment"]
            m = data.get("mentions", 0) + data.get("news_mentions", 0)
            if s > 2 and m >= 3:
                boost = 1.3
            elif s > 1 and m >= 2:
                boost = 1.15
            elif s < -2 and m >= 3:
                boost = 0.7
            elif s < -1 and m >= 2:
                boost = 0.85

        # Fear & Greed modulation
        crypto_fg = fg.get("crypto")
        market_fg = fg.get("market")
        fg_val = None
        if sym.endswith("-USD") and crypto_fg:
            fg_val = crypto_fg["value"]
        elif market_fg:
            fg_val = market_fg["value"]

        if fg_val is not None:
            if fg_val <= 20:
                boost *= 0.8
            elif fg_val <= 35:
                boost *= 0.9
            elif fg_val >= 80:
                boost *= 0.85
            elif fg_val >= 65:
                boost *= 1.1

        return round(max(0.5, min(1.5, boost)), 2)

    # ── API output ──

    def to_dict(self):
        trending_list = []
        for sym, data in self.trending:
            trending_list.append({
                "sym": sym,
                "mentions": data.get("mentions", 0),
                "news_mentions": data.get("news_mentions", 0),
                "sentiment": data["sentiment"],
                "label": data["label"],
                "engagement": data.get("engagement", 0),
                "top_posts": data.get("top_posts", []),
                "news_headlines": data.get("news_headlines", []),
            })
        return {
            "last_scan": self.last_scan,
            "last_news_scan": self.last_news_scan,
            "trending": trending_list,
            "total_posts": len(self.reddit_data),
            "total_news": len(self.news_data),
            "total_tickers": len(self.sentiment_scores),
            "subreddits": REDDIT_SUBS,
            "news_feeds": [f["name"] for f in NEWS_RSS_FEEDS],
            "fear_greed": self.fear_greed,
            "top_headlines": self.news_headlines[:10],
        }
