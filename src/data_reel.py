"""
Donnees reelles — Rex Clemens Capital
═════════════════════════════════════
Telechargement de vrais prix via yfinance.
Actions, crypto, devises — les 3 marches vises.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


CACHE_DIR = Path("data/cache")


ACTIFS = {
    "actions": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "AMZN": "Amazon",
        "TSLA": "Tesla",
    },
    "crypto": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "SOL-USD": "Solana",
    },
    "devises": {
        "EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD",
        "USDJPY=X": "USD/JPY",
    },
}


def telecharger(
    symbole: str,
    periode: str = "5y",
    intervalle: str = "1d",
    cache: bool = True,
) -> pd.Series:
    """
    Telecharge les prix de cloture d'un actif.
    Cache local pour eviter de retelecharger.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbole.replace('=', '_').replace('-', '_')}_{periode}.csv"

    if cache and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if len(df) > 100:
            return df["Close"].dropna()

    print(f"  Telechargement {symbole} ({periode})...")
    ticker = yf.Ticker(symbole)
    df = ticker.history(period=periode, interval=intervalle)

    if df.empty:
        raise ValueError(f"Aucune donnee pour {symbole}")

    df.to_csv(cache_path)
    return df["Close"].dropna()


def telecharger_tous(
    categorie: str = "all",
    periode: str = "5y",
) -> dict[str, pd.Series]:
    """Telecharge tous les actifs d'une categorie (ou toutes)."""
    resultats = {}

    categories = ACTIFS if categorie == "all" else {categorie: ACTIFS[categorie]}

    for cat, actifs in categories.items():
        for symbole, nom in actifs.items():
            try:
                prix = telecharger(symbole, periode)
                resultats[f"{nom} ({symbole})"] = prix
                print(f"    {nom}: {len(prix)} jours, "
                      f"{prix.index[0].strftime('%Y-%m-%d')} -> {prix.index[-1].strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"    {nom}: ERREUR - {e}")

    return resultats
