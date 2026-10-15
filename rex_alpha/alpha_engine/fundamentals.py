"""
Analyse fondamentale — Rex Alpha Engine
Utilise les données yfinance pour le prototype.
Source production : SEC EDGAR.
"""
from __future__ import annotations

import yfinance as yf


class FundamentalAnalyzer:
    """Analyse la qualité fondamentale d'une action."""

    def analyser(self, symbole: str) -> dict:
        try:
            t = yf.Ticker(symbole)
            info = t.info or {}
        except Exception:
            return {"symbole": symbole, "ok": False, "raison": "donnees indisponibles"}

        # Rentabilité
        roe = info.get("returnOnEquity")
        roa = info.get("returnOnAssets")
        marge_nette = info.get("profitMargins")
        marge_op = info.get("operatingMargins")

        # Cash flow
        fcf = info.get("freeCashflow")
        op_cf = info.get("operatingCashflow")

        # Dette
        dette_equity = info.get("debtToEquity")
        current_ratio = info.get("currentRatio")

        # Valorisation
        pe = info.get("trailingPE")
        pe_forward = info.get("forwardPE")
        pb = info.get("priceToBook")
        ps = info.get("priceToSalesTrailing12Months")
        ev_ebitda = info.get("enterpriseToEbitda")

        # Croissance
        rev_growth = info.get("revenueGrowth")
        earn_growth = info.get("earningsGrowth")

        # Market cap
        market_cap = info.get("marketCap", 0)

        # Score qualité (0-100)
        score = 0
        checks = []

        if roe and roe > 0.10:
            score += 15
            checks.append("ROE > 10%")
        elif roe and roe > 0:
            score += 8

        if marge_nette and marge_nette > 0.08:
            score += 12
            checks.append("Marge nette > 8%")
        elif marge_nette and marge_nette > 0:
            score += 5

        if fcf and fcf > 0:
            score += 12
            checks.append("FCF positif")

        if dette_equity is not None and dette_equity < 100:
            score += 10
            checks.append("Dette/Equity < 100%")
        elif dette_equity is not None and dette_equity < 200:
            score += 5

        if current_ratio and current_ratio > 1.5:
            score += 8
            checks.append("Current ratio > 1.5")

        if pe and 0 < pe < 25:
            score += 10
            checks.append(f"PE = {pe:.1f}")
        elif pe and 0 < pe < 40:
            score += 5

        if rev_growth and rev_growth > 0.05:
            score += 8
            checks.append("Croissance CA > 5%")

        if marge_op and marge_op > 0.15:
            score += 10
            checks.append("Marge op > 15%")

        if ev_ebitda and 0 < ev_ebitda < 15:
            score += 8
            checks.append(f"EV/EBITDA = {ev_ebitda:.1f}")

        score = min(score, 100)

        return {
            "symbole": symbole,
            "ok": True,
            "score_qualite": round(score, 1),
            "checks": checks,
            "roe": round(roe * 100, 1) if roe else None,
            "marge_nette": round(marge_nette * 100, 1) if marge_nette else None,
            "marge_op": round(marge_op * 100, 1) if marge_op else None,
            "dette_equity": round(dette_equity, 1) if dette_equity else None,
            "current_ratio": round(current_ratio, 2) if current_ratio else None,
            "pe": round(pe, 1) if pe else None,
            "pe_forward": round(pe_forward, 1) if pe_forward else None,
            "pb": round(pb, 2) if pb else None,
            "ps": round(ps, 2) if ps else None,
            "ev_ebitda": round(ev_ebitda, 1) if ev_ebitda else None,
            "fcf": fcf,
            "rev_growth": round(rev_growth * 100, 1) if rev_growth else None,
            "earn_growth": round(earn_growth * 100, 1) if earn_growth else None,
            "market_cap": market_cap,
        }
