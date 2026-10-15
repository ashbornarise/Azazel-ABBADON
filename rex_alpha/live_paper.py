"""Live market monitor and paper portfolio for Rex Alpha.

Market data is read from Yahoo Finance through yfinance. Orders are routed only
to the local ExecutionSimulator; this module has no broker execution path.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime, time as day_time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from rex_alpha.alpha_engine import (
    AlphaScorer,
    FundamentalAnalyzer,
    ReversalDetector,
    StagnationDetector,
)
from rex_alpha.config import RexConfig
from rex_alpha.execution import ExecutionSimulator
from rex_alpha.risk_engine import RiskManager
from rex_alpha.storage import RexDB


NY = ZoneInfo("America/New_York")
DEFAULT_SYMBOLS = ("AAPL", "MSFT", "NVDA", "SPY")
STATE_PATH = Path("data/rex_alpha/live_paper_state.json")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RexAlphaLivePaper:
    """Fetch minute quotes, evaluate completed daily bars, and paper trade."""

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        poll_seconds: int | None = None,
        state_path: Path | str = STATE_PATH,
        database_path: Path | str | None = None,
    ) -> None:
        configured = os.getenv("REX_ALPHA_LIVE_SYMBOLS", "").strip()
        raw_symbols = symbols or tuple(
            part.strip().upper()
            for part in (configured.split(",") if configured else DEFAULT_SYMBOLS)
            if part.strip()
        )
        self.symbols = tuple(dict.fromkeys(raw_symbols))
        self.poll_seconds = max(
            60,
            poll_seconds or int(os.getenv("REX_ALPHA_LIVE_POLL_SECONDS", "60")),
        )
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = RexConfig()
        self.session_id = _utc_now().strftime("%Y%m%dT%H%M%SZ")
        session_database = database_path or Path(
            f"data/rex_alpha/live_paper_{self.session_id}.db"
        )
        self.db = RexDB(session_database)
        self.risk = RiskManager(self.config)
        self.simulator = ExecutionSimulator(self.db, self.risk, self.config)
        self.stagnation = StagnationDetector()
        self.reversal = ReversalDetector()
        self.fundamentals = FundamentalAnalyzer()
        self.scorer = AlphaScorer()
        self.lock = threading.RLock()
        self.daily: dict[str, pd.DataFrame] = {}
        self.last_daily_refresh = 0.0
        self.quality: dict[str, dict] = {}
        self.last_signal_day: dict[str, str] = {}
        self.paper_decision_day: dict[str, str] = {}
        self.fundamentals_cache: dict[str, dict] = {}
        self.last_fundamentals_at: dict[str, float] = {}
        self.price_history: dict[str, list[list]] = {s: [] for s in self.symbols}
        self.equity_history: list[list] = []
        self.signals: dict[str, dict] = {}
        self.trade_history: list[dict] = []
        self.last_prices: dict[str, float] = {}
        self.last_quote_times: dict[str, str] = {}
        self.errors: list[str] = []
        self.state: dict = {
            "mode": "PAPER",
            "session_id": self.session_id,
            "status": "initialisation",
            "source": "Yahoo Finance via yfinance · bougies 1 minute",
            "poll_seconds": self.poll_seconds,
            "capital_initial": self.config.capital_initial,
            "capital": self.config.capital_initial,
            "cash": self.config.capital_initial,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "drawdown": 0.0,
            "prices": {},
            "signals": {},
            "positions": [],
            "trades": [],
            "equity": [],
            "updated_at": None,
            "message": "Connexion à la source de cours en cours.",
        }
        self._save_state()

    def run_forever(self) -> None:
        while True:
            started = time.monotonic()
            try:
                self._update_once()
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
                self.errors = self.errors[-5:]
                with self.lock:
                    self.state["status"] = "erreur source"
                    self.state["message"] = self.errors[-1]
                    self.state["updated_at"] = _utc_now().isoformat()
                    self._save_state()
            time.sleep(max(1, self.poll_seconds - (time.monotonic() - started)))

    def _update_once(self) -> None:
        self._refresh_daily_history()
        quotes = self._fetch_intraday()
        now = _utc_now()

        with self.lock:
            for symbol, quote in quotes.items():
                price = float(quote["price"])
                stamp = quote["timestamp"]
                self.last_prices[symbol] = price
                self.last_quote_times[symbol] = stamp
                points = self.price_history.setdefault(symbol, [])
                for point in quote.get("history", []):
                    if points and points[-1][0] == point[0]:
                        points[-1] = point
                    elif not points or point[0] > points[-1][0]:
                        points.append(point)
                del points[:-500]

            self._evaluate_new_daily_bars(now)
            self._apply_daily_signals(now)
            self._manage_open_positions(now)

            fresh_prices = {
                symbol: price
                for symbol, price in self.last_prices.items()
                if self._quote_is_fresh(symbol, now)
            }
            if fresh_prices:
                self.simulator.update_prix(fresh_prices)
            portfolio = self.simulator.snapshot()
            equity_point = [now.isoformat(), portfolio["capital"]]
            self.equity_history.append(equity_point)
            del self.equity_history[:-1000]

            any_fresh = bool(fresh_prices)
            market_open = self._is_us_regular_session(now)
            if any_fresh and market_open:
                status = "cours actualisés · paper trading"
                message = "Cotations intraday reçues; ordres virtuels uniquement."
            elif any_fresh:
                status = "cours disponibles · marché US fermé"
                message = "Les cours sont visibles; aucune nouvelle position hors séance régulière."
            else:
                status = "attente de cotations fraîches"
                message = "La source n’a pas encore fourni de bougie intraday récente."

            positions = []
            for symbol, position in self.simulator.positions.items():
                if position.get("quantite", 0) <= 0:
                    continue
                price = self.last_prices.get(symbol, position["prix_moyen"])
                positions.append({
                    "symbol": symbol,
                    "quantity": round(position["quantite"], 6),
                    "entry_price": round(position["prix_moyen"], 4),
                    "price": round(price, 4),
                    "unrealized_pnl": round(
                        position["quantite"] * (price - position["prix_moyen"]), 2
                    ),
                })

            self.state.update({
                "status": status,
                "message": message,
                "capital_initial": self.config.capital_initial,
                "capital": portfolio["capital"],
                "cash": portfolio["cash"],
                "pnl": portfolio["pnl_total"],
                "pnl_pct": portfolio["pnl_pct"],
                "drawdown": portfolio["drawdown"],
                "prices": {
                    symbol: {
                        "price": round(self.last_prices[symbol], 5),
                        "timestamp": self.last_quote_times.get(symbol),
                        "history": self.price_history.get(symbol, []),
                        "signal": self.signals.get(symbol, {}),
                    }
                    for symbol in self.last_prices
                },
                "signals": self.signals,
                "positions": positions,
                "trades": self.trade_history[-100:],
                "equity": self.equity_history,
                "updated_at": now.isoformat(),
                "source_note": (
                    "Yahoo Finance/yfinance n’est pas un flux de ticks garanti; "
                    "les cotations peuvent être retardées ou indisponibles."
                ),
                "errors": self.errors,
            })
            self._save_state()

    def _refresh_daily_history(self) -> None:
        now = time.monotonic()
        if now - self.last_daily_refresh < 3600:
            return
        for symbol in self.symbols:
            try:
                history = yf.Ticker(symbol).history(
                    period="2y", interval="1d", auto_adjust=True, timeout=20
                )
                if history.empty or "Close" not in history:
                    self.quality[symbol] = {"ok": False, "message": "historique journalier absent"}
                    continue
                history = history.dropna(subset=["Close"])
                self.daily[symbol] = history
                self.quality[symbol] = {
                    "ok": len(history) >= 252,
                    "bars": len(history),
                    "message": "OK" if len(history) >= 252 else "historique inférieur à 252 séances",
                }
            except Exception as exc:
                self.quality[symbol] = {
                    "ok": False,
                    "message": f"historique indisponible ({type(exc).__name__})",
                }
        self.last_daily_refresh = now

    def _fetch_intraday(self) -> dict[str, dict]:
        data = yf.download(
            tickers=" ".join(self.symbols),
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=True,
            prepost=False,
            progress=False,
            threads=False,
            timeout=20,
        )
        result: dict[str, dict] = {}
        for symbol in self.symbols:
            frame = self._ticker_frame(data, symbol)
            if frame.empty or "Close" not in frame:
                continue
            closes = frame["Close"].dropna()
            if closes.empty:
                continue
            stamp = closes.index[-1]
            if getattr(stamp, "tzinfo", None) is None:
                stamp = stamp.tz_localize(NY)
            result[symbol] = {
                "price": float(closes.iloc[-1]),
                "timestamp": stamp.to_pydatetime().astimezone(timezone.utc).isoformat(),
                "history": [
                    [
                        index.to_pydatetime().astimezone(timezone.utc).isoformat()
                        if getattr(index, "tzinfo", None)
                        else index.tz_localize(NY).to_pydatetime().astimezone(timezone.utc).isoformat(),
                        round(float(value), 5),
                    ]
                    for index, value in closes.tail(120).items()
                ],
            }
        return result

    @staticmethod
    def _ticker_frame(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()
        if not isinstance(data.columns, pd.MultiIndex):
            return data if len(data.columns) else pd.DataFrame()
        for level in range(data.columns.nlevels):
            if symbol in data.columns.get_level_values(level):
                try:
                    frame = data.xs(symbol, axis=1, level=level)
                    if "Close" in frame.columns:
                        return frame
                except KeyError:
                    pass
        return pd.DataFrame()

    def _evaluate_new_daily_bars(self, now: datetime) -> None:
        now_ny = now.astimezone(NY)
        for symbol, history in self.daily.items():
            if symbol == self.config.benchmark:
                self.signals[symbol] = {
                    "decision": "BENCHMARK",
                    "message": "Référence de marché; aucun ordre simulé.",
                }
                continue
            if not self.quality.get(symbol, {}).get("ok"):
                self.signals[symbol] = {
                    "decision": "HISTORIQUE INSUFFISANT",
                    "message": self.quality[symbol]["message"],
                }
                continue

            completed = history
            last_day = completed.index[-1]
            last_date = last_day.date()
            if last_date >= now_ny.date() and now_ny.time() < day_time(16, 5):
                completed = completed.iloc[:-1]
            if len(completed) < 252:
                continue
            signal_day = str(completed.index[-1].date())
            if self.last_signal_day.get(symbol) == signal_day:
                continue

            prices = completed["Close"]
            volumes = completed["Volume"] if "Volume" in completed else None
            quality = self._get_fundamentals(symbol)
            if not quality.get("ok"):
                self.signals[symbol] = {
                    "decision": "DONNÉES FONDAMENTALES INDISPONIBLES",
                    "date_signal": signal_day,
                    "message": quality.get("raison", "source fondamentale indisponible"),
                }
                continue

            stagnation = self.stagnation.analyser(prices, volumes)
            reversal = self.reversal.analyser(prices, volumes)
            candidate = self.scorer.scorer(
                symbol,
                float(prices.iloc[-1]),
                stagnation,
                quality,
                reversal,
            )
            signal = asdict(candidate)
            signal["date_signal"] = signal_day
            signal["bar_status"] = "dernière séance clôturée"
            self.signals[symbol] = signal
            self.last_signal_day[symbol] = signal_day
            self.db.insert_signal(
                symbol,
                candidate.decision,
                candidate.score_alpha,
                candidate.details or {},
                "rex-alpha-live-0.1",
            )

    def _get_fundamentals(self, symbol: str) -> dict:
        now = time.monotonic()
        if now - self.last_fundamentals_at.get(symbol, 0) > 86_400:
            self.fundamentals_cache[symbol] = self.fundamentals.analyser(symbol)
            self.last_fundamentals_at[symbol] = now
        return self.fundamentals_cache.get(symbol, {"ok": False, "raison": "en attente"})

    def _apply_daily_signals(self, now: datetime) -> None:
        if not self._is_us_regular_session(now):
            return
        for symbol, signal in self.signals.items():
            signal_day = signal.get("date_signal")
            if not signal_day or self.paper_decision_day.get(symbol) == signal_day:
                continue
            if not self._quote_is_fresh(symbol, now):
                continue
            score = signal.get("score_alpha")
            if score is None:
                continue
            price = self.last_prices.get(symbol)
            if not price:
                continue

            position = self.simulator.positions.get(symbol, {})
            quantity = float(position.get("quantite", 0))
            if quantity > 0:
                if score < self.scorer.SEUIL_CANDIDAT:
                    result = self.simulator.vendre(
                        symbol, price, "score alpha sous le seuil candidat"
                    )
                    if result.get("ok"):
                        self._record_trade(symbol, "VENTE", price, result, signal_day)
                self.paper_decision_day[symbol] = signal_day
                continue

            if signal.get("decision") == "ACHAT_RECOMMANDE":
                state = self.simulator.get_state()
                max_risk_amount = state.capital * self.config.max_risque_par_trade_pct / 100
                amount = min(
                    max_risk_amount,
                    state.capital * self.config.max_poids_par_titre,
                    state.cash * 0.95,
                ) * min(float(score) / 100, 1.0)
                if amount > 0:
                    result = self.simulator.acheter(
                        symbol,
                        price,
                        round(amount, 2),
                        f"score Alpha {float(score):.1f} · signal sur clôture {signal_day}",
                    )
                    if result.get("ok"):
                        self._record_trade(symbol, "ACHAT", price, result, signal_day)
            self.paper_decision_day[symbol] = signal_day

    def _manage_open_positions(self, now: datetime) -> None:
        if not self._is_us_regular_session(now):
            return
        for symbol, position in list(self.simulator.positions.items()):
            quantity = float(position.get("quantite", 0))
            price = self.last_prices.get(symbol)
            if quantity <= 0 or price is None or not self._quote_is_fresh(symbol, now):
                continue
            stop_price = float(position["prix_moyen"]) * (
                1 - self.config.stop_loss_pct / 100
            )
            if price <= stop_price:
                result = self.simulator.vendre(symbol, price, "stop virtuel")
                if result.get("ok"):
                    self._record_trade(symbol, "VENTE", price, result, "stop intraday")

    def _record_trade(self, symbol: str, action: str, price: float, result: dict, signal_day: str) -> None:
        self.trade_history.append({
            "timestamp": _utc_now().isoformat(),
            "signal_day": signal_day,
            "symbol": symbol,
            "action": action,
            "price": round(price, 4),
            "quantity": result.get("quantite", 0),
            "pnl": result.get("pnl"),
            "reason": result.get("raison", ""),
        })
        self.trade_history = self.trade_history[-100:]

    def _quote_is_fresh(self, symbol: str, now: datetime) -> bool:
        stamp = self.last_quote_times.get(symbol)
        if not stamp:
            return False
        age = (now - datetime.fromisoformat(stamp)).total_seconds()
        return -5 <= age <= max(180, self.poll_seconds * 3)

    @staticmethod
    def _is_us_regular_session(now: datetime) -> bool:
        local = now.astimezone(NY)
        return (
            local.weekday() < 5
            and day_time(9, 30) <= local.time() < day_time(16, 0)
        )

    def _save_state(self) -> None:
        temp_path = self.state_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(self.state_path)
