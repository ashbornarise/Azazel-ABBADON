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
INTRADAY_TIMEFRAMES = {"1m": "1min", "5m": "5min", "1h": "1h"}
PAPER_SLEEVES = ("1m", "5m", "1h", "1d")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RexAlphaLivePaper:
    """Paper trade Rex Alpha independently on minute, 5-minute, hourly and daily bars."""

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
        self.sleeve_capital = self.config.capital_initial / len(PAPER_SLEEVES)
        self.combined_peak = self.config.capital_initial
        self.session_id = _utc_now().strftime("%Y%m%dT%H%M%SZ")
        session_database = database_path or Path(
            f"data/rex_alpha/live_paper_{self.session_id}_1d.db"
        )
        self.db = RexDB(session_database)
        daily_config = RexConfig(capital_initial=self.sleeve_capital)
        self.risk = RiskManager(daily_config)
        self.simulator = ExecutionSimulator(self.db, self.risk, daily_config)
        self.simulators = {"1d": self.simulator}
        self.databases = {"1d": self.db}
        for timeframe in PAPER_SLEEVES:
            if timeframe == "1d":
                continue
            frame_db = RexDB(
                f"data/rex_alpha/live_paper_{self.session_id}_{timeframe}.db"
            )
            frame_config = RexConfig(capital_initial=self.sleeve_capital)
            frame_risk = RiskManager(frame_config)
            self.databases[timeframe] = frame_db
            self.simulators[timeframe] = ExecutionSimulator(
                frame_db, frame_risk, frame_config
            )
        self.stagnation = StagnationDetector()
        self.reversal = ReversalDetector()
        self.fundamentals = FundamentalAnalyzer()
        self.scorer = AlphaScorer()
        self.lock = threading.RLock()
        self.daily: dict[str, pd.DataFrame] = {}
        self.last_daily_refresh = 0.0
        self.hourly: dict[str, pd.DataFrame] = {}
        self.last_hourly_refresh = 0.0
        self.intraday: dict[str, pd.DataFrame] = {}
        self.quality: dict[str, dict] = {}
        self.last_signal_day: dict[str, str] = {}
        self.paper_decision_day: dict[str, str] = {}
        self.last_timeframe_bar: dict[str, dict[str, str]] = {
            tf: {} for tf in PAPER_SLEEVES
        }
        self.paper_timeframe_decision: dict[str, dict[str, str]] = {
            tf: {} for tf in PAPER_SLEEVES
        }
        self.fundamentals_cache: dict[str, dict] = {}
        self.last_fundamentals_at: dict[str, float] = {}
        self.price_history: dict[str, list[list]] = {s: [] for s in self.symbols}
        self.equity_history: list[list] = []
        self.signals: dict[str, dict] = {}
        self.timeframe_signals: dict[str, dict[str, dict]] = {
            tf: {} for tf in PAPER_SLEEVES
        }
        self.trade_history: list[dict] = []
        self.last_prices: dict[str, float] = {}
        self.last_quote_times: dict[str, str] = {}
        self.errors: list[str] = []
        self.state: dict = {
            "mode": "PAPER",
            "session_id": self.session_id,
            "timeframes": list(PAPER_SLEEVES),
            "sleeve_capital": self.sleeve_capital,
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
        self._refresh_hourly_history()
        quotes = self._fetch_intraday()
        now = _utc_now()

        with self.lock:
            for symbol, quote in quotes.items():
                price = float(quote["price"])
                stamp = quote["timestamp"]
                self.last_prices[symbol] = price
                self.last_quote_times[symbol] = stamp
                if "bars" in quote:
                    self.intraday[symbol] = quote["bars"]
                    self.hourly[symbol] = self._merge_intraday_hourly(
                        self.hourly.get(symbol, pd.DataFrame()), quote["bars"]
                    )
                points = self.price_history.setdefault(symbol, [])
                for point in quote.get("history", []):
                    if points and points[-1][0] == point[0]:
                        points[-1] = point
                    elif not points or point[0] > points[-1][0]:
                        points.append(point)
                del points[:-500]

            self._evaluate_new_daily_bars(now)
            self.timeframe_signals["1d"] = dict(self.signals)
            self._evaluate_intraday_bars(now)
            self._apply_daily_signals(now)
            self._apply_intraday_signals(now)
            self._manage_open_positions(now)

            fresh_prices = {
                symbol: price
                for symbol, price in self.last_prices.items()
                if self._quote_is_fresh(symbol, now)
            }
            if fresh_prices:
                for simulator in self.simulators.values():
                    simulator.update_prix(fresh_prices)
            sleeve_snapshots = {
                timeframe: simulator.snapshot()
                for timeframe, simulator in self.simulators.items()
            }
            portfolio = self._combined_snapshot(sleeve_snapshots)
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
            for timeframe, simulator in self.simulators.items():
                for symbol, position in simulator.positions.items():
                    if position.get("quantite", 0) <= 0:
                        continue
                    price = self.last_prices.get(symbol, position["prix_moyen"])
                    positions.append({
                        "timeframe": timeframe,
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
                "sleeve_capital": self.sleeve_capital,
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
                "signals_by_timeframe": self.timeframe_signals,
                "timeframe_portfolios": sleeve_snapshots,
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

    def _combined_snapshot(self, sleeves: dict[str, dict]) -> dict:
        capital = round(sum(s["capital"] for s in sleeves.values()), 2)
        cash = round(sum(s["cash"] for s in sleeves.values()), 2)
        pnl = round(capital - self.config.capital_initial, 2)
        self.combined_peak = max(self.combined_peak, capital)
        return {
            "capital": capital,
            "cash": cash,
            "pnl_total": pnl,
            "pnl_pct": round(pnl / self.config.capital_initial * 100, 2),
            "drawdown": round(
                (capital - self.combined_peak) / self.combined_peak * 100,
                2,
            ) if self.combined_peak else 0,
            "nb_positions": sum(s["nb_positions"] for s in sleeves.values()),
        }

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

    def _refresh_hourly_history(self) -> None:
        now = time.monotonic()
        if now - self.last_hourly_refresh < 3600:
            return
        for symbol in self.symbols:
            if symbol == self.config.benchmark:
                continue
            try:
                history = yf.Ticker(symbol).history(
                    period="2y", interval="1h", auto_adjust=True, timeout=20
                )
                if not history.empty and "Close" in history:
                    self.hourly[symbol] = history.dropna(subset=["Close"])
            except Exception as exc:
                self.errors.append(f"{symbol} 1h: {type(exc).__name__}: {exc}")
                self.errors = self.errors[-5:]
        self.last_hourly_refresh = now

    def _fetch_intraday(self) -> dict[str, dict]:
        data = yf.download(
            tickers=" ".join(self.symbols),
            period="5d",
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
                "bars": frame,
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
    def _closed_bars(frame: pd.DataFrame, interval: str, now: datetime) -> pd.DataFrame:
        if frame.empty or "Close" not in frame:
            return frame.iloc[0:0]
        closes_at = {
            "1min": pd.Timedelta(minutes=1),
            "5min": pd.Timedelta(minutes=5),
            "1h": pd.Timedelta(hours=1),
        }[interval]
        end = now.astimezone(NY)
        if frame.index.tz is None:
            frame = frame.copy()
            frame.index = frame.index.tz_localize(NY)
        return frame.loc[frame.index + closes_at <= end]

    @staticmethod
    def _merge_intraday_hourly(
        history: pd.DataFrame, minute_bars: pd.DataFrame
    ) -> pd.DataFrame:
        recent = minute_bars.resample(
            "1h", origin="start_day", offset="30min", label="left", closed="left"
        ).agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna(subset=["Close"])
        combined = pd.concat([history, recent])
        return combined.loc[~combined.index.duplicated(keep="last")].sort_index()

    def _evaluate_intraday_bars(self, now: datetime) -> None:
        for timeframe, interval in INTRADAY_TIMEFRAMES.items():
            for symbol in self.symbols:
                if symbol == self.config.benchmark:
                    self.timeframe_signals[timeframe][symbol] = {
                        "decision": "BENCHMARK",
                        "message": "Référence de marché; aucun ordre simulé.",
                    }
                    continue
                if timeframe == "1h":
                    frame = self.hourly.get(symbol, pd.DataFrame())
                else:
                    source = self.intraday.get(symbol, pd.DataFrame())
                    if source.empty:
                        continue
                    if timeframe == "1m":
                        frame = source
                    else:
                        frame = source.resample(
                            interval, label="left", closed="left"
                        ).agg({
                            "Open": "first", "High": "max", "Low": "min",
                            "Close": "last", "Volume": "sum",
                        }).dropna(subset=["Close"])

                completed = self._closed_bars(frame, interval, now)
                if len(completed) < 252:
                    self.timeframe_signals[timeframe][symbol] = {
                        "decision": "HISTORIQUE INSUFFISANT",
                        "message": f"{len(completed)}/252 bougies clôturées pour {timeframe}.",
                        "date_signal": str(completed.index[-1]) if not completed.empty else None,
                    }
                    continue
                bar_id = completed.index[-1].isoformat()
                if self.last_timeframe_bar[timeframe].get(symbol) == bar_id:
                    continue

                prices = completed["Close"]
                volumes = completed["Volume"] if "Volume" in completed else None
                quality = self._get_fundamentals(symbol)
                if not quality.get("ok"):
                    self.timeframe_signals[timeframe][symbol] = {
                        "decision": "DONNÉES FONDAMENTALES INDISPONIBLES",
                        "date_signal": bar_id,
                        "message": quality.get("raison", "source fondamentale indisponible"),
                    }
                    continue

                stagnation = self.stagnation.analyser(prices, volumes)
                reversal = self.reversal.analyser(prices, volumes)
                candidate = self.scorer.scorer(
                    symbol, float(prices.iloc[-1]), stagnation, quality, reversal
                )
                signal = asdict(candidate)
                signal.update({
                    "timeframe": timeframe,
                    "date_signal": bar_id,
                    "bar_status": "dernière bougie clôturée",
                    "message": candidate.raison,
                })
                self.timeframe_signals[timeframe][symbol] = signal
                self.last_timeframe_bar[timeframe][symbol] = bar_id
                self.databases[timeframe].insert_signal(
                    symbol, candidate.decision, candidate.score_alpha,
                    candidate.details or {}, f"rex-alpha-live-{timeframe}-0.1",
                )

    def _apply_intraday_signals(self, now: datetime) -> None:
        if not self._is_us_regular_session(now):
            return
        for timeframe in INTRADAY_TIMEFRAMES:
            simulator = self.simulators[timeframe]
            for symbol, signal in self.timeframe_signals[timeframe].items():
                signal_bar = signal.get("date_signal")
                if not signal_bar or self.paper_timeframe_decision[timeframe].get(symbol) == signal_bar:
                    continue
                if not self._quote_is_fresh(symbol, now):
                    continue
                score = signal.get("score_alpha")
                price = self.last_prices.get(symbol)
                if score is None or not price:
                    continue

                position = simulator.positions.get(symbol, {})
                if float(position.get("quantite", 0)) > 0:
                    if score < self.scorer.SEUIL_CANDIDAT:
                        result = simulator.vendre(
                            symbol, price, f"{timeframe}: score Alpha sous le seuil candidat"
                        )
                        if result.get("ok"):
                            self._record_trade(
                                symbol, "VENTE", price, result, signal_bar, timeframe
                            )
                elif signal.get("decision") == "ACHAT_RECOMMANDE":
                    account = simulator.get_state()
                    config = simulator.config
                    amount = min(
                        account.capital * config.max_risque_par_trade_pct / 100,
                        account.capital * config.max_poids_par_titre,
                        account.cash * 0.95,
                    ) * min(float(score) / 100, 1.0)
                    if amount > 0:
                        result = simulator.acheter(
                            symbol, price, round(amount, 2),
                            f"{timeframe}: score Alpha {float(score):.1f} sur bougie clôturée",
                        )
                        if result.get("ok"):
                            self._record_trade(
                                symbol, "ACHAT", price, result, signal_bar, timeframe
                            )
                self.paper_timeframe_decision[timeframe][symbol] = signal_bar

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
        for timeframe, simulator in self.simulators.items():
            for symbol, position in list(simulator.positions.items()):
                quantity = float(position.get("quantite", 0))
                price = self.last_prices.get(symbol)
                if quantity <= 0 or price is None or not self._quote_is_fresh(symbol, now):
                    continue
                stop_price = float(position["prix_moyen"]) * (
                    1 - simulator.config.stop_loss_pct / 100
                )
                if price <= stop_price:
                    result = simulator.vendre(
                        symbol, price, f"{timeframe}: stop virtuel"
                    )
                    if result.get("ok"):
                        self._record_trade(
                            symbol, "VENTE", price, result, "stop intraday", timeframe
                        )

    def _record_trade(
        self, symbol: str, action: str, price: float, result: dict,
        signal_day: str, timeframe: str = "1d",
    ) -> None:
        self.trade_history.append({
            "timestamp": _utc_now().isoformat(),
            "timeframe": timeframe,
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
