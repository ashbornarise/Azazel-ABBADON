"""Run comparable, walk-forward research on Rex Alpha's readable strategies.

Opaque MT4/MT5 executables are catalogued but are never executed by this module.
They require their matching MetaTrader Strategy Tester and instrument data.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
MATRIX = ROOT / "rex_alpha" / "robot_test_matrix.csv"
DEFAULT_OUTPUT = ROOT / "data" / "rex_alpha" / "robot_lab_dashboard.json"
ROUND_TRIP_COST = 0.001  # 0.1% per unit change in exposure, as in project backtests
OOS_FRACTION = 0.30

ASSET_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "TSLA": "Tesla",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
    "EURUSD=X": "EUR/USD",
}
CACHE_SYMBOLS = {
    "AAPL": "AAPL", "MSFT": "MSFT", "TSLA": "TSLA",
    "BTC_USD": "BTC-USD", "ETH_USD": "ETH-USD", "SOL_USD": "SOL-USD",
    "EURUSD_X": "EURUSD=X",
}

NATIVE_STRATEGIES = [
    {
        "id": "buy_hold", "name": "Buy & Hold", "family": "Référence passive",
        "why": "Reste investi; mesure la prime de marché sans essayer de timer les entrées.",
        "works_when": "Marché haussier de long terme.",
        "fails_when": "Baisse prolongée; drawdown proche de celui du marché.",
    },
    {
        "id": "ma_20_100", "name": "MA 20/100", "family": "Suivi de tendance",
        "why": "Achète lorsque la moyenne courte dépasse la moyenne longue et sort au croisement inverse.",
        "works_when": "Tendances longues et persistantes.",
        "fails_when": "Marché latéral: croisements répétés et frais.",
    },
    {
        "id": "ma_5_20", "name": "MA 5/20", "family": "Suivi de tendance rapide",
        "why": "Même règle que MA 20/100, avec des fenêtres plus courtes et davantage de rotations.",
        "works_when": "Mouvements directionnels rapides.",
        "fails_when": "Bruit, spread et retournements fréquents.",
    },
    {
        "id": "ml_random_forest", "name": "Random Forest walk-forward", "family": "Classification ML",
        "why": "Estime la probabilité d'une hausse suivante avec momentum, volatilité, RSI et distances aux moyennes.",
        "works_when": "Combinaisons de signaux stables hors échantillon.",
        "fails_when": "Changement de régime, surapprentissage ou probabilité mal calibrée.",
    },
    {
        "id": "random_control", "name": "Témoin aléatoire (10 graines)", "family": "Contrôle nul",
        "why": "Expositions aléatoires reproductibles; détecte un résultat chanceux ou un biais de test.",
        "works_when": "Ne doit pas produire un avantage stable après coûts.",
        "fails_when": "Un bon résultat isolé peut arriver par hasard; comparer la distribution des graines.",
    },
]

RESOURCE_PACKS = [
    {"name": "Freqtrade", "type": "Framework de bots crypto", "role": "Permet d'écrire, backtester et paper-trader des stratégies Python; l'archive seule n'est pas une stratégie configurée.", "status": "Ressource; aucune stratégie personnalisée Rex importée."},
    {"name": "NautilusTrader", "type": "Moteur événementiel", "role": "Outils de backtest et d'exécution événementielle multi-actifs; nécessite une stratégie et des données adaptées.", "status": "Ressource; non connecté au dashboard live."},
    {"name": "TradingAgents", "type": "Recherche multi-agents", "role": "Produit une analyse argumentée par rôles d'agents; ce n'est pas un EA MT4/MT5 autonome.", "status": "Ressource; clés et configuration absentes, aucun résultat présenté comme testé."},
    {"name": "OpenBB", "type": "Plateforme de données", "role": "Aide à rassembler des données et indicateurs; ne définit pas à elle seule une règle d'entrée/sortie.", "status": "Ressource data; pas une stratégie de trading."},
    {"name": "Robot Trading Brief", "type": "Spécification du projet", "role": "Décrit le croisement de moyennes mobiles, les coûts, la séparation temporelle et le témoin aléatoire.", "status": "Principes réutilisés par le banc de recherche."},
]


def classify_expert(name: str) -> dict[str, Any]:
    """Infer a likely family from the filename only; never treat it as verified."""
    low = name.casefold()
    traits: list[str] = []
    if re.search(r"grid|marti|recovery|hedge|dual balance", low):
        family = "Grille / récupération"
        traits = ["peut empiler des positions", "risque de perte extrême si le marché tend contre la grille"]
    elif re.search(r"scalp|scalper|m1|m5|stepbyscalp", low):
        family = "Scalping probable"
        traits = ["horizon court probable", "spread, commission et qualité des ticks déterminants"]
    elif re.search(r"breakout|break out|orb", low):
        family = "Cassure probable"
        traits = ["cherche une sortie de range", "faux breakouts et gaps à mesurer"]
    elif re.search(r"trend|harvester", low):
        family = "Suivi de tendance probable"
        traits = ["cherche à garder les mouvements directionnels", "peut souffrir en marché latéral"]
    elif re.search(r"reversal|mean.?reversion|colorreversal", low):
        family = "Retour à la moyenne probable"
        traits = ["cherche un rebond vers une valeur moyenne", "risque de renforcer trop tôt une tendance forte"]
    elif re.search(r"smc|supply.?demand|order|breaker|price.?action|candle|pattern", low):
        family = "Price action / zones probable"
        traits = ["signaux basés sur structures ou motifs de prix", "règles exactes non visibles dans le binaire"]
    elif re.search(r"copy.?trade|multimaster", low):
        family = "Copie / orchestration"
        traits = ["peut répliquer des comptes ou des signaux", "ne constitue pas forcément un signal autonome"]
    elif re.search(r"protect|tp.?sl|manager|panel|account", low):
        family = "Outil de gestion probable"
        traits = ["peut gérer le risque ou les ordres", "pas nécessairement une stratégie d'entrée"]
    elif re.search(r"ai|robot|ea", low):
        family = "EA opaque — rôle inconnu"
        traits = ["le nom ne révèle pas la règle", "documentation ou rapport MT requis"]
    else:
        family = "Rôle inconnu"
        traits = ["aucune logique déductible du nom"]
    platform = "MT4" if name.casefold().endswith(".ex4") else "MT5"
    return {
        "family_guess": family,
        "traits_guess": traits,
        "confidence": "faible — inféré du nom uniquement",
        "platform": platform,
        "test_status": "À tester dans MetaTrader Strategy Tester",
        "test_note": "Binaire compilé absent du dépôt; aucune performance inférée.",
        "needs": "Binaire disponible sur le poste MT; historique et rapport MetaTrader requis.",
    }


def _read_prices(refresh: bool = False) -> dict[str, pd.Series]:
    if refresh:
        try:
            import yfinance as yf
        except ImportError:
            yf = None
    else:
        yf = None

    result: dict[str, pd.Series] = {}
    for stem, symbol in CACHE_SYMBOLS.items():
        path = CACHE / f"{stem}_5y.csv"
        frame: pd.DataFrame | None = None
        if yf is not None:
            try:
                downloaded = yf.Ticker(symbol).history(
                    period="5y", interval="1d", auto_adjust=True, timeout=20
                )
                if not downloaded.empty:
                    frame = downloaded
            except Exception:
                pass
        if frame is None and path.exists():
            try:
                frame = pd.read_csv(path, index_col=0, parse_dates=True)
            except Exception:
                continue
        if frame is None or "Close" not in frame:
            continue
        prices = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        prices = prices.loc[np.isfinite(prices) & (prices > 0)]
        if not prices.empty:
            zone = "UTC" if symbol.endswith("-USD") else "America/New_York"
            latest = pd.Timestamp(prices.index[-1])
            latest = latest.tz_localize(zone) if latest.tzinfo is None else latest.tz_convert(zone)
            now_local = pd.Timestamp.now(tz=zone)
            close_time = "23:59" if symbol.endswith("-USD") else "17:05" if symbol.endswith("=X") else "16:05"
            if latest.date() == now_local.date() and now_local.strftime("%H:%M") < close_time:
                prices = prices.iloc[:-1]
        if len(prices) >= 300:
            result[symbol] = prices.sort_index()
    return result


def _strategies() -> dict[str, Any]:
    from src.strategies.buy_hold import BuyHold
    from src.strategies.ma_cross import MACross
    from src.strategies.ml_strategy import MLStrategy
    from src.strategies.random_control import RandomControl

    return {
        "buy_hold": BuyHold(),
        "ma_20_100": MACross(20, 100),
        "ma_5_20": MACross(5, 20),
        "ml_random_forest": MLStrategy(
            taille_min_train=252, pas_reentrain=21, n_arbres=50, seed=42
        ),
    }


def _evaluate(
    prices: pd.Series,
    strategy: Any,
    cost: float = ROUND_TRIP_COST,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    signal = strategy.calculer_exposition(prices).reindex(prices.index).fillna(0).clip(0, 1)
    position = signal.shift(1).fillna(0)
    market_returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).clip(-0.99, None)
    turnover = position.diff().abs().fillna(position.abs())
    returns = (position * market_returns - turnover * cost).replace([np.inf, -np.inf], 0).fillna(0)
    split = int(len(prices) * (1 - OOS_FRACTION))
    oos_returns = returns.iloc[split:]
    oos_position = position.iloc[split:]
    if oos_returns.empty:
        return {"error": "échantillon hors test insuffisant"}
    equity = (1 + oos_returns).cumprod()
    years = len(oos_returns) / periods_per_year
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else -1.0
    std = float(oos_returns.std())
    sharpe = float(oos_returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    drawdown = float((equity / equity.cummax() - 1).min())
    wins = float((oos_returns > 0).mean())
    pos_sum = float(oos_returns.clip(lower=0).sum())
    neg_sum = float(-oos_returns.clip(upper=0).sum())
    profit_factor = pos_sum / neg_sum if neg_sum > 0 else (None if pos_sum == 0 else 99.0)
    # The signal observed at t is aligned with the realized return at t+1.
    # In the OOS block this includes the final pre-cutoff signal for its first bar.
    forecast_probability = signal.shift(1).iloc[split:].clip(0, 1)
    forecast_actual = (market_returns.iloc[split:] > 0).astype(float)
    prediction_hit_rate = float(
        ((forecast_probability >= 0.5) == (forecast_actual >= 0.5)).mean()
    )
    prediction_brier_score = float(
        ((forecast_probability - forecast_actual) ** 2).mean()
    )
    return {
        "oos_start": str(oos_returns.index[0])[:10],
        "oos_end": str(oos_returns.index[-1])[:10],
        "oos_bars": int(len(oos_returns)),
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "profit_factor_daily": profit_factor,
        "winning_days": wins,
        "turnover": float(turnover.iloc[split:].sum()),
        "average_exposure": float(oos_position.mean()),
        "last_signal": float(signal.iloc[-1]),
        "prediction_hit_rate": prediction_hit_rate,
        "prediction_brier_score": prediction_brier_score,
        "equity": [[str(i)[:10], round(float(v), 6)] for i, v in equity.iloc[::max(1, len(equity)//160)].items()],
    }


def _load_catalog() -> list[dict[str, Any]]:
    import csv

    robots: list[dict[str, Any]] = []
    if not MATRIX.exists():
        return robots
    with MATRIX.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            robots.append({
                "id": row.get("ID", ""),
                "name": row.get("Nom", ""),
                "market_hint": row.get("Marche_suppose", "À déterminer"),
                "timeframe_hint": row.get("Unite_temps", "À définir"),
                "verified": row.get("Role_verifie", "Non"),
                "documentation": row.get("Documentation", "À collecter"),
                "risk_name_hint": row.get("Indice_risque_nom", ""),
                **classify_expert(row.get("Nom", "")),
            })
    return robots


def _summary(results: list[dict[str, Any]], strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    buy_hold = {
        r["symbol"]: r["metrics"]["total_return"]
        for r in results
        if r["strategy_id"] == "buy_hold" and "error" not in r["metrics"]
    }
    for spec in strategies:
        samples = [r for r in results if r["strategy_id"] == spec["id"] and "error" not in r["metrics"]]
        if not samples:
            rows.append({**spec, "assets_tested": 0, "median_cagr": None, "median_sharpe": None, "median_drawdown": None, "positive_assets": 0, "beats_buy_hold": 0})
            continue
        returns = [r["metrics"]["total_return"] for r in samples]
        rows.append({
            **spec,
            "assets_tested": len(samples),
            "median_cagr": median(r["metrics"]["cagr"] for r in samples),
            "median_sharpe": median(r["metrics"]["sharpe"] for r in samples),
            "median_drawdown": median(r["metrics"]["max_drawdown"] for r in samples),
            "median_profit_factor_daily": median(
                r["metrics"]["profit_factor_daily"] or 0 for r in samples
            ),
            "positive_assets": sum(v > 0 for v in returns),
            "beats_buy_hold": sum(
                r["metrics"]["total_return"] > buy_hold.get(r["symbol"], float("inf"))
                for r in samples
            ),
            "median_oos_return": median(returns),
        })
    return rows


def generate_dashboard_data(refresh: bool = False, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    readable_results: list[dict[str, Any]] = []
    forecasts: list[dict[str, Any]] = []
    specs = list(NATIVE_STRATEGIES)
    series_by_asset = _read_prices(refresh=refresh)
    strategy_cache: dict[str, Any] = {}

    for symbol, prices in series_by_asset.items():
        cut = int(len(prices) * (1 - OOS_FRACTION))
        if cut < 300 or len(prices) - cut < 50:
            continue
        periods_per_year = 365 if symbol.endswith("-USD") else 252
        for spec in specs:
            if spec["id"] == "random_control":
                seed_results = []
                for seed in range(10):
                    from src.strategies.random_control import RandomControl
                    metrics = _evaluate(
                        prices, RandomControl(seed=seed), periods_per_year=periods_per_year
                    )
                    seed_results.append(metrics)
                metric_rows = [m for m in seed_results if "error" not in m]
                if not metric_rows:
                    continue
                combined = {
                    key: median(m[key] for m in metric_rows)
                    for key in ("total_return", "cagr", "sharpe", "max_drawdown", "winning_days", "average_exposure")
                }
                combined.update({
                    "oos_start": metric_rows[0]["oos_start"],
                    "oos_end": metric_rows[0]["oos_end"],
                    "oos_bars": metric_rows[0]["oos_bars"],
                    "profit_factor_daily": median(m["profit_factor_daily"] or 0 for m in metric_rows),
                    "turnover": median(m["turnover"] for m in metric_rows),
                    "last_signal": median(m["last_signal"] for m in metric_rows),
                    "equity": metric_rows[0]["equity"],
                    "random_seed_runs": len(metric_rows),
                })
                metrics = combined
            else:
                strategy = strategy_cache.get(spec["id"])
                if strategy is None:
                    strategy = _strategies()[spec["id"]]
                    strategy_cache[spec["id"]] = strategy
                metrics = _evaluate(
                    prices, strategy, periods_per_year=periods_per_year
                )

            readable_results.append({
                "strategy_id": spec["id"],
                "strategy_name": spec["name"],
                "symbol": symbol,
                "asset_name": ASSET_NAMES.get(symbol, symbol),
                "data_start": str(prices.index[0])[:10],
                "data_end": str(prices.index[-1])[:10],
                "bars": int(len(prices)),
                "metrics": metrics,
            })

            if spec["id"] == "ml_random_forest" and "error" not in metrics:
                probability = float(metrics["last_signal"])
                direction = "HAUSSE probable" if probability >= 0.55 else "BAISSE probable" if probability <= 0.45 else "INCERTAIN"
                short_ma = float(prices.rolling(20).mean().iloc[-1])
                long_ma = float(prices.rolling(100).mean().iloc[-1])
                forecasts.append({
                    "symbol": symbol,
                    "asset_name": ASSET_NAMES.get(symbol, symbol),
                    "as_of": str(prices.index[-1])[:10],
                    "price": float(prices.iloc[-1]),
                    "probability_up": probability,
                    "oos_hit_rate": metrics["prediction_hit_rate"],
                    "oos_brier_score": metrics["prediction_brier_score"],
                    "direction": direction,
                    "ma_20_100": "HAUSSIER" if short_ma > long_ma else "BAISSIER",
                    "model_note": "Random Forest walk-forward; proba calculée avec le passé disponible uniquement.",
                })

    robots = _load_catalog()
    data_end = max((str(s.index[-1])[:10] for s in series_by_asset.values()), default=None)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if readable_results else "no_data",
        "data_source": "cache local data/cache; Yahoo Finance seulement avec --refresh",
        "data_end": data_end,
        "oos_fraction": OOS_FRACTION,
        "cost_per_exposure_change": ROUND_TRIP_COST,
        "assets": sorted(series_by_asset),
        "compiled_robot_count": len(robots),
        "compiled_robots_tested": 0,
        "compiled_robots": robots,
        "native_strategies": _summary(readable_results, specs),
        "native_results": readable_results,
        "forecasts": forecasts,
        "resource_packs": RESOURCE_PACKS,
        "limitations": [
            "Les 114 lignes du catalogue .ex4/.ex5 sont des noms et métadonnées; aucun binaire n'est embarqué dans le dépôt ou le serveur Linux.",
            "Les robots compilés doivent être exécutés dans MetaTrader Strategy Tester avec leur symbole, période, paramètres et historique correspondants.",
            "La famille et les particularités des EA compilés sont des hypothèses déduites du nom, jamais une lecture du code.",
            "Les stratégies Python sont comparées sur les mêmes données journalières, avec coûts de 0,1% par variation d'exposition et un test chronologique final de 30%.",
            "La prévision ML est une estimation probabiliste à un pas; ce n'est pas une certitude ni un conseil d'investissement.",
            "Les archives Freqtrade, NautilusTrader, TradingAgents et OpenBB sont des frameworks/outils, pas quatre EA prêts à comparer tels quels.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Rex robot research dashboard data")
    parser.add_argument("--refresh", action="store_true", help="refresh daily prices from Yahoo Finance before tests")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = generate_dashboard_data(args.refresh, args.output)
    print(f"Dashboard data: {args.output}")
    print(f"Catalogued compiled robots: {data['compiled_robot_count']}; tested compiled robots: 0")
    print(f"Native results: {len(data['native_results'])}; forecasts: {len(data['forecasts'])}")
    print(f"Market data through: {data['data_end']}")


if __name__ == "__main__":
    main()
