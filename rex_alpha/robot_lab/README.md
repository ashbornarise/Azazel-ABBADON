# Rex Robot Lab

The lab compares readable Python strategies on shared daily price series and provides a catalog for the 114 compiled MT4/MT5 experts. The live server serves the interface at `/robot-lab`.

## Generate results and forecasts

From the repository root:

```bash
source venv/bin/activate
python -m rex_alpha.robot_lab.research --refresh
```

This refreshes available daily Yahoo Finance histories, runs a chronological 70/30 walk-forward comparison with the final 30% held out, and writes `data/rex_alpha/robot_lab_dashboard.json`. It compares Buy & Hold, MA 20/100, MA 5/20, Random Forest, and a random-control distribution from ten seeds. All strategies use the same assets, 0.1% cost per unit of exposure change, and no leverage. Forecasts are one-step probability estimates, not certainties.

The precomputed JSON is served through `/api/robot-lab/state`; `/robot-lab` is the read-only dashboard. The research command must be run after a new deployment or when you want refreshed results.

## Import MetaTrader reports

In the dashboard, select the matching EA, choose whether the HTML report is a development run or a final out-of-sample run, and import the Strategy Tester `.htm`/`.html` file. The browser extracts recognized metrics and stores them in that browser's local storage; it does not upload the report or the EA binary to the server. Use **Exporter les imports JSON** to save a portable copy. The OOS counter includes only robots with an imported OOS report. Compare reports only when symbol, timeframe, date range, preset, and broker costs match; the inferred family labels do not reveal the compiled strategy's actual logic.

## Compiled MT4/MT5 experts

The CSV matrix contains 114 filename records. `.ex4`/`.ex5` files are compiled binaries; this Python lab does not inspect or execute them. The family labels in the dashboard are weak filename-based hints, not verified descriptions. Testing an expert requires its platform's Strategy Tester, a suitable symbol and timeframe history, the correct preset and broker costs, and an exported tester report. A robot remains untested until that report is recorded.

The inventory found six `.ex5` files in the local MT5 Experts directory, but the binaries are not part of this repository and are not present on the Linux dashboard server. The existing price cache is daily data and has no XAUUSD M1/tick series, so it cannot validate the many Gold scalpers against their intended execution conditions.

## ZIP research resources

Freqtrade and NautilusTrader are execution/backtest frameworks, TradingAgents is a multi-agent research project, and OpenBB is a data platform. None of those archives alone supplies a validated Rex trading strategy or comparable test result. Their roles and missing prerequisites are shown separately in the dashboard.
