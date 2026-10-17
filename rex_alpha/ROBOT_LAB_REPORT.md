# Rex Robot Lab — Rapport Consolidé
## Extraction de stratégies depuis 130+ robots de trading
**Date** : 2026-10-17 | **Projet** : Rex Clemens Capital

---

## Sources analysées

| Source | Fichiers | Type | Statut |
|---|---|---|---|
| freqtrade-develop | 887 | Python/Cython | Analysé (6 stratégies) |
| TradingAgents-main | 232 | Python/LangGraph | Analysé (10 agents) |
| python-trading-robot | ~50 | Python | Analysé (15 indicateurs) |
| pytrader | ~30 | Python | Analysé (9 classifieurs ML) |
| BitMEX robot | ~10 | Python | Analysé (MACD cross) |
| MT5 sources (MetaQuotes) | 30+ | MQL5 (.mq5) | Analysé (9 stratégies) |
| Robots Telegram | 115 | MT5 (.ex5/.ex4) | Catalogué (binaires chiffrés) |
| Presets MT4/MT5/cTrader | 8 packs | .set/.zip | Copiés |
| nautilus_trader | 6183 | Python/Cython/Rust | Non analysé (institutionnel) |
| OpenBB | 4634 | Python | Non analysé (terminal données) |

---

## Stratégies extraites et implémentées

### 1. Features ML V2 (31 features) — IMPLÉMENTÉ
Sources : freqtrade, PyRobot, pytrader, MT5 Free Robots

| Feature | Source | Importance |
|---|---|---|
| adx_14 (ADX) | freqtrade reco #1 | 9-11% (top 1) |
| volume_trend | freqtrade, PyTrader | 8-10% |
| skew_20j | PyRobot | 8% |
| bb_width (Bollinger Width) | freqtrade reco #3 | 6-11% |
| jour_semaine | freqtrade reco #8 | 5-7% |
| macd_hist | freqtrade, BitMEX | 3-4% |
| stoch_k/d | freqtrade, MT5 | 3-5% |
| atr_14 | freqtrade, PyRobot | 3-4% |
| pattern_score | MT5 Free Robots | 1-2% |
| cci_20 (CCI) | MT5 Free Robots | 1-2% |
| engulfing | MT5 Free Robots | <1% |
| macd_cross | MT5 MACD Sample | <1% |

### 2. Multi-Classifier Benchmark — IMPLÉMENTÉ
Sources : freqtrade (FreqAI), pytrader (9 classifieurs)

| Classifieur | BTC-USD | ETH-USD | MSFT | SOL-USD |
|---|---|---|---|---|
| RandomForest | 49% | **52%** | **43%** | 44% |
| GradientBoosting | 56% | 44% | 43% | 48% |
| LightGBM | 52% | 48% | 41% | 56% |
| **XGBoost** | **68%** | 40% | 33% | **49%** |

Auto-sélection par LogLoss (le meilleur modèle change par actif).

### 3. Protections de Trading — IMPLÉMENTÉ
Sources : freqtrade (MaxDrawdown, StoplossGuard, CooldownPeriod), pytrader (fee filter)

- **MaxDrawdown** : arrêt total si drawdown > -5%
- **StoplossGuard** : pause 600s après 3 pertes consécutives
- **CooldownPeriod** : 300s entre trades sur même actif
- **FeeFilter** : gain doit dépasser 2x les frais

### 4. Seuils Dynamiques — IMPLÉMENTÉ
Source : freqtrade (freqai_test_strat)

- Seuil d'achat = mean + N*std (au lieu de fixe 50%)
- Seuil de vente = mean - N*std (au lieu de fixe 35%)
- Bornes de sécurité : achat [52-75%], vente [25-48%]

### 5. Candlestick Patterns — IMPLÉMENTÉ
Source : MT5 Free Robots (MetaQuotes)

- **Engulfing** (Bullish/Bearish) + RSI filtre
- **Harami** (Bullish/Bearish) + MFI filtre
- **3 White Soldiers / 3 Black Crows** + Stochastic filtre
- **Dark Cloud Cover / Piercing Line** + CCI filtre
- **MACD Cross** (MACD Sample EA) + EMA trend filtre

### 6. Patterns identifiés mais NON implémentés (backlog)

| Pattern | Source | Raison d'attente |
|---|---|---|
| Grid trading | MT5 robots (55 Gold EAs) | Interdit par règles de survie |
| Martingale | MT5 ZONE MARTI EA | Interdit (levier implicite) |
| DCA (Dollar Cost Averaging) | freqtrade HyperoptableStrategy | Compatible mais à valider |
| Débat Bull/Bear LLM | TradingAgents | Nécessite API LLM |
| Mémoire avec réflexion | TradingAgents | Architecture complexe |
| Multi-timeframe features | freqtrade FreqAI | Données intraday nécessaires |
| SVM Outlier Removal | freqtrade FreqAI | Complexité ML avancée |
| Parabolic SAR trailing | MT5 ExpertMAPSAR | Nécessite ta-lib |
| News Filter | Gold Advanced preset | Nécessite API actualités |
| Optuna hyperopt | freqtrade | À implémenter prochainement |

---

## Architecture actuelle

```
run_realtime.py (V2)
├── src/ml/features.py          → 31 features (V1: 11, V2: 31)
├── src/ml/candlestick_patterns.py → 5 patterns MT5
├── src/ml/multi_classifier.py  → 4 classifieurs (RF, GBM, LightGBM, XGBoost)
├── src/ml/protections.py       → MaxDrawdown, StoplossGuard, Cooldown, FeeFilter
├── src/strategies/ml_strategy.py → Walk-forward RF (legacy, utilisé pour exposition)
├── dashboard_mega.html         → Bougies OHLCV, 4 onglets, benchmark classifieurs
└── robots_telegram/            → 115 EA + 8 presets + 5 zips
```

---

## Prochaines étapes (priorité)

1. **Optuna hyperopt** pour optimiser les hyperparamètres des classifieurs
2. **DCA (position adjustment)** quand perte > seuil
3. **Multi-timeframe** : mêmes indicateurs sur 5min, 1h, 1j
4. **Déploiement Google Cloud** avec le moteur V2
5. **70+ US stocks** (depuis RFI) au lieu de 4 actifs
