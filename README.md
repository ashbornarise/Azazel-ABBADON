# 🤖 Azazel-ABBADON — Rex Clemens Capital

Robot de trading éducatif. **La survie du compte passe avant le gain.**

## Principe

Le robot ne prédit pas le prix. Il estime avec prudence, limite ce qu'il risque, coupe ses pertes et mesure en continu s'il se trompe plus que prévu.

> Un robot qui perd tout est mort ; un robot qui ne gagne pas peut encore apprendre.

## Rex Alpha paper trading

Le tableau de bord Rex Alpha affiche les cours intraday, les scores calculés sur les dernières séances clôturées, la courbe du portefeuille virtuel et les opérations simulées.

```powershell
pip install pandas numpy scikit-learn yfinance tzdata
python run_live.py
```

Ouvre ensuite `http://localhost:8777/`. La première veille suit `AAPL`, `MSFT`, `NVDA` et `SPY`; la liste se remplace avec `REX_ALPHA_LIVE_SYMBOLS` (symboles séparés par des virgules). Les bougies d'une minute viennent de Yahoo Finance via `yfinance` et sont interrogées toutes les 60 secondes. Leur fraîcheur dépend du fournisseur; le tableau affiche l'horodatage reçu. Les ordres restent dans le simulateur virtuel. L'ancien tableau de bord est accessible sur `/legacy`.

## Architecture

```
config.toml              ← seuils de survie (jamais codés en dur)
data/                    ← données en cache
src/
  survie.py              ← MODULE 1 : gardien du compte
  watchdog.py            ← arrêt d'urgence indépendant
  backtest.py            ← MODULE 3 : moteur de test
  stress.py              ← MODULE 2 : banc de stress (Phase 2)
  data.py                ← chargement et qualité des données
  strategies/
    base.py              ← interface commune
    buy_hold.py           ← référence passive
    ma_cross.py          ← croisement de moyennes mobiles
    random_control.py    ← témoin aléatoire (ne doit JAMAIS avoir l'air bon)
  report.py              ← rapport avec verdict (Phase 2)
tests/                   ← tests automatiques de chaque règle
```

## Règles de survie

| Règle | Seuil |
|---|---|
| Risque max par trade | 1 % du compte |
| Levier | Interdit (expo max 1.0) |
| Perte max journalière | 3 % → arrêt jusqu'au lendemain |
| Drawdown max | 10 % → arrêt total (reset par Clément) |
| Taille max position | 25 % du compte |
| Données invalides | → état liquide |
| Connexion perdue | → état liquide |

## Exécution des tests

```bash
pip install pytest numpy pandas
pytest tests/ -v
```

## ⚠️ Avertissement

- **Aucun argent réel**, aucune connexion courtier, aucune clé API.
- Un résultat de test n'est **pas** une promesse de gain.
- Données passées, frais estimés, risque de surajustement.

## Feuille de route

- [x] Phase 0 : Squelette du dépôt, config.toml, données
- [x] Phase 1 : survie.py, watchdog.py et tests
- [x] Phase 2 : Banc de stress (stress.py, rejeu crises, Monte Carlo, chocs, rapport)
- [ ] Phase 3 : Stratégies + témoin aléatoire (verdict honnête)
- [ ] Phase 4 : Trading sur papier (3-6 mois)
- [ ] Phase 5 : Réel (décision Clément uniquement)
