"""
Module de rapport — Rex Clemens Capital
═══════════════════════════════════════
Rapport clair avec verdict réussi / échoué.
Chaque rapport mentionne les limites :
  - Données passées (pas une garantie du futur)
  - Frais estimés (les vrais frais peuvent être différents)
  - Risque de surajustement (overfitting)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .backtest import ResultatBacktest
from .stress import ResultatStress, TypeScenario


# ═══════════════════════════════════════════════════════════════
# Rapport de backtest
# ═══════════════════════════════════════════════════════════════

def rapport_backtest(
    resultats: list[ResultatBacktest],
    titre: str = "Rapport de backtest",
) -> str:
    """Génère un rapport lisible pour une série de backtests."""
    lignes = [
        f"{'═' * 60}",
        f"  {titre}",
        f"  {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"{'═' * 60}",
        "",
    ]

    # Tableau des résultats
    lignes.append(f"{'Stratégie':20s} │ {'Rdt/an':>8s} │ {'Sharpe':>7s} │ {'Max DD':>8s} │ {'Trades':>6s} │ {'Viol.':>5s}")
    lignes.append("─" * 70)

    for r in resultats:
        lignes.append(
            f"{r.nom_strategie:20s} │ {r.rendement_annuel:>7.1%} │ {r.sharpe:>7.2f} │ "
            f"{r.pire_baisse:>7.1%} │ {r.nb_trades:>6d} │ {r.violations_survie:>5d}"
        )

    lignes.append("")

    # Avertissements obligatoires
    lignes.extend(_avertissements())

    return "\n".join(lignes)


# ═══════════════════════════════════════════════════════════════
# Rapport de stress
# ═══════════════════════════════════════════════════════════════

def rapport_stress(resultat: ResultatStress) -> str:
    """Génère un rapport complet du banc de stress."""
    verdict = "✅ RÉUSSI" if resultat.passe else "❌ ÉCHOUÉ"

    lignes = [
        f"{'═' * 60}",
        f"  Banc de stress — {resultat.nom_strategie}",
        f"  {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"{'═' * 60}",
        "",
        f"  VERDICT : {verdict}",
        f"  Scénarios réussis : {resultat.nb_reussis}/{resultat.nb_total} "
        f"({resultat.taux_reussite:.1%}, seuil {resultat.seuil_reussite:.0%})",
        "",
    ]

    # Détail par type de scénario
    for type_sc in TypeScenario:
        sc_type = [s for s in resultat.scenarios if s.type == type_sc]
        if not sc_type:
            continue

        nom_type = {
            TypeScenario.CRISE_HISTORIQUE: "🔥 Crises historiques",
            TypeScenario.MONTE_CARLO: "🎲 Monte Carlo",
            TypeScenario.CHOC_SYNTHETIQUE: "💥 Chocs synthétiques",
        }[type_sc]

        n_ok = sum(1 for s in sc_type if s.passe)
        lignes.append(f"  {nom_type} : {n_ok}/{len(sc_type)} réussis")

        # Pour les crises et chocs, détailler chaque scénario
        if type_sc != TypeScenario.MONTE_CARLO:
            for s in sc_type:
                icone = "✅" if s.passe else "❌"
                lignes.append(
                    f"    {icone} {s.nom:40s} │ "
                    f"capital {s.capital_final_pct:>6.1%} │ "
                    f"max DD {s.pire_baisse:>6.1%} │ "
                    f"violations {s.violations}"
                )
        else:
            # Monte Carlo : résumé statistique
            caps = [s.capital_final_pct for s in sc_type]
            dds = [s.pire_baisse for s in sc_type]
            import numpy as np
            lignes.append(
                f"    Capital final — "
                f"médiane {np.median(caps):.1%}, "
                f"min {min(caps):.1%}, "
                f"max {max(caps):.1%}, "
                f"5e pctile {np.percentile(caps, 5):.1%}"
            )
            lignes.append(
                f"    Pire baisse   — "
                f"médiane {np.median(dds):.1%}, "
                f"min {min(dds):.1%}, "
                f"max {max(dds):.1%}"
            )

        lignes.append("")

    # Pire scénario
    if resultat.pire_scenario:
        p = resultat.pire_scenario
        lignes.append(
            f"  ⚠ Pire scénario : {p.nom}\n"
            f"    Capital final {p.capital_final_pct:.1%}, "
            f"pire baisse {p.pire_baisse:.1%}, "
            f"violations {p.violations}"
        )
        lignes.append("")

    # Avertissements
    lignes.extend(_avertissements())

    return "\n".join(lignes)


# ═══════════════════════════════════════════════════════════════
# Avertissements obligatoires
# ═══════════════════════════════════════════════════════════════

def _avertissements() -> list[str]:
    """Avertissements à inclure dans chaque rapport."""
    return [
        "─" * 60,
        "  ⚠ LIMITES — À LIRE",
        "  • Ces résultats portent sur des données PASSÉES (simulées).",
        "    Ils ne sont PAS une promesse de gain futur.",
        "  • Les frais sont ESTIMÉS. Les vrais frais de courtage,",
        "    le glissement et la liquidité réels peuvent être différents.",
        "  • Risque de SURAJUSTEMENT : une stratégie optimisée sur",
        "    le passé peut échouer sur des données inédites.",
        "  • Aucun argent réel n'est engagé dans ces tests.",
        "─" * 60,
    ]
