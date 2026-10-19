"""
REX CLEMENS — Gene Hunter
===========================
Moteur d'evolution genetique d'indicateurs de trading.
Rex ne copie pas les formules existantes. Il en invente de nouvelles.

Principe (inspire de Jim Simons / Renaissance Technologies) :
1. Definir des "genes" = operations mathematiques de base
2. Combiner au hasard en "chromosomes" = formules inedites
3. Tester chaque formule sur l'historique
4. Garder les meilleurs, muter, croiser
5. Au fil du temps, Rex developpe son propre ADN unique

La defaite n'est pas une option. Rex cherche jusqu'a trouver.
"""

import json
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


ADN_PATH = Path("data/rex_adn.json")


GENES = {
    "prix": lambda closes, highs, lows, volumes, i: closes[i],
    "volume": lambda closes, highs, lows, volumes, i: volumes[i] if volumes[i] > 0 else 1,
    "range": lambda closes, highs, lows, volumes, i: highs[i] - lows[i] if highs[i] > lows[i] else 0.001,
    "momentum_5": lambda closes, highs, lows, volumes, i: closes[i] / closes[i-5] - 1 if i >= 5 and closes[i-5] > 0 else 0,
    "momentum_10": lambda closes, highs, lows, volumes, i: closes[i] / closes[i-10] - 1 if i >= 10 and closes[i-10] > 0 else 0,
    "momentum_20": lambda closes, highs, lows, volumes, i: closes[i] / closes[i-20] - 1 if i >= 20 and closes[i-20] > 0 else 0,
    "ma_ratio_5": lambda closes, highs, lows, volumes, i: closes[i] / (sum(closes[i-5:i]) / 5) - 1 if i >= 5 else 0,
    "ma_ratio_20": lambda closes, highs, lows, volumes, i: closes[i] / (sum(closes[i-20:i]) / 20) - 1 if i >= 20 else 0,
    "ma_ratio_50": lambda closes, highs, lows, volumes, i: closes[i] / (sum(closes[i-50:i]) / 50) - 1 if i >= 50 else 0,
    "vol_ratio": lambda closes, highs, lows, volumes, i: volumes[i] / (sum(volumes[i-20:i]) / 20) - 1 if i >= 20 and sum(volumes[i-20:i]) > 0 else 0,
    "high_low_pct": lambda closes, highs, lows, volumes, i: (highs[i] - lows[i]) / closes[i] if closes[i] > 0 else 0,
    "close_position": lambda closes, highs, lows, volumes, i: (closes[i] - lows[i]) / (highs[i] - lows[i]) if highs[i] > lows[i] else 0.5,
    "std_10": lambda closes, highs, lows, volumes, i: float(np.std(closes[i-10:i])) / closes[i] if i >= 10 and closes[i] > 0 else 0,
    "std_20": lambda closes, highs, lows, volumes, i: float(np.std(closes[i-20:i])) / closes[i] if i >= 20 and closes[i] > 0 else 0,
    "gap": lambda closes, highs, lows, volumes, i: closes[i] / closes[i-1] - 1 if i >= 1 and closes[i-1] > 0 else 0,
    "max_20_dist": lambda closes, highs, lows, volumes, i: closes[i] / max(closes[i-20:i]) - 1 if i >= 20 and max(closes[i-20:i]) > 0 else 0,
    "min_20_dist": lambda closes, highs, lows, volumes, i: closes[i] / min(closes[i-20:i]) - 1 if i >= 20 and min(closes[i-20:i]) > 0 else 0,
}

OPERATIONS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b if abs(b) > 1e-10 else 0,
    "max": lambda a, b: max(a, b),
    "min": lambda a, b: min(a, b),
    "abs_diff": lambda a, b: abs(a - b),
    "avg": lambda a, b: (a + b) / 2,
    "ratio": lambda a, b: a / b if abs(b) > 1e-10 else 0,
}

TRANSFORMATIONS = {
    "none": lambda x: x,
    "abs": lambda x: abs(x),
    "neg": lambda x: -x,
    "sign": lambda x: 1 if x > 0 else (-1 if x < 0 else 0),
    "clip": lambda x: max(-1, min(1, x)),
    "square": lambda x: x * x if abs(x) < 100 else 0,
    "sqrt_abs": lambda x: np.sqrt(abs(x)) * (1 if x >= 0 else -1),
}


@dataclass
class Chromosome:
    gene_a: str
    gene_b: str
    operation: str
    transform: str
    nom: str = ""
    fitness: float = 0.0
    win_rate: float = 0.0
    nb_trades: int = 0
    sharpe: float = 0.0
    generation: int = 0
    naissance: str = ""
    id: str = ""

    def __post_init__(self):
        if not self.nom:
            self.nom = f"RX_{self.gene_a[:3]}_{self.operation[:3]}_{self.gene_b[:3]}_{self.transform[:3]}"
        if not self.naissance:
            self.naissance = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        if not self.id:
            self.id = f"{random.randint(1000,9999)}"

    def evaluer(self, closes, highs, lows, volumes, i) -> float:
        try:
            gene_fn_a = GENES[self.gene_a]
            gene_fn_b = GENES[self.gene_b]
            op_fn = OPERATIONS[self.operation]
            trans_fn = TRANSFORMATIONS[self.transform]
            val_a = gene_fn_a(closes, highs, lows, volumes, i)
            val_b = gene_fn_b(closes, highs, lows, volumes, i)
            result = op_fn(val_a, val_b)
            return trans_fn(result)
        except (IndexError, ZeroDivisionError, ValueError):
            return 0.0


def generer_chromosome(generation: int = 0) -> Chromosome:
    return Chromosome(
        gene_a=random.choice(list(GENES.keys())),
        gene_b=random.choice(list(GENES.keys())),
        operation=random.choice(list(OPERATIONS.keys())),
        transform=random.choice(list(TRANSFORMATIONS.keys())),
        generation=generation,
    )


def muter(parent: Chromosome, generation: int) -> Chromosome:
    enfant = Chromosome(
        gene_a=parent.gene_a,
        gene_b=parent.gene_b,
        operation=parent.operation,
        transform=parent.transform,
        generation=generation,
    )
    mutation = random.choice(["gene_a", "gene_b", "operation", "transform"])
    if mutation == "gene_a":
        enfant.gene_a = random.choice(list(GENES.keys()))
    elif mutation == "gene_b":
        enfant.gene_b = random.choice(list(GENES.keys()))
    elif mutation == "operation":
        enfant.operation = random.choice(list(OPERATIONS.keys()))
    else:
        enfant.transform = random.choice(list(TRANSFORMATIONS.keys()))
    enfant.nom = ""
    enfant.__post_init__()
    return enfant


def croiser(parent_a: Chromosome, parent_b: Chromosome, generation: int) -> Chromosome:
    return Chromosome(
        gene_a=random.choice([parent_a.gene_a, parent_b.gene_a]),
        gene_b=random.choice([parent_a.gene_b, parent_b.gene_b]),
        operation=random.choice([parent_a.operation, parent_b.operation]),
        transform=random.choice([parent_a.transform, parent_b.transform]),
        generation=generation,
    )


def backtester_chromosome(chrom: Chromosome, closes, highs, lows, volumes,
                          lookback: int = 50, horizon: int = 5, seuil: float = 0.0) -> dict:
    if len(closes) < lookback + horizon + 60:
        return {"fitness": -999, "win_rate": 0, "nb_trades": 0, "sharpe": 0}

    signals = []
    start = max(60, lookback)
    end = len(closes) - horizon

    for i in range(start, end):
        val = chrom.evaluer(closes, highs, lows, volumes, i)
        if val > seuil:
            rendement = closes[i + horizon] / closes[i] - 1
            signals.append(rendement)

    if len(signals) < 10:
        return {"fitness": -999, "win_rate": 0, "nb_trades": 0, "sharpe": 0}

    gains = [r for r in signals if r > 0]
    pertes = [r for r in signals if r <= 0]
    win_rate = len(gains) / len(signals) * 100
    mean_r = np.mean(signals)
    std_r = np.std(signals)
    sharpe = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0
    fitness = sharpe * win_rate / 100

    return {
        "fitness": round(fitness, 4),
        "win_rate": round(win_rate, 1),
        "nb_trades": len(signals),
        "sharpe": round(sharpe, 2),
        "mean_return": round(mean_r * 100, 3),
    }


class GeneHunter:
    def __init__(self, taille_population: int = 50, elite_pct: float = 0.2):
        self.taille_pop = taille_population
        self.elite_pct = elite_pct
        self.generation = 0
        self.population: list[Chromosome] = []
        self.hall_of_fame: list[Chromosome] = []
        self.charger_adn()

    def charger_adn(self):
        if ADN_PATH.exists():
            try:
                with open(ADN_PATH) as f:
                    data = json.load(f)
                self.generation = data.get("generation", 0)
                self.hall_of_fame = [Chromosome(**c) for c in data.get("hall_of_fame", [])]
                self.population = [Chromosome(**c) for c in data.get("population", [])]
                print(f"[GENE] ADN charge: gen {self.generation}, {len(self.hall_of_fame)} champions, {len(self.population)} actifs")
            except Exception as e:
                print(f"[GENE] Erreur chargement ADN: {e}")

    def sauver_adn(self):
        ADN_PATH.parent.mkdir(exist_ok=True)
        data = {
            "generation": self.generation,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "hall_of_fame": [asdict(c) for c in self.hall_of_fame[:20]],
            "population": [asdict(c) for c in self.population],
        }
        with open(ADN_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def evoluer(self, closes, highs, lows, volumes) -> list[Chromosome]:
        self.generation += 1
        print(f"[GENE] === Generation {self.generation} ===")

        if len(self.population) < self.taille_pop:
            nouveaux = self.taille_pop - len(self.population)
            for _ in range(nouveaux):
                self.population.append(generer_chromosome(self.generation))

        for chrom in self.population:
            result = backtester_chromosome(chrom, closes, highs, lows, volumes)
            chrom.fitness = result["fitness"]
            chrom.win_rate = result["win_rate"]
            chrom.nb_trades = result["nb_trades"]
            chrom.sharpe = result["sharpe"]

        self.population.sort(key=lambda c: c.fitness, reverse=True)

        nb_elite = max(5, int(len(self.population) * self.elite_pct))
        elites = self.population[:nb_elite]

        for elite in elites:
            if elite.fitness > 0.5 and elite.win_rate > 52:
                if not any(h.nom == elite.nom for h in self.hall_of_fame):
                    self.hall_of_fame.append(elite)
                    print(f"[GENE] DECOUVERTE: {elite.nom} | fitness={elite.fitness:.3f} "
                          f"win={elite.win_rate:.0f}% sharpe={elite.sharpe:.2f} trades={elite.nb_trades}")

        self.hall_of_fame.sort(key=lambda c: c.fitness, reverse=True)
        self.hall_of_fame = self.hall_of_fame[:20]

        nouvelle_pop = list(elites)

        while len(nouvelle_pop) < self.taille_pop:
            r = random.random()
            if r < 0.3:
                nouvelle_pop.append(generer_chromosome(self.generation))
            elif r < 0.6:
                parent = random.choice(elites)
                nouvelle_pop.append(muter(parent, self.generation))
            else:
                pa, pb = random.sample(elites, min(2, len(elites)))
                nouvelle_pop.append(croiser(pa, pb, self.generation))

        self.population = nouvelle_pop

        best = self.population[0] if self.population else None
        if best:
            print(f"[GENE] Meilleur gen {self.generation}: {best.nom} "
                  f"| fitness={best.fitness:.3f} win={best.win_rate:.0f}% sharpe={best.sharpe:.2f}")

        if self.hall_of_fame:
            champion = self.hall_of_fame[0]
            print(f"[GENE] Champion absolu: {champion.nom} "
                  f"| fitness={champion.fitness:.3f} win={champion.win_rate:.0f}% "
                  f"sharpe={champion.sharpe:.2f} (gen {champion.generation})")

        self.sauver_adn()
        return self.hall_of_fame

    def get_signal(self, closes, highs, lows, volumes, i: int, top_n: int = 5) -> float:
        if not self.hall_of_fame:
            return 0.0
        scores = []
        for chrom in self.hall_of_fame[:top_n]:
            try:
                val = chrom.evaluer(closes, highs, lows, volumes, i)
                scores.append(1.0 if val > 0 else 0.0)
            except:
                pass
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def get_adn_resume(self) -> dict:
        return {
            "generation": self.generation,
            "nb_champions": len(self.hall_of_fame),
            "nb_population": len(self.population),
            "champion": self.hall_of_fame[0].nom if self.hall_of_fame else "aucun",
            "champion_fitness": self.hall_of_fame[0].fitness if self.hall_of_fame else 0,
            "champion_winrate": self.hall_of_fame[0].win_rate if self.hall_of_fame else 0,
            "formules_uniques": len(set(c.nom for c in self.hall_of_fame)),
        }
