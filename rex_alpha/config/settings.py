"""
Configuration centrale — Rex Alpha Engine
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RexConfig:
    # Capital
    capital_initial: float = 100_000.0
    devise: str = "USD"

    # Univers
    univers_min_prix: float = 5.0
    univers_min_volume_moyen: int = 500_000
    univers_min_market_cap: float = 500_000_000

    # Horizons d'évaluation (en séances)
    horizons: list[int] = field(default_factory=lambda: [20, 40, 60])

    # Allocation
    max_positions: int = 20
    max_poids_par_titre: float = 0.10
    max_exposition_totale: float = 0.95

    # Risque
    max_drawdown_pct: float = 15.0
    max_perte_jour_pct: float = 5.0
    max_risque_par_trade_pct: float = 2.0
    levier_max: float = 1.0
    stop_loss_pct: float = 8.0

    # Frais
    frais_par_trade_pct: float = 0.10
    slippage_pct: float = 0.05

    # Données
    historique_min_jours: int = 504
    donnees_dir: Path = field(default_factory=lambda: Path("data/rex_alpha"))

    # Modèle
    taille_train_min: int = 252
    reentrain_tous_les: int = 21
    random_seed: int = 42

    # Référence
    benchmark: str = "SPY"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8888

    def __post_init__(self):
        self.donnees_dir = Path(self.donnees_dir)
        self.donnees_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = RexConfig()
