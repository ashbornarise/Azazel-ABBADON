"""Research registry and reproducible test protocol for imported trading robots."""

from .models import BacktestSpec, MarketDataset, RobotArtifact
from .repository import RobotLabStore

__all__ = ["BacktestSpec", "MarketDataset", "RobotArtifact", "RobotLabStore"]
