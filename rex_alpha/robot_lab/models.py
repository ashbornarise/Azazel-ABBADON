"""Typed records for cataloguing robot artifacts and simulation inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


PLATFORMS = {"MT4", "MT5"}
TIMEFRAMES = {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}
TICK_MODELS = {"real_ticks", "generated_ticks", "ohlc_1m"}


@dataclass(frozen=True)
class RobotArtifact:
    robot_id: str
    file_name: str
    platform: str
    source_path: str
    sha256: str | None = None
    role: str = "unverified"
    market_hint: str | None = None
    timeframe_hint: str | None = None
    documentation: str | None = None
    license_status: str = "unknown"
    dependencies: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.robot_id.strip():
            raise ValueError("robot_id is required")
        if self.platform not in PLATFORMS:
            raise ValueError(f"platform must be one of {sorted(PLATFORMS)}")
        if not self.file_name.strip():
            raise ValueError("file_name is required")
        expected_platform = {".ex4": "MT4", ".ex5": "MT5"}.get(
            Path(self.file_name).suffix.lower()
        )
        if expected_platform and expected_platform != self.platform:
            raise ValueError("file extension does not match the declared platform")


@dataclass(frozen=True)
class MarketDataset:
    dataset_id: str
    symbol: str
    timeframe: str
    file_format: str
    source_path: str
    start_date: date
    end_date: date
    sha256: str
    rows: int | None = None
    has_bid_ask: bool = False
    timezone_name: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.symbol.strip():
            raise ValueError("dataset_id and symbol are required")
        if self.timeframe not in TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {sorted(TIMEFRAMES)}")
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in self.sha256):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        if self.rows is not None and self.rows < 0:
            raise ValueError("rows cannot be negative")


@dataclass(frozen=True)
class BacktestSpec:
    robot_id: str
    dataset_id: str
    symbol: str
    timeframe: str
    start_date: date
    end_date: date
    development_end: date
    tick_model: str = "real_ticks"
    spread_points: float | None = None
    slippage_points: float = 0.0
    commission_per_lot: float = 0.0
    initial_deposit: float = 10_000.0
    currency: str = "USD"
    leverage: str = "1:1"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.robot_id.strip() or not self.dataset_id.strip() or not self.symbol.strip():
            raise ValueError("robot_id and dataset_id are required")
        if self.timeframe not in TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {sorted(TIMEFRAMES)}")
        if self.tick_model not in TICK_MODELS:
            raise ValueError(f"tick_model must be one of {sorted(TICK_MODELS)}")
        if not self.start_date < self.development_end < self.end_date:
            raise ValueError("dates must define development and out-of-sample periods")
        if self.spread_points is not None and self.spread_points < 0:
            raise ValueError("spread_points cannot be negative")
        if self.slippage_points < 0 or self.commission_per_lot < 0:
            raise ValueError("slippage and commission cannot be negative")
        if self.initial_deposit <= 0:
            raise ValueError("initial_deposit must be positive")
