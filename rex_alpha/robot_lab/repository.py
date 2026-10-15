"""SQLite persistence for robot metadata, market data versions, and test runs."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import BacktestSpec, MarketDataset, RobotArtifact


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RobotLabStore:
    """Store reproducible research records separately from the trading portfolio."""

    def __init__(self, path: str | Path = "data/rex_alpha/robot_lab.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def _initialize(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS robot_artifacts (
                robot_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                platform TEXT NOT NULL CHECK(platform IN ('MT4', 'MT5')),
                source_path TEXT NOT NULL,
                sha256 TEXT,
                role TEXT NOT NULL DEFAULT 'unverified',
                market_hint TEXT,
                timeframe_hint TEXT,
                documentation TEXT,
                license_status TEXT NOT NULL DEFAULT 'unknown',
                dependencies_json TEXT NOT NULL DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS market_datasets (
                dataset_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                file_format TEXT NOT NULL,
                source_path TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                rows INTEGER,
                has_bid_ask INTEGER NOT NULL DEFAULT 0,
                timezone_name TEXT,
                registered_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id TEXT PRIMARY KEY,
                robot_id TEXT NOT NULL REFERENCES robot_artifacts(robot_id),
                dataset_id TEXT NOT NULL REFERENCES market_datasets(dataset_id),
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                start_date TEXT NOT NULL,
                development_end TEXT NOT NULL,
                end_date TEXT NOT NULL,
                tick_model TEXT NOT NULL,
                spread_points REAL,
                slippage_points REAL NOT NULL,
                commission_per_lot REAL NOT NULL,
                initial_deposit REAL NOT NULL,
                currency TEXT NOT NULL,
                leverage TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'completed', 'failed', 'blocked')),
                metrics_json TEXT,
                report_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_robot_runs ON backtest_runs(robot_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_dataset_runs ON backtest_runs(dataset_id, status);
            """
        )
        self.conn.commit()

    def register_robot(self, artifact: RobotArtifact) -> None:
        self.conn.execute(
            """INSERT INTO robot_artifacts (
                robot_id, file_name, platform, source_path, sha256, role,
                market_hint, timeframe_hint, documentation, license_status,
                dependencies_json, notes, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(robot_id) DO UPDATE SET
                file_name=excluded.file_name, platform=excluded.platform,
                source_path=excluded.source_path, sha256=excluded.sha256,
                role=excluded.role, market_hint=excluded.market_hint,
                timeframe_hint=excluded.timeframe_hint,
                documentation=excluded.documentation,
                license_status=excluded.license_status,
                dependencies_json=excluded.dependencies_json,
                notes=excluded.notes, updated_at=excluded.updated_at""",
            (
                artifact.robot_id, artifact.file_name, artifact.platform,
                artifact.source_path, artifact.sha256, artifact.role,
                artifact.market_hint, artifact.timeframe_hint,
                artifact.documentation, artifact.license_status,
                json.dumps(artifact.dependencies), artifact.notes, _now(),
            ),
        )
        self.conn.commit()

    def register_dataset(self, dataset: MarketDataset) -> None:
        self.conn.execute(
            """INSERT INTO market_datasets (
                dataset_id, symbol, timeframe, file_format, source_path,
                start_date, end_date, sha256, rows, has_bid_ask,
                timezone_name, registered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                symbol=excluded.symbol, timeframe=excluded.timeframe,
                file_format=excluded.file_format, source_path=excluded.source_path,
                start_date=excluded.start_date, end_date=excluded.end_date,
                sha256=excluded.sha256, rows=excluded.rows,
                has_bid_ask=excluded.has_bid_ask,
                timezone_name=excluded.timezone_name,
                registered_at=excluded.registered_at""",
            (
                dataset.dataset_id, dataset.symbol, dataset.timeframe,
                dataset.file_format, dataset.source_path,
                dataset.start_date.isoformat(), dataset.end_date.isoformat(),
                dataset.sha256, dataset.rows, int(dataset.has_bid_ask),
                dataset.timezone_name, _now(),
            ),
        )
        self.conn.commit()

    def queue_run(self, spec: BacktestSpec, run_id: str | None = None) -> str:
        dataset = self.conn.execute(
            "SELECT symbol, timeframe, start_date, end_date FROM market_datasets WHERE dataset_id=?",
            (spec.dataset_id,),
        ).fetchone()
        if dataset is None:
            raise ValueError(f"unknown market dataset: {spec.dataset_id}")
        if (dataset["symbol"], dataset["timeframe"]) != (spec.symbol, spec.timeframe):
            raise ValueError("test symbol/timeframe must match the registered dataset")
        if spec.start_date.isoformat() < dataset["start_date"] or spec.end_date.isoformat() > dataset["end_date"]:
            raise ValueError("test period must be fully covered by the registered dataset")

        run_id = run_id or str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO backtest_runs (
                run_id, robot_id, dataset_id, symbol, timeframe, start_date,
                development_end, end_date, tick_model, spread_points,
                slippage_points, commission_per_lot, initial_deposit, currency,
                leverage, parameters_json, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)""",
            (
                run_id, spec.robot_id, spec.dataset_id, spec.symbol,
                spec.timeframe, spec.start_date.isoformat(),
                spec.development_end.isoformat(), spec.end_date.isoformat(),
                spec.tick_model, spec.spread_points, spec.slippage_points,
                spec.commission_per_lot, spec.initial_deposit, spec.currency,
                spec.leverage, json.dumps(spec.parameters, sort_keys=True), _now(),
            ),
        )
        self.conn.commit()
        return run_id

    def mark_running(self, run_id: str) -> None:
        self._transition(run_id, "queued", "running")

    def complete_run(
        self,
        run_id: str,
        metrics: dict[str, Any],
        report_path: str | None = None,
    ) -> None:
        cursor = self.conn.execute(
            """UPDATE backtest_runs
               SET status='completed', metrics_json=?, report_path=?, completed_at=?
               WHERE run_id=? AND status='running'""",
            (json.dumps(metrics, sort_keys=True), report_path, _now(), run_id),
        )
        if cursor.rowcount != 1:
            self.conn.rollback()
            raise ValueError("run must exist and be running before completion")
        self.conn.commit()

    def fail_run(self, run_id: str, error: str, blocked: bool = False) -> None:
        cursor = self.conn.execute(
            """UPDATE backtest_runs
               SET status=?, error=?, completed_at=?
               WHERE run_id=? AND status IN ('queued', 'running')""",
            ("blocked" if blocked else "failed", error, _now(), run_id),
        )
        if cursor.rowcount != 1:
            self.conn.rollback()
            raise ValueError("run must exist and be queued or running")
        self.conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM backtest_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return self._decode_run(row)

    def list_runs(self, robot_id: str | None = None) -> list[dict[str, Any]]:
        if robot_id is None:
            rows = self.conn.execute(
                "SELECT * FROM backtest_runs ORDER BY created_at, run_id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM backtest_runs WHERE robot_id=? ORDER BY created_at, run_id",
                (robot_id,),
            ).fetchall()
        return [self._decode_run(row) for row in rows]

    def list_robots(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM robot_artifacts ORDER BY platform, file_name"
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["dependencies"] = json.loads(item.pop("dependencies_json"))
            result.append(item)
        return result

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RobotLabStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _decode_run(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["parameters"] = json.loads(result.pop("parameters_json"))
        metrics_json = result.pop("metrics_json")
        result["metrics"] = json.loads(metrics_json) if metrics_json else None
        return result

    def _transition(self, run_id: str, old: str, new: str) -> None:
        cursor = self.conn.execute(
            "UPDATE backtest_runs SET status=? WHERE run_id=? AND status=?",
            (new, run_id, old),
        )
        if cursor.rowcount != 1:
            self.conn.rollback()
            raise ValueError(f"run must exist in '{old}' state")
        self.conn.commit()
