"""Seed the research registry from the checked-in robot test matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import import_test_matrix
from .repository import RobotLabStore


def main() -> None:
    package_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Import the Rex robot test matrix")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=package_dir / "robot_test_matrix.csv",
        help="path to robot_test_matrix.csv",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/rex_alpha/robot_lab.db"),
        help="SQLite registry path",
    )
    args = parser.parse_args()

    with RobotLabStore(args.database) as store:
        imported = import_test_matrix(args.matrix, store)
        print(f"Registered {imported} robot artifacts in {args.database}")


if __name__ == "__main__":
    main()
