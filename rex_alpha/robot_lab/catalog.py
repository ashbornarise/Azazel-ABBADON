"""Import the current CSV inventory into the robot research registry."""

from __future__ import annotations

import csv
from pathlib import Path

from .models import RobotArtifact
from .repository import RobotLabStore


REQUIRED_COLUMNS = {"ID", "Nom", "Plateforme", "Marche_suppose", "Origine"}


def import_test_matrix(path: str | Path, store: RobotLabStore) -> int:
    """Register the executable rows from ``robot_test_matrix.csv``.

    The inventory contains filenames and hints only, so imported roles and
    licenses remain unverified until documentation or platform behavior is
    reviewed.
    """
    matrix_path = Path(path)
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"matrix is missing required columns: {sorted(missing)}")

        count = 0
        for row in reader:
            filename = row["Nom"].strip()
            if not filename:
                continue
            store.register_robot(
                RobotArtifact(
                    robot_id=row["ID"].strip(),
                    file_name=filename,
                    platform=row["Plateforme"].strip(),
                    source_path=f"{row['Origine'].strip()}/{filename}",
                    market_hint=row["Marche_suppose"].strip() or None,
                    role="unverified",
                    license_status="unknown",
                    notes=(
                        "Imported from the filename-based test matrix; "
                        f"risk hint: {row.get('Indice_risque_nom', '').strip()}"
                    ),
                )
            )
            count += 1
    return count
