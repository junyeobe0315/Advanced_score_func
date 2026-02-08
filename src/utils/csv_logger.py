from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    """Minimal append-only CSV logger for scalar metrics."""

    def __init__(self, path: str | Path) -> None:
        """Initialize CSV logger path and header state.

        Args:
            path: Destination CSV file path.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Header is inferred from first logged row.
        self._fieldnames: list[str] = []
        self._initialized = False

    def _ensure_writer(self, row: dict[str, Any]) -> None:
        """Create CSV file and header if this is the first write.

        Args:
            row: First metrics row used to determine field order.
        """
        if self._initialized:
            return
        self._fieldnames = list(row.keys())
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
        self._initialized = True

    def log(self, row: dict[str, Any]) -> None:
        """Append one metrics row to CSV file.

        Args:
            row: Mapping from metric names to scalar values.
        """
        self._ensure_writer(row)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)
