from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] = []
        self._initialized = False

    def _ensure_writer(self, row: dict[str, Any]) -> None:
        if self._initialized:
            return
        self._fieldnames = list(row.keys())
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
        self._initialized = True

    def log(self, row: dict[str, Any]) -> None:
        self._ensure_writer(row)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)
