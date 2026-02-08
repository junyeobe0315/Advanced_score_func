from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency at runtime
    SummaryWriter = None


class TBLogger:
    def __init__(self, log_dir: str | Path, enabled: bool = True) -> None:
        self.enabled = enabled and SummaryWriter is not None
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir)) if self.enabled else None

    def log_scalars(self, metrics: dict[str, Any], step: int) -> None:
        if not self.writer:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        if self.writer:
            self.writer.close()
