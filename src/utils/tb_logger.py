from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency at runtime
    SummaryWriter = None


class TBLogger:
    """Thin wrapper over TensorBoard SummaryWriter with graceful fallback."""

    def __init__(self, log_dir: str | Path, enabled: bool = True) -> None:
        """Initialize tensorboard logger.

        Args:
            log_dir: TensorBoard event directory.
            enabled: Requested logging flag.

        How it works:
            Logger auto-disables when tensorboard dependency is unavailable.
        """
        self.enabled = enabled and SummaryWriter is not None
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir)) if self.enabled else None

    def log_scalars(self, metrics: dict[str, Any], step: int) -> None:
        """Write scalar metrics to TensorBoard.

        Args:
            metrics: Mapping from metric names to values.
            step: Global training/eval step index.
        """
        if not self.writer:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        """Close underlying SummaryWriter if active."""
        if self.writer:
            self.writer.close()
