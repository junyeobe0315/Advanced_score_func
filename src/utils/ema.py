from __future__ import annotations

from copy import deepcopy

import torch


class EMA:
    """Exponential moving average wrapper for model parameters.

    Notes:
        EMA stabilizes evaluation by smoothing parameter updates over training
        steps. This is standard in diffusion/score-model training pipelines.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        """Create shadow model and initialize EMA state.

        Args:
            model: Source model to track.
            decay: EMA decay factor in [0,1).
        """
        self.decay = float(decay)
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update shadow parameters from current model weights.

        Args:
            model: Live training model whose parameters are averaged in.

        How it works:
            Applies ``shadow = decay * shadow + (1-decay) * param`` for each
            parameter and copies buffers directly.
        """
        shadow_state = dict(self.shadow.named_parameters())
        for name, param in model.named_parameters():
            if name not in shadow_state:
                continue
            shadow_state[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

        shadow_buffers = dict(self.shadow.named_buffers())
        for name, buf in model.named_buffers():
            if name in shadow_buffers:
                shadow_buffers[name].copy_(buf.detach())

    def to(self, device: torch.device | str) -> None:
        """Move shadow model to target device."""
        self.shadow.to(device)

    def state_dict(self) -> dict:
        """Export EMA state dictionary for checkpointing."""
        return {"decay": self.decay, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore EMA state dictionary from checkpoint payload."""
        self.decay = float(state["decay"])
        self.shadow.load_state_dict(state["shadow"])
