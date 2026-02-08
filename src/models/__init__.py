from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .potential_mlp_toy import PotentialMLPToy
from .potential_unet import PotentialUNet
from .score_mlp_toy import ScoreMLPToy
from .score_unet import ScoreUNet


def build_model(cfg: dict) -> nn.Module:
    dataset_name = cfg["dataset"]["name"]
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]

    if dataset_name == "toy":
        dim = int(cfg["dataset"]["toy"].get("dim", 2))
        if model_type == "mlp":
            return ScoreMLPToy(
                dim=dim,
                hidden_dim=int(model_cfg.get("hidden_dim", 256)),
                depth=int(model_cfg.get("depth", 3)),
                sigma_embed_dim=int(model_cfg.get("sigma_embed_dim", 64)),
            )
        if model_type == "mlp_potential":
            return PotentialMLPToy(
                dim=dim,
                hidden_dim=int(model_cfg.get("hidden_dim", 256)),
                depth=int(model_cfg.get("depth", 3)),
                sigma_embed_dim=int(model_cfg.get("sigma_embed_dim", 64)),
            )
    else:
        channels = int(cfg["dataset"]["channels"])
        image_size = int(cfg["dataset"]["image_size"])
        kwargs = dict(
            channels=channels,
            image_size=image_size,
            base_channels=int(model_cfg.get("base_channels", 64)),
            channel_mults=list(model_cfg.get("channel_mults", [1, 2, 2])),
            num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
            attn_resolutions=list(model_cfg.get("attn_resolutions", [16])),
            sigma_embed_dim=int(model_cfg.get("sigma_embed_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        if model_type == "unet":
            return ScoreUNet(**kwargs)
        if model_type == "potential_unet":
            return PotentialUNet(**kwargs)

    raise ValueError(f"unsupported model type: {dataset_name=} {model_type=}")


def score_fn_from_model(
    model: nn.Module,
    variant: str,
    *,
    create_graph: bool = True,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if variant != "struct":
        return model

    def _score_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_req = x.requires_grad_(True)
        phi = model(x_req, sigma)
        if phi.ndim == 1:
            phi = phi[:, None]
        grad = torch.autograd.grad(phi.sum(), x_req, create_graph=create_graph)[0]
        return grad

    return _score_fn
