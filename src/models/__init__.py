from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch
import torch.nn as nn

from src.utils.config import resolve_model_id

from .hybrid_wrapper import HybridWrapper
from .potential_net import score_from_potential
from .potential_mlp_toy import PotentialMLPToy
from .potential_unet import PotentialUNet
from .score_mlp_toy import ScoreMLPToy
from .score_unet import ScoreUNet


def _is_toy_dataset(dataset_name: str) -> bool:
    """Check whether dataset name corresponds to toy distribution family."""
    return dataset_name.lower().startswith("toy")


def _default_score_type(dataset_name: str) -> str:
    """Return default score-network type for dataset family."""
    return "mlp" if _is_toy_dataset(dataset_name) else "unet"


def _default_potential_type(dataset_name: str) -> str:
    """Return default potential-network type for dataset family."""
    return "mlp_potential" if _is_toy_dataset(dataset_name) else "potential_unet"


def _build_score_network(cfg: dict, model_cfg: dict) -> nn.Module:
    """Build unconstrained score network branch.

    Args:
        cfg: Full experiment config dictionary.
        model_cfg: Model subsection used for constructor arguments.

    Returns:
        Score network module.
    """
    dataset_name = cfg["dataset"]["name"]

    if _is_toy_dataset(dataset_name):
        dim = int(cfg["dataset"]["toy"].get("dim", 2))
        return ScoreMLPToy(
            dim=dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            depth=int(model_cfg.get("depth", 3)),
            sigma_embed_dim=int(model_cfg.get("sigma_embed_dim", 64)),
        )

    channels = int(cfg["dataset"]["channels"])
    image_size = int(cfg["dataset"]["image_size"])
    return ScoreUNet(
        channels=channels,
        image_size=image_size,
        base_channels=int(model_cfg.get("base_channels", 64)),
        channel_mults=list(model_cfg.get("channel_mults", [1, 2, 2])),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
        attn_resolutions=list(model_cfg.get("attn_resolutions", [16])),
        sigma_embed_dim=int(model_cfg.get("sigma_embed_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def _build_potential_network(cfg: dict, model_cfg: dict) -> nn.Module:
    """Build potential-network branch.

    Args:
        cfg: Full experiment config dictionary.
        model_cfg: Model subsection used for constructor arguments.

    Returns:
        Potential network module.
    """
    dataset_name = cfg["dataset"]["name"]

    if _is_toy_dataset(dataset_name):
        dim = int(cfg["dataset"]["toy"].get("dim", 2))
        return PotentialMLPToy(
            dim=dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            depth=int(model_cfg.get("depth", 3)),
            sigma_embed_dim=int(model_cfg.get("sigma_embed_dim", 64)),
        )

    channels = int(cfg["dataset"]["channels"])
    image_size = int(cfg["dataset"]["image_size"])
    return PotentialUNet(
        channels=channels,
        image_size=image_size,
        base_channels=int(model_cfg.get("base_channels", 64)),
        channel_mults=list(model_cfg.get("channel_mults", [1, 2, 2])),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
        attn_resolutions=list(model_cfg.get("attn_resolutions", [16])),
        sigma_embed_dim=int(model_cfg.get("sigma_embed_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def build_model(cfg: dict) -> nn.Module:
    """Instantiate model object for canonical 5-way setup (M0..M4).

    Args:
        cfg: Full experiment config dictionary.

    Returns:
        Torch ``nn.Module`` matching resolved model id.

    How it works:
        Resolves ``model_id`` and builds:
        - M0/M1/M3: unconstrained score net
        - M2: fully conservative potential net
        - M4: hybrid wrapper (high-noise score + low-noise potential)
    """
    dataset_name = cfg["dataset"]["name"]
    model_id = resolve_model_id(cfg)
    model_cfg = deepcopy(cfg["model"])

    if model_id in {"M0", "M1", "M3"}:
        model_cfg["type"] = model_cfg.get("type", _default_score_type(dataset_name))
        return _build_score_network(cfg, model_cfg)

    if model_id == "M2":
        model_cfg["type"] = model_cfg.get("type", _default_potential_type(dataset_name))
        return _build_potential_network(cfg, model_cfg)

    if model_id == "M4":
        hybrid_cfg = model_cfg.get("hybrid", {})

        high_cfg = deepcopy(model_cfg)
        high_cfg["type"] = hybrid_cfg.get("high_type", _default_score_type(dataset_name))
        for key in ["low_type", "low_base_channels", "low_hidden_dim", "low_depth"]:
            high_cfg.pop(key, None)

        low_cfg = deepcopy(model_cfg)
        low_cfg["type"] = hybrid_cfg.get("low_type", _default_potential_type(dataset_name))
        if "low_base_channels" in hybrid_cfg:
            low_cfg["base_channels"] = int(hybrid_cfg["low_base_channels"])
        if "low_hidden_dim" in hybrid_cfg:
            low_cfg["hidden_dim"] = int(hybrid_cfg["low_hidden_dim"])
        if "low_depth" in hybrid_cfg:
            low_cfg["depth"] = int(hybrid_cfg["low_depth"])

        sigma_c = float(cfg["loss"].get("sigma_c", cfg["loss"].get("sigma0", 0.25)))
        return HybridWrapper(
            score_high=_build_score_network(cfg, high_cfg),
            potential_low=_build_potential_network(cfg, low_cfg),
            sigma_c=sigma_c,
        )

    raise ValueError(f"unsupported model_id: {model_id}")


def score_fn_from_model(
    model: nn.Module,
    variant_or_model_id: str | None = None,
    *,
    variant: str | None = None,
    create_graph: bool = True,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Expose a score-function callable for all model variants.

    Args:
        model: Base model instance.
        variant_or_model_id: Legacy variant or canonical model id.
        variant: Backward-compatible alias for ``variant_or_model_id``.
        create_graph: Whether gradient computation should keep graph for higher
            order derivatives.

    Returns:
        Callable ``score_fn(x, sigma) -> score``.

    How it works:
        Baseline/reg variants already output score directly. Struct variant
        outputs scalar potential ``phi`` and this wrapper returns ``grad_x phi``.
    """
    mapping = {"baseline": "M0", "reg": "M1", "struct": "M2"}
    token_source = variant_or_model_id if variant_or_model_id is not None else variant
    if token_source is None:
        raise ValueError("variant_or_model_id or variant must be provided")
    token = str(token_source)
    model_id = mapping.get(token.lower(), token.upper())

    if model_id in {"M0", "M1", "M3"}:
        return model

    def _score_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute score for M2/M4 by differentiating potential branch."""
        if model_id == "M2":
            return score_from_potential(model, x, sigma, create_graph=create_graph)
        if model_id == "M4":
            assert isinstance(model, HybridWrapper)
            return model.score(x, sigma, create_graph=create_graph)
        raise ValueError(f"unsupported model id for score wrapper: {model_id}")

    return _score_fn
