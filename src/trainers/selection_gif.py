from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from src.data.toy import toy_mixture_parameters
from src.eval.sampling import sampler_by_name
from src.models import score_fn_from_model


def _subsample_indices(length: int, max_frames: int) -> list[int]:
    if length <= 0:
        return []
    if max_frames <= 0 or length <= max_frames:
        return list(range(length))
    return sorted({int(v) for v in np.linspace(0, length - 1, num=max_frames)})


def _as_pil_from_figure(fig: Figure) -> Image.Image:
    if Image is None:
        raise RuntimeError("Pillow is required to write GIF reports.")
    canvas = FigureCanvas(fig)
    canvas.draw()
    arr = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
    return Image.fromarray(arr.copy())


def _render_toy_frames(
    trajectory: list[torch.Tensor],
    cfg: dict[str, Any],
    sampler_name: str,
    max_frames: int,
) -> list[Image.Image]:
    if Image is None:
        raise RuntimeError("Pillow is required to write GIF reports.")
    idxs = _subsample_indices(len(trajectory), max_frames=max_frames)
    if not idxs:
        return []

    centers_t, stds_t = toy_mixture_parameters(cfg, dtype=torch.float32, device=torch.device("cpu"))
    centers = centers_t.cpu().numpy()
    stds = stds_t.cpu().numpy()
    states = [trajectory[i].detach().cpu().float().numpy() for i in idxs]

    max_abs = 0.0
    for st in states:
        if st.size > 0:
            max_abs = max(max_abs, float(np.max(np.abs(st[:, :2]))))
    if centers.size > 0:
        max_abs = max(max_abs, float(np.max(np.abs(centers[:, :2]))))
    std_pad = float(np.max(stds)) if stds.size > 0 else 0.0
    span = max(1.0, max_abs + 3.0 * std_pad + 0.1)

    frames: list[Image.Image] = []
    total_steps = len(trajectory)
    for i, st in zip(idxs, states):
        fig = Figure(figsize=(4.0, 4.0), dpi=120)
        ax = fig.add_subplot(111)
        ax.scatter(st[:, 0], st[:, 1], s=6, alpha=0.35, color="#1f77b4", linewidths=0.0)
        if centers.size > 0:
            ax.scatter(centers[:, 0], centers[:, 1], s=24, marker="x", color="#d62728", linewidths=1.0)
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_title(f"{sampler_name.upper()} | step {i + 1}/{total_steps}", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        fig.tight_layout(pad=0.3)
        frames.append(_as_pil_from_figure(fig))
    return frames


def _make_image_grid_uint8(batch: torch.Tensor, max_images: int = 16) -> np.ndarray:
    x = batch.detach().cpu().float()
    x = x[: max(1, int(max_images))]

    if x.ndim != 4:
        raise ValueError(f"image batch must be BCHW, got shape={tuple(x.shape)}")
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] > 3:
        x = x[:, :3]

    lo = torch.quantile(x, 0.01)
    hi = torch.quantile(x, 0.99)
    if not torch.isfinite(lo) or not torch.isfinite(hi) or float((hi - lo).abs()) < 1.0e-8:
        lo = x.min()
        hi = x.max()
    x = (x - lo) / (hi - lo + 1.0e-6)
    x = x.clamp(0.0, 1.0)

    b, c, h, w = x.shape
    ncol = int(math.ceil(math.sqrt(b)))
    nrow = int(math.ceil(b / ncol))
    canvas = torch.zeros((3, nrow * h, ncol * w), dtype=x.dtype)
    for i in range(b):
        r = i // ncol
        col = i % ncol
        canvas[:, r * h : (r + 1) * h, col * w : (col + 1) * w] = x[i]

    grid = (canvas.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return grid


def _render_image_frames(
    trajectory: list[torch.Tensor],
    max_frames: int,
    max_images: int,
) -> list[Image.Image]:
    if Image is None:
        raise RuntimeError("Pillow is required to write GIF reports.")
    idxs = _subsample_indices(len(trajectory), max_frames=max_frames)
    frames: list[Image.Image] = []
    for i in idxs:
        grid = _make_image_grid_uint8(trajectory[i], max_images=max_images)
        frames.append(Image.fromarray(grid))
    return frames


def create_best_ckpt_sampling_gif(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    model_id: str,
    device: torch.device,
    out_path: str | Path,
    sampler_name: str,
    nfe: int,
    num_samples: int,
    max_frames: int = 40,
    fps: int = 8,
    max_images: int = 16,
) -> dict[str, Any] | None:
    """Create a trajectory GIF from best-checkpoint sampling.

    Returns metadata dictionary when GIF is written, otherwise ``None``.
    """
    dataset_name = str(cfg.get("dataset", {}).get("name", "")).lower()
    if Image is None:
        raise RuntimeError("Pillow is required to write GIF reports.")
    is_toy = dataset_name.startswith("toy")
    score_requires_grad = str(model_id).upper() in {"M2", "M4"}

    if is_toy:
        dim = int(cfg.get("dataset", {}).get("toy", {}).get("dim", 2))
        shape = (max(1, int(num_samples)), dim)
    else:
        channels = int(cfg.get("dataset", {}).get("channels", 3))
        image_size = int(cfg.get("dataset", {}).get("image_size", 32))
        shape = (max(1, int(num_samples)), channels, image_size, image_size)

    score_fn = score_fn_from_model(model, str(model_id).upper(), create_graph=False)
    sampler = sampler_by_name(sampler_name)

    _, stats = sampler(
        score_fn=score_fn,
        shape=shape,
        sigma_min=float(cfg["loss"]["sigma_min"]),
        sigma_max=float(cfg["loss"]["sigma_max"]),
        nfe=max(1, int(nfe)),
        device=device,
        score_requires_grad=score_requires_grad,
        return_trajectory=True,
    )

    trajectory = stats.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) < 2:
        return None

    if is_toy and trajectory[0].ndim == 2 and int(trajectory[0].shape[1]) >= 2:
        frames = _render_toy_frames(
            trajectory=trajectory,
            cfg=cfg,
            sampler_name=sampler_name,
            max_frames=max_frames,
        )
    elif trajectory[0].ndim == 4:
        frames = _render_image_frames(
            trajectory=trajectory,
            max_frames=max_frames,
            max_images=max_images,
        )
    else:
        return None

    if len(frames) < 2:
        return None

    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(1, int(fps))))
    frames[0].save(
        dst,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return {
        "path": str(dst),
        "frames": int(len(frames)),
        "fps": int(fps),
        "nfe": int(nfe),
        "num_samples": int(num_samples),
        "sampler": str(sampler_name),
    }
