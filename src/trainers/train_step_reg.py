from __future__ import annotations

import torch

from src.losses import reg_loop_estimator, reg_sym_estimator

from .common import compute_dsm_for_score


def _should_apply_regularizer(step: int, cfg: dict, sigma: torch.Tensor) -> bool:
    """Decide whether integrability regularizers should run this step.

    Args:
        step: Current global training step.
        cfg: Resolved config dictionary.
        sigma: Batch sigma tensor sampled for current step.

    Returns:
        ``True`` when regularizers should be evaluated, else ``False``.

    How it works:
        Applies frequency gating first, then optional low-noise gating based on
        fraction of samples below configured sigma threshold.
    """
    freq = int(cfg["loss"].get("reg_freq", 1))
    if step % max(freq, 1) != 0:
        return False

    if bool(cfg["loss"].get("reg_low_noise_only", True)):
        thr = float(cfg["loss"].get("sigma0", cfg["loss"].get("reg_low_noise_threshold", 0.25)))
        return bool((sigma <= thr).any().item())
    return True


def train_step_reg(
    model: torch.nn.Module,
    x0: torch.Tensor,
    cfg: dict,
    step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one training step for soft-integrability regularized model.

    Args:
        model: Score model returning direct vector field.
        x0: Clean training batch.
        cfg: Resolved config dictionary.
        step: Current global training step.

    Returns:
        Tuple ``(loss, metrics)`` with total objective and detached scalars.

    How it works:
        Computes DSM loss, optionally evaluates ``R_sym`` and ``R_loop``, then
        forms weighted sum ``DSM + lambda*R_sym + mu*R_loop``.
    """
    sigma_min = float(cfg["loss"]["sigma_min"])
    sigma_max = float(cfg["loss"]["sigma_max"])
    weight_mode = str(cfg["loss"].get("weight_mode", "sigma2"))

    # Regularizer hyperparameters.
    lam = float(cfg["loss"].get("lambda_sym", 0.0))
    mu = float(cfg["loss"].get("mu_loop", 0.0))
    reg_k = int(cfg["loss"].get("reg_k", 1))
    reg_method = str(cfg["loss"].get("reg_sym_method", "jvp_vjp"))
    eps_fd = float(cfg["loss"].get("reg_sym_eps_fd", 1.0e-3))
    delta = float(cfg["loss"].get("loop_delta", 0.01))
    sparse_ratio = float(cfg["loss"].get("loop_sparse_ratio", 1.0))

    dsm, cache = compute_dsm_for_score(
        score_fn=model,
        x0=x0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        weight_mode=weight_mode,
    )

    x = cache["x"]
    sigma = cache["sigma"]

    # Initialize regularizer terms to zero for skipped steps.
    sym = torch.zeros((), device=x0.device)
    loop = torch.zeros((), device=x0.device)
    lam_eff = lam

    if _should_apply_regularizer(step=step, cfg=cfg, sigma=sigma):
        if bool(cfg["loss"].get("reg_low_noise_only", True)):
            sigma0 = float(cfg["loss"].get("sigma0", cfg["loss"].get("reg_low_noise_threshold", 0.25)))
            # Low-noise gate scales regularization by active low-noise ratio.
            lam_eff = lam * float((sigma <= sigma0).float().mean().item())

        if lam > 0.0:
            sym = reg_sym_estimator(model, x=x, sigma=sigma, K=reg_k, method=reg_method, eps_fd=eps_fd)
        if mu > 0.0:
            loop = reg_loop_estimator(model, x=x, sigma=sigma, delta=delta, sparse_ratio=sparse_ratio)

    total = dsm + lam_eff * sym + mu * loop
    metrics = {
        "loss_total": float(total.detach().item()),
        "loss_dsm": float(dsm.detach().item()),
        "loss_sym": float(sym.detach().item()),
        "loss_loop": float(loop.detach().item()),
        "loss_loop_multi": float(loop.detach().item()),
        "loss_cycle": 0.0,
        "loss_match": 0.0,
    }
    return total, metrics
