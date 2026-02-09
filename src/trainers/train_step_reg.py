from __future__ import annotations

import torch

from src.losses import reg_loop_estimator, reg_sym_estimator

from .common import compute_dsm_for_score


def _regularizer_batch(
    x: torch.Tensor,
    sigma: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor | None]:
    """Select batch slice used by optional M1 regularizers."""
    if not bool(cfg["loss"].get("reg_low_noise_only", True)):
        return x, sigma, 1.0, None

    sigma0 = float(cfg["loss"].get("sigma0", cfg["loss"].get("reg_low_noise_threshold", 0.25)))
    low_mask = sigma <= sigma0
    low_factor = float(low_mask.float().mean().item())
    if low_factor <= 0.0:
        return x[:0], sigma[:0], 0.0, low_mask
    return x[low_mask], sigma[low_mask], low_factor, low_mask


def _take_subset(
    x: torch.Tensor,
    sigma: torch.Tensor,
    subset_size: int | None,
    score: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Optionally sample a random batch subset for expensive regularizers."""
    if subset_size is None:
        return x, sigma, score
    n = int(subset_size)
    if n <= 0 or x.shape[0] <= n:
        return x, sigma, score
    idx = torch.randperm(x.shape[0], device=x.device)[:n]
    score_sub = None if score is None else score[idx]
    return x[idx], sigma[idx], score_sub


def _cap_subset(subset_size: int | None, cap: int | None) -> int | None:
    """Apply optional upper cap to subset size."""
    if cap is None:
        return subset_size
    cap_i = int(cap)
    if cap_i <= 0:
        return subset_size
    if subset_size is None:
        return cap_i
    n = int(subset_size)
    if n <= 0:
        return subset_size
    return min(n, cap_i)


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
    loss_cfg = cfg["loss"]
    sigma_min = float(loss_cfg["sigma_min"])
    sigma_max = float(loss_cfg["sigma_max"])
    weight_mode = str(loss_cfg.get("weight_mode", "sigma2"))
    objective = str(loss_cfg.get("objective", "dsm_score"))
    sigma_sampling = str(loss_cfg.get("sigma_sampling", "log_uniform"))
    edm_p_mean = float(loss_cfg.get("edm_p_mean", -1.2))
    edm_p_std = float(loss_cfg.get("edm_p_std", 1.2))
    sigma_sample_clamp = bool(loss_cfg.get("sigma_sample_clamp", True))
    sigma_data = float(loss_cfg.get("sigma_data", cfg.get("model", {}).get("preconditioning", {}).get("sigma_data", 0.5)))

    # Regularizer hyperparameters.
    lam = float(loss_cfg.get("lambda_sym", 0.0))
    mu = float(loss_cfg.get("mu_loop", 0.0))
    reg_k = int(loss_cfg.get("reg_k", 1))
    reg_method = str(loss_cfg.get("reg_sym_method", "auto_fast"))
    reg_variant = str(loss_cfg.get("reg_sym_variant", "skew_fro"))
    reg_probe_dist = str(loss_cfg.get("reg_sym_probe_dist", "gaussian"))
    reg_scale_sigma2 = bool(loss_cfg.get("reg_sym_scale_sigma2", False))
    eps_fd = float(loss_cfg.get("reg_sym_eps_fd", 1.0e-3))
    delta = float(loss_cfg.get("loop_delta", 0.01))
    sparse_ratio = float(loss_cfg.get("loop_sparse_ratio", 1.0))

    dsm, cache = compute_dsm_for_score(
        score_fn=model,
        x0=x0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        weight_mode=weight_mode,
        objective=objective,
        sigma_sampling=sigma_sampling,
        edm_p_mean=edm_p_mean,
        edm_p_std=edm_p_std,
        sigma_sample_clamp=sigma_sample_clamp,
        sigma_data=sigma_data,
        cache_score=mu > 0.0,
    )

    x = cache["x"]
    sigma = cache["sigma"]
    score = cache.get("score")

    x_reg, sigma_reg, low_noise_factor, low_mask = _regularizer_batch(x, sigma, cfg)
    score_reg = None
    if score is not None:
        score_reg = score if low_mask is None else score[low_mask]

    # Initialize regularizer terms to zero for skipped steps.
    sym = torch.zeros((), device=x0.device)
    loop = torch.zeros((), device=x0.device)
    lam_eff = lam
    mu_eff = mu

    if _should_apply_regularizer(step=step, cfg=cfg, sigma=sigma) and low_noise_factor > 0.0:
        if bool(loss_cfg.get("reg_low_noise_only", True)):
            # Keep previous lambda scaling behavior under low-noise gating.
            lam_eff = lam * low_noise_factor

        if lam > 0.0 and x_reg.shape[0] > 0:
            sym_subset = _cap_subset(loss_cfg.get("sym_subset", loss_cfg.get("reg_subset")), loss_cfg.get("sym_subset_cap", 16))
            x_sym, sigma_sym, _ = _take_subset(x_reg, sigma_reg, sym_subset)
            sym_values = reg_sym_estimator(
                model,
                x=x_sym,
                sigma=sigma_sym,
                K=reg_k,
                method=reg_method,
                variant=reg_variant,
                probe_dist=reg_probe_dist,
                eps_fd=eps_fd,
                return_per_sample=reg_scale_sigma2,
            )
            if reg_scale_sigma2:
                sym = (sym_values * sigma_sym.square()).mean()
            else:
                sym = sym_values
        if mu > 0.0 and x_reg.shape[0] > 0:
            loop_subset = _cap_subset(
                loss_cfg.get("loop_subset", loss_cfg.get("reg_subset")),
                loss_cfg.get("loop_subset_cap", 16),
            )
            x_loop, sigma_loop, score_loop = _take_subset(x_reg, sigma_reg, loop_subset, score=score_reg)
            loop = reg_loop_estimator(
                model,
                x=x_loop,
                sigma=sigma_loop,
                delta=delta,
                sparse_ratio=sparse_ratio,
                base_score=score_loop,
            )

    total = dsm + lam_eff * sym + mu_eff * loop
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
