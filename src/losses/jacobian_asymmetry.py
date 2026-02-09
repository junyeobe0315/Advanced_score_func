from __future__ import annotations

from typing import Callable

import torch


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _flatten_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Flatten non-batch dimensions for per-sample norm evaluation."""
    return x.reshape(x.shape[0], -1)


def _jvp_forward_mode(
    score_fn: ScoreFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute Jacobian-vector product ``Jv`` via autograd JVP."""
    _, jvp = torch.autograd.functional.jvp(
        lambda inp: score_fn(inp, sigma),
        (x,),
        (v,),
        create_graph=True,
        strict=False,
    )
    return jvp


def _sample_probe(x: torch.Tensor, dist: str) -> torch.Tensor:
    """Sample Hutchinson probe vector from requested distribution."""
    token = str(dist).lower()
    if token in {"gaussian", "normal"}:
        return torch.randn_like(x)
    if token in {"rademacher", "rad"}:
        v = torch.randint(0, 2, x.shape, device=x.device, dtype=torch.int64).to(dtype=x.dtype)
        return 2.0 * v - 1.0
    raise ValueError(f"unknown probe distribution: {dist}")


def _jvp_finite_difference(
    score_fn: ScoreFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    v: torch.Tensor,
    eps_fd: float,
    s_base: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate Jacobian-vector product ``Jv`` using finite difference.

    Args:
        score_fn: Score function ``s(x, sigma)``.
        x: Input batch with gradient enabled.
        sigma: Per-sample sigma tensor.
        v: Probe directions matching ``x`` shape.
        eps_fd: Finite-difference epsilon.

    Returns:
        Approximate ``Jv`` tensor.
    """
    s_plus = score_fn(x + eps_fd * v, sigma)
    base = s_base if s_base is not None else score_fn(x, sigma)
    return (s_plus - base) / eps_fd


def _resolve_estimator_method(method: str, x: torch.Tensor) -> str:
    """Resolve runtime Jacobian estimator method from policy token."""
    method_now = str(method)
    if method_now == "auto_fast":
        # For very small and large tensors, finite differences typically
        # outperform forward-mode JVP in wall-clock time.
        flat_dim = int(x[0].numel())
        if flat_dim <= 32 or flat_dim >= 128:
            return "finite_diff"
        return "jvp_vjp"
    return method_now


def jacobian_asymmetry_estimator(
    score_fn: ScoreFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    num_probes: int = 1,
    method: str = "jvp_vjp",
    variant: str = "skew_fro",
    probe_dist: str = "gaussian",
    eps_fd: float = 1.0e-3,
    return_per_sample: bool = False,
) -> torch.Tensor:
    """Estimate ``||J - J^T||_F^2`` with Hutchinson probing.

    Args:
        score_fn: Score function ``s(x, sigma)``.
        x: Batched noisy inputs.
        sigma: Per-sample sigma values ``[B]``.
        num_probes: Number of Gaussian probe vectors.
        method: ``"jvp_vjp"``, ``"finite_diff"``, ``"auto"``, or ``"auto_fast"``.
        variant: Energy definition token.
            - ``"skew_fro"``: ``E||Jv - J^Tv||^2`` (default)
            - ``"qcsbm_trace"``: ``E[||J^Tv||^2 - <Jv, J^Tv>]``
              matching QCSBM code path.
        probe_dist: Probe distribution, ``"gaussian"`` or ``"rademacher"``.
        eps_fd: Finite-difference epsilon for fallback mode.
        return_per_sample: Return per-sample values when True.

    Returns:
        Scalar tensor estimating Jacobian asymmetry energy.

    How it works:
        Let ``A = J - J^T``. Hutchinson identity gives
        ``||A||_F^2 = E_v ||Av||^2``. For each probe ``v`` we compute:
        - ``J^T v`` via VJP: ``grad_x <s(x), v>``
        - ``J v`` via JVP or finite difference fallback
        and average ``||Jv - J^Tv||^2`` over probes.
    """
    probes = max(int(num_probes), 1)
    x_req = x.requires_grad_(True)
    method_resolved = _resolve_estimator_method(method, x_req)

    # Reuse base score for all VJP probes to avoid repeated forward passes.
    s = score_fn(x_req, sigma)
    total = torch.zeros((x_req.shape[0],), device=x_req.device, dtype=x_req.dtype)
    for _ in range(probes):
        # Hutchinson probe (Gaussian or Rademacher).
        v = _sample_probe(x_req, probe_dist)

        jtv = torch.autograd.grad((s * v).sum(), x_req, create_graph=True, retain_graph=True)[0]

        method_now = method_resolved
        if method_now not in {"jvp_vjp", "finite_diff", "auto"}:
            raise ValueError(f"unknown jacobian estimator method: {method_now}")

        if method_now == "finite_diff":
            jv = _jvp_finite_difference(score_fn, x_req, sigma, v, eps_fd=eps_fd, s_base=s)
        else:
            try:
                jv = _jvp_forward_mode(score_fn, x_req, sigma, v)
            except RuntimeError:
                if method_now == "jvp_vjp":
                    raise
                jv = _jvp_finite_difference(score_fn, x_req, sigma, v, eps_fd=eps_fd, s_base=s)

        if variant == "skew_fro":
            diff = jv - jtv
            contrib = (_flatten_per_sample(diff) ** 2).sum(dim=1)
        elif variant in {"qcsbm", "qcsbm_trace"}:
            flat_jtv = _flatten_per_sample(jtv)
            flat_jv = _flatten_per_sample(jv)
            trace_jjt = (flat_jtv**2).sum(dim=1)
            trace_jj = (flat_jtv * flat_jv).sum(dim=1)
            contrib = trace_jjt - trace_jj
        else:
            raise ValueError(f"unknown jacobian asymmetry variant: {variant}")
        total = total + contrib

    per_sample = total / float(probes)
    if return_per_sample:
        return per_sample
    return per_sample.mean()


def low_noise_gate(sigma: torch.Tensor, sigma0: float) -> torch.Tensor:
    """Return a binary low-noise gate mask ``1[sigma <= sigma0]``.

    Args:
        sigma: Per-sample sigma tensor.
        sigma0: Low-noise threshold.

    Returns:
        Float mask tensor with same shape as ``sigma``.
    """
    return (sigma <= float(sigma0)).to(dtype=sigma.dtype)
