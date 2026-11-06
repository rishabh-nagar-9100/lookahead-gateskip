"""Cross-token EMA memory (ARCHITECTURE §2).

    μ_t = α · x_t + (1 − α) · μ_(t−1),   μ_0 = 0 (or a carried-over state)

where x_t = RMSNorm(h_t) at one layer. Two entry points that must agree (test T-EMA):

- :func:`ema_scan`: all positions of a sequence at once (prefill, training). It is a chunked
  causal scan: inside a chunk of C positions it uses the closed form
  μ_t = Σ_(s≤t) α(1−α)^(t−s) x_s + (1−α)^(t−t0+1) μ_(t0−1) as a C×C lower-triangular matmul,
  and it carries the state between chunks. Decay powers never exceed C, so nothing
  underflows for long sequences, and cost is O(T·C·H) instead of O(T²·H).
- :func:`ema_step`: one recurrent update per decoded token.

α may be a float or a tensor (scalar or per-channel, e.g. a learnable α later) in (0, 1].
"""

from __future__ import annotations

import torch


def _decay_matrix(alpha: torch.Tensor, n: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """W[t, s] = α(1−α)^(t−s) for s ≤ t (else 0), and carry[t] = (1−α)^(t+1), both for scalar α."""
    t = torch.arange(n, device=device, dtype=dtype)
    diff = t[:, None] - t[None, :]
    one_minus = (1 - alpha).to(dtype)
    # torch.pow handles 1−α = 0 (α = 1) correctly: 0^0 = 1 on the diagonal, 0 elsewhere.
    pw = torch.pow(one_minus, diff.clamp(min=0))
    w = torch.where(diff >= 0, alpha.to(dtype) * pw, torch.zeros((), device=device, dtype=dtype))
    carry = torch.pow(one_minus, t + 1)
    return w, carry


def ema_scan(
    x: torch.Tensor,
    alpha: float | torch.Tensor,
    mu0: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """EMA over the sequence dimension (dim −2) of ``x`` with shape (..., T, H).

    ``mu0`` (shape (..., H)) is the state before the first position; zeros if None.
    Returns μ with the same shape as ``x``; μ[..., t, :] includes x[..., t, :].
    """
    alpha_t = torch.as_tensor(alpha, device=x.device, dtype=x.dtype)
    if alpha_t.numel() != 1:
        return _ema_scan_per_channel(x, alpha_t, mu0)
    if not 0.0 < float(alpha_t) <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {float(alpha_t)}")
    T = x.shape[-2]
    state = torch.zeros_like(x[..., 0, :]) if mu0 is None else mu0.to(x.dtype)
    outs = []
    for start in range(0, T, chunk_size):
        xc = x[..., start : start + chunk_size, :]
        n = xc.shape[-2]
        w, carry = _decay_matrix(alpha_t, n, x.device, x.dtype)
        mu = torch.matmul(w, xc) + carry[:, None] * state[..., None, :]
        outs.append(mu)
        state = mu[..., -1, :]
    return torch.cat(outs, dim=-2)


def _ema_scan_per_channel(x: torch.Tensor, alpha: torch.Tensor, mu0: torch.Tensor | None) -> torch.Tensor:
    """Per-channel α (shape (H,)): plain recurrence. Only used for the optional learnable-α ablation."""
    state = torch.zeros_like(x[..., 0, :]) if mu0 is None else mu0.to(x.dtype)
    outs = []
    for t in range(x.shape[-2]):
        state = ema_step(x[..., t, :], state, alpha)
        outs.append(state)
    return torch.stack(outs, dim=-2)


def ema_step(x_t: torch.Tensor, mu_prev: torch.Tensor | None, alpha: float | torch.Tensor) -> torch.Tensor:
    """One decode step: μ_t = α·x_t + (1−α)·μ_(t−1). ``mu_prev`` None means μ_0 = 0."""
    if mu_prev is None:
        return alpha * x_t
    return alpha * x_t + (1 - alpha) * mu_prev
