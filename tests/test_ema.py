"""T-EMA: the prefill scan equals the step-by-step decode recurrence (ARCHITECTURE §2, §8)."""

import pytest
import torch

from lagate.routing.ema import ema_scan, ema_step


def reference_recurrence(x, alpha, mu0=None):
    mu = torch.zeros_like(x[..., 0, :]) if mu0 is None else mu0
    out = []
    for t in range(x.shape[-2]):
        mu = alpha * x[..., t, :] + (1 - alpha) * mu
        out.append(mu)
    return torch.stack(out, dim=-2)


@pytest.mark.parametrize("alpha", [0.5, 0.7, 0.9, 1.0])
@pytest.mark.parametrize("T, chunk", [(1, 64), (7, 64), (64, 64), (200, 64), (37, 5)])
def test_t_ema_scan_equals_recurrence(alpha, T, chunk):
    torch.manual_seed(0)
    x = torch.randn(2, T, 16, dtype=torch.float64)
    mu0 = torch.randn(2, 16, dtype=torch.float64)
    for init in (None, mu0):
        ref = reference_recurrence(x, alpha, init)
        got = ema_scan(x, alpha, mu0=init, chunk_size=chunk)
        torch.testing.assert_close(got, ref, rtol=0, atol=1e-12)


def test_t_ema_prefill_then_decode_matches_full_scan():
    """Prefill a prompt with the scan, then decode with ema_step: same as one scan over everything."""
    torch.manual_seed(1)
    x = torch.randn(1, 50, 32)
    full = ema_scan(x, 0.7)
    mu = ema_scan(x[:, :30], 0.7)[:, -1]
    for t in range(30, 50):
        mu = ema_step(x[:, t], mu, 0.7)
        torch.testing.assert_close(mu, full[:, t], rtol=0, atol=1e-6)


def test_t_ema_long_sequence_is_finite_fp32():
    x = torch.randn(1, 4096, 8)
    mu = ema_scan(x, 0.5)
    assert torch.isfinite(mu).all()
    torch.testing.assert_close(mu, reference_recurrence(x, 0.5), rtol=0, atol=1e-5)


def test_ema_zero_init_and_gradients():
    x = torch.randn(1, 3, 4, requires_grad=True)
    mu = ema_scan(x, 0.7)
    torch.testing.assert_close(mu[:, 0], 0.7 * x[:, 0])  # μ_0 = 0
    mu.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_ema_per_channel_alpha():
    x = torch.randn(1, 9, 4, dtype=torch.float64)
    alpha = torch.tensor([0.5, 0.7, 0.9, 1.0], dtype=torch.float64)
    torch.testing.assert_close(ema_scan(x, alpha), reference_recurrence(x, alpha), rtol=0, atol=1e-12)


def test_ema_rejects_bad_alpha():
    with pytest.raises(ValueError):
        ema_scan(torch.randn(1, 3, 4), 0.0)
