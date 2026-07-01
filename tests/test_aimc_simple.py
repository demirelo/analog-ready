"""The pure-PyTorch AIMC fallback: uniform weight quantization (coarsens distinct values),
additive Gaussian noise (perturbs), and a power-law conductance drift (identity at t=0)."""
import torch


def _backend():
    from analog_ready.core.backend import get

    b = get("aimc_simple")
    return b() if isinstance(b, type) else b


def test_quantize_reduces_distinct_values():
    w = torch.randn(256)
    q = _backend().quantize_weights(w, bits=3)
    assert q.unique().numel() <= 2 ** 3
    assert q.shape == w.shape


def test_apply_noise_perturbs():
    torch.manual_seed(0)
    x = torch.randn(128)
    y = _backend().apply_noise(x, sigma=0.1)
    assert not torch.allclose(x, y, atol=1e-4)


def test_drift_is_identity_at_t0():
    w = torch.randn(64)
    drifted = _backend().drift(w, t=0.0)
    assert torch.allclose(w, drifted, atol=1e-6)
