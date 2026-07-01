"""NoisyLinear/NoisyConv2d compose the original module: at sigma=0 they reproduce it exactly; at
sigma>0 they perturb the output; they preserve dtype/device and keep gradients flowing to the base
parameters (composition, not weight-copying)."""
import torch
import torch.nn as nn


def _noisy_linear(base, sigma):
    from analog_ready.instrument.replace import NoisyLinear
    from analog_ready.noise import GaussianNoise

    return NoisyLinear(base, GaussianNoise(sigma=sigma))


def test_zero_noise_matches_base():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    x = torch.randn(4, 16)
    noisy = _noisy_linear(base, sigma=0.0)
    assert torch.allclose(noisy(x), base(x), atol=1e-6)
    assert noisy(x).shape == (4, 8)


def test_positive_noise_diverges():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    x = torch.randn(4, 16)
    noisy = _noisy_linear(base, sigma=0.5)
    assert not torch.allclose(noisy(x), base(x), atol=1e-4)


def test_preserves_dtype_and_grad_flows_to_base():
    base = nn.Linear(16, 8)
    noisy = _noisy_linear(base, sigma=0.1)
    x = torch.randn(4, 16, dtype=torch.float32)
    out = noisy(x)
    assert out.dtype == torch.float32
    out.sum().backward()
    # gradient must reach the ORIGINAL base parameters (composition, not a detached copy)
    assert base.weight.grad is not None
    assert base.weight.grad.abs().sum() > 0
