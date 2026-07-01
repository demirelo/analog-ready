"""W3 acceptance oracle — degradation sweeps. Deterministic via fixed seeds + mean over draws;
asserts endpoints and trend, never brittle exact values."""
import pytest
import torch
import torch.nn as nn

from analog_ready.sweeps import fidelity, degradation_curve


def _tiny_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    m.eval()
    return m


def _inputs(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(4, 16)


def test_fidelity_identical_is_one():
    y = torch.randn(3, 5)
    assert fidelity(y, y) == pytest.approx(1.0, abs=1e-6)


def test_fidelity_bounded_and_lower_for_noisy():
    y = torch.randn(8, 16)
    n = y + 0.5 * torch.randn_like(y)
    f = fidelity(y, n)
    assert -1.0 - 1e-6 <= f <= 1.0 + 1e-6
    assert f < 1.0


def test_sigma_sweep_endpoints_and_monotone_trend():
    m, x = _tiny_model(), _inputs()
    with torch.no_grad():
        s = float(m(x).std())
    sigmas = [0.0, 0.25 * s, 0.5 * s, 1.0 * s, 2.0 * s]
    curve = degradation_curve(m, x, param="sigma", values=sigmas, draws=16, seed=0)
    assert [p["value"] for p in curve] == sigmas
    fids = [p["fidelity"] for p in curve]
    assert fids[0] == pytest.approx(1.0, abs=1e-6)        # sigma=0 is a no-op
    for a, b in zip(fids, fids[1:]):                      # non-increasing within eps (mean over draws)
        assert b <= a + 0.03
    assert fids[-1] < 0.9                                 # 2x-signal noise clearly degrades cosine


def test_sigma_sweep_deterministic_given_seed():
    vals = [0.0, 0.2, 0.4]
    a = degradation_curve(_tiny_model(), _inputs(), param="sigma", values=vals, draws=8, seed=7)
    b = degradation_curve(_tiny_model(), _inputs(), param="sigma", values=vals, draws=8, seed=7)
    assert [p["fidelity"] for p in a] == [p["fidelity"] for p in b]


def test_weight_bits_sweep_degrades_monotonically():
    m, x = _tiny_model(), _inputs()
    bits = [8, 6, 4, 3, 2]
    curve = degradation_curve(m, x, param="weight_bits", values=bits, draws=1, seed=0)
    fids = [p["fidelity"] for p in curve]
    assert fids[0] == pytest.approx(1.0, abs=0.05)        # 8-bit fake-quant ~ identity
    for a, b in zip(fids, fids[1:]):
        assert b <= a + 0.02
    assert fids[-1] < fids[0] - 0.01                      # 2-bit clearly worse than 8-bit
