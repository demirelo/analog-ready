"""W3 acceptance oracle — recovery via quant-scale calibration. The gain must come from better
quantisation, not from reducing noise (it must persist at sigma=0)."""
import pytest
import torch
import torch.nn as nn

from analog_ready.quant import QuantSpec, fake_quant
from analog_ready.recovery import calibrate_quant_scale, recover


def _outlier_linear() -> nn.Linear:
    torch.manual_seed(0)
    lin = nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        lin.weight.copy_(0.5 * torch.randn(4, 8))
        lin.weight[0, 0] = 8.0          # one large outlier that wastes naive quant levels
    return lin


def _dead_outlier_inputs() -> torch.Tensor:
    torch.manual_seed(2)
    x = torch.randn(16, 8)
    x[:, 0] = 0.0                       # column 0 is dead -> clipping the outlier costs nothing
    return x


def test_calibration_reduces_output_error_and_clips_outlier():
    lin = _outlier_linear()
    x = _dead_outlier_inputs()
    bits = 3
    clip = calibrate_quant_scale(lin, x, bits)
    w = lin.weight.detach()
    clean = x @ w.T
    naive = x @ fake_quant(w, QuantSpec(bits=bits)).T
    calib = x @ fake_quant(w.clamp(-clip, clip), QuantSpec(bits=bits)).T
    assert ((calib - clean) ** 2).mean() < ((naive - clean) ** 2).mean()
    assert clip < float(w.abs().max())          # it actually tightens the range


def test_recovery_improves_fidelity_under_same_noise():
    lin, x = _outlier_linear(), _dead_outlier_inputs()
    out = recover(lin, x, bits=3, sigma=0.02, seed=0)
    assert out["recovered_fidelity"] > out["degraded_fidelity"]
    assert out["sigma"] == 0.02                  # anti-cheat: noise level was NOT reduced


def test_recovery_gain_persists_with_zero_noise():
    lin, x = _outlier_linear(), _dead_outlier_inputs()
    out = recover(lin, x, bits=3, sigma=0.0, seed=0)
    assert out["recovered_fidelity"] > out["degraded_fidelity"]   # gain is purely from calibration
