"""W12 · F1/F2 — fixed-range ADC + saturation (immutable oracle).

W9's ADC auto-ranges to each sample's own peak (best-case). A real ADC has a FIXED full-scale
reference: values beyond it SATURATE, and a single range is shared (per-tensor), not fitted per
sample. `adc_quantize(..., full_scale=V)` models that; `full_scale=None` keeps the W9 auto behaviour
(backward-compatible). `adc_saturation_fraction` reports how much of the signal clips.
"""
import torch

from analog_ready.adc import ADCQuant, adc_quantize, adc_saturation_fraction
from analog_ready.sweeps import fidelity


def test_full_scale_none_is_the_unchanged_auto_behaviour():
    x = torch.randn(4, 32)
    assert torch.equal(adc_quantize(x, bits=6), adc_quantize(x, bits=6, full_scale=None))


def test_fixed_range_saturates_values_beyond_full_scale():
    x = torch.tensor([[0.5, -0.5, 10.0, -8.0]])
    q = adc_quantize(x, bits=8, full_scale=1.0)
    assert float(q.abs().max()) <= 1.0 + 1e-6          # 10.0 / -8.0 saturate to the fixed range


def test_fixed_range_is_never_better_than_best_case_auto():
    # auto-ranging is the optimistic best case; a single fixed range (even generously sized to the
    # global peak, so nothing saturates) is equal-or-worse for the same bit budget.
    x = torch.randn(8, 64)
    full = float(x.abs().max())
    auto = fidelity(x, adc_quantize(x, bits=4))
    fixed = fidelity(x, adc_quantize(x, bits=4, full_scale=full))
    assert fixed <= auto + 1e-6


def test_saturation_fraction_counts_out_of_range_values():
    x = torch.tensor([1.0, -2.0, 3.0, -4.0])
    assert abs(adc_saturation_fraction(x, full_scale=2.5) - 0.5) < 1e-9   # 3, -4 clip
    assert adc_saturation_fraction(x, full_scale=100.0) == 0.0
    assert 0.0 <= adc_saturation_fraction(torch.randn(50), full_scale=0.3) <= 1.0


def test_fixed_range_is_deterministic_and_non_mutating():
    x = torch.randn(4, 16)
    x0 = x.clone()
    a = adc_quantize(x, bits=6, full_scale=2.0)
    b = adc_quantize(x, bits=6, full_scale=2.0)
    assert torch.equal(a, b)
    assert torch.equal(x, x0)


def test_adcquant_still_defaults_to_auto():
    x = torch.randn(4, 20)
    assert torch.equal(ADCQuant(6)(x), adc_quantize(x, bits=6))
