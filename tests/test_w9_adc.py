"""W9 · F1 — ADC readout quantization (immutable acceptance oracle).

The analog dot-product is read back to digital through an ADC with only a few effective bits — a
dominant precision bottleneck the cost model already prices (enob_avail) but nothing yet applies to
the signal. `adc_quantize` uniformly quantizes a layer's OUTPUT to `bits` effective bits over its
per-sample dynamic range. Deterministic (no RNG), non-mutating; high bits -> ~identity.
"""
import torch

from analog_ready.adc import ADCQuant, adc_quantize
from analog_ready.sweeps import fidelity


def test_high_bits_is_near_identity():
    x = torch.randn(4, 32)
    assert fidelity(x, adc_quantize(x, bits=12)) > 0.999


def test_low_bits_degrades_more_than_high_bits():
    x = torch.randn(4, 32)
    assert fidelity(x, adc_quantize(x, bits=2)) < fidelity(x, adc_quantize(x, bits=12))


def test_more_bits_never_lower_fidelity():
    x = torch.randn(8, 64)
    fids = [fidelity(x, adc_quantize(x, bits=b)) for b in (2, 4, 8)]
    assert fids[0] <= fids[1] <= fids[2]


def test_deterministic_and_non_mutating():
    x = torch.randn(4, 16)
    x0 = x.clone()
    a = adc_quantize(x, bits=6)
    b = adc_quantize(x, bits=6)
    assert torch.equal(a, b)
    assert torch.equal(x, x0)


def test_all_zero_input_is_safe():
    x = torch.zeros(3, 8)
    q = adc_quantize(x, bits=6)
    assert not torch.isnan(q).any()
    assert torch.equal(q, x)


def test_adcquant_callable_matches_function_and_is_the_noise_interface():
    x = torch.randn(4, 20)
    assert torch.equal(ADCQuant(6)(x), adc_quantize(x, bits=6))
