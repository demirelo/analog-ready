"""W16 - resolution-coupled ADC energy with a separate nominal-resolution input.

The delivered ``enob_avail`` belongs to the precision gate.  The optional ADC energy model uses
``adc_enob_for_energy`` so device noise does not silently double-count converter resolution.
"""
import math
from types import SimpleNamespace

import pytest

from analog_ready.cost_model import (
    ADC_ENERGY_ENOB_BASE,
    ADC_ENERGY_REF_ENOB,
    adc_energy_pj_for_enob,
    estimate_op,
)
from analog_ready.core.profile import HardwareProfile, ProfileField


def _feat(M=64, K=512, N=512):
    macs = M * K * N
    return SimpleNamespace(
        name="op", M=M, K=K, N=N, macs=macs, flops=2 * macs,
        weight_numel=K * N, act_in_numel=M * K, act_out_numel=M * N)


def _profile(**overrides):
    values = {
        "array_rows": 256, "array_cols": 256, "weight_bits": 4, "input_bits": 4,
        "enob_avail": 6.0, "mac_energy_pj": 0.04, "adc_energy_pj": 3.17,
        "dac_energy_pj": 10.0, "mem_energy_pj_per_byte": 0.1, "digital_mac_energy_pj": 0.07,
    }
    values.update(overrides)
    return HardwareProfile(
        name="test",
        fields={key: ProfileField(value=value, redaction="exact_ok")
                for key, value in values.items()},
    )


def _adc_ops(feat, profile):
    rows = int(profile.fields["array_rows"].value)
    return feat.M * feat.N * math.ceil(feat.K / rows)


def test_scaling_law_uses_seven_bit_literature_anchor():
    assert ADC_ENERGY_REF_ENOB == 7.0
    assert ADC_ENERGY_ENOB_BASE == 2.0
    assert adc_energy_pj_for_enob(7.0, 3.17) == pytest.approx(3.17)
    assert adc_energy_pj_for_enob(8.0, 3.17) == pytest.approx(6.34)
    assert adc_energy_pj_for_enob(6.0, 3.17) == pytest.approx(1.585)
    assert adc_energy_pj_for_enob(9.0, 3.17, base=4.0) == pytest.approx(3.17 * 16)


def test_scaling_law_rejects_invalid_inputs():
    for bad in (0.0, 1.0, 5.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            adc_energy_pj_for_enob(7.0, 3.17, base=bad)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            adc_energy_pj_for_enob(bad, 3.17)
        with pytest.raises(ValueError):
            adc_energy_pj_for_enob(7.0, 3.17, ref_enob=bad)
        with pytest.raises(ValueError):
            adc_energy_pj_for_enob(7.0, bad)


def test_flat_default_keeps_conversion_energy_independent_of_effective_enob():
    feat = _feat()
    low = estimate_op(feat, _profile(enob_avail=4.0))
    high = estimate_op(feat, _profile(enob_avail=12.0))
    assert low.conversion_pj == pytest.approx(high.conversion_pj)


def test_coupled_mode_requires_nominal_converter_resolution():
    with pytest.raises(ValueError, match="adc_enob_for_energy"):
        estimate_op(_feat(), _profile(adc_energy_model="coupled"))


def test_coupled_mode_separates_converter_resolution_from_delivered_enob():
    feat = _feat()
    low_effective = estimate_op(
        feat, _profile(adc_energy_model="coupled", adc_enob_for_energy=8.0, enob_avail=4.0))
    high_effective = estimate_op(
        feat, _profile(adc_energy_model="coupled", adc_enob_for_energy=8.0, enob_avail=12.0))
    assert low_effective.conversion_pj == pytest.approx(high_effective.conversion_pj)

    low_nominal = estimate_op(
        feat, _profile(adc_energy_model="coupled", adc_enob_for_energy=6.0, enob_avail=6.0))
    high_nominal = estimate_op(
        feat, _profile(adc_energy_model="coupled", adc_enob_for_energy=8.0, enob_avail=6.0))
    assert high_nominal.conversion_pj > low_nominal.conversion_pj
    adc_ops = _adc_ops(feat, _profile())
    dac_and_low_adc = low_nominal.conversion_pj
    expected_low = adc_ops * adc_energy_pj_for_enob(6.0, 3.17)
    # Subtract the unchanged DAC term to isolate the coupled ADC contribution.
    assert dac_and_low_adc - expected_low == pytest.approx(
        (_feat().M * _feat().K * math.ceil(_feat().N / 256)) * 10.0)
    assert low_nominal.coefficients["adc_energy_pj"].value == pytest.approx(3.17)
    assert low_nominal.coefficients["adc_energy_effective_pj"].value == pytest.approx(
        adc_energy_pj_for_enob(6.0, 3.17))


def test_unknown_adc_model_is_rejected():
    with pytest.raises(ValueError, match="unknown adc_energy_model"):
        estimate_op(_feat(), _profile(adc_energy_model="magic"))
