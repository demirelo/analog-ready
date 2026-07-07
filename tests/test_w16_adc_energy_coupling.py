"""W16 · ADC energy <-> resolution coupling (the X<->Z gate coupling).

A flat per-conversion `adc_energy_pj` decouples the X (converter-energy) and Z (precision =
`enob_avail`) break-even gates: an ENOB sweep moves Z while conversion energy sits flat. Real ADC
energy scales ~2^ENOB..4^ENOB (Murmann ADC-survey Walden/Schreier FoMs). These oracles pin:
  * the pure `adc_energy_pj_for_enob` scaling law,
  * flat is the DEFAULT (v0.1 back-compat — no existing number moves), and
  * `adc_energy_model: coupled` makes the cost model's converter energy track `enob_avail`.

Deliberately torch-free: `cost_model` and `core.profile` import without torch, so the coupling is
unit-tested directly on a duck-typed op feature and a hand-built profile — no zoo model needed."""
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
    """A duck-typed OpFeature (only the attributes the cost model reads), so no torch import."""
    macs = M * K * N
    return SimpleNamespace(
        name="op", M=M, K=K, N=N, macs=macs, flops=2 * macs,
        weight_numel=K * N, act_in_numel=M * K, act_out_numel=M * N)


def _profile(**overrides):
    """A minimal cost-model-complete profile; `overrides` add/replace fields (value only)."""
    base = {
        "array_rows": 256, "array_cols": 256, "weight_bits": 4, "input_bits": 4,
        "enob_avail": 6.0, "mac_energy_pj": 0.04, "adc_energy_pj": 3.17, "dac_energy_pj": 10.0,
        "mem_energy_pj_per_byte": 0.1, "digital_mac_energy_pj": 0.07,
    }
    base.update(overrides)
    fields = {k: ProfileField(value=v, redaction="exact_ok") for k, v in base.items()}
    return HardwareProfile(name="t", fields=fields)


def _tiles(feat, profile):
    rows = int(profile.fields["array_rows"].value)
    cols = int(profile.fields["array_cols"].value)
    dac_ops = feat.M * feat.K * math.ceil(feat.N / cols)
    adc_ops = feat.M * feat.N * math.ceil(feat.K / rows)
    return dac_ops, adc_ops


# --------------------------------------------------------------- the scaling law itself
def test_scaling_law_doubles_per_bit_by_default():
    # Walden default (base=2): +1 ENOB doubles, -1 halves, at the reference it is the identity.
    assert adc_energy_pj_for_enob(ADC_ENERGY_REF_ENOB, 3.17) == pytest.approx(3.17)
    assert adc_energy_pj_for_enob(9.0, 3.17) == pytest.approx(6.34)
    assert adc_energy_pj_for_enob(7.0, 3.17) == pytest.approx(1.585)
    assert ADC_ENERGY_ENOB_BASE == 2.0


def test_scaling_law_thermal_regime_quadruples_per_bit():
    assert adc_energy_pj_for_enob(9.0, 3.17, base=4.0) == pytest.approx(3.17 * 4)
    assert adc_energy_pj_for_enob(10.0, 3.17, base=4.0) == pytest.approx(3.17 * 16)


def test_scaling_law_is_strictly_monotonic_in_enob():
    e = [adc_energy_pj_for_enob(b, 3.17) for b in range(4, 13)]
    assert all(b > a for a, b in zip(e, e[1:]))


def test_scaling_law_rejects_nonpositive_base():
    for bad in (0.0, -2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            adc_energy_pj_for_enob(8.0, 3.17, base=bad)


# --------------------------------------------------------------- default is flat (back-compat)
def test_flat_is_the_default_and_ignores_enob():
    feat = _feat()
    dac_ops, adc_ops = _tiles(feat, _profile())
    lo = estimate_op(feat, _profile(enob_avail=4.0))
    hi = estimate_op(feat, _profile(enob_avail=12.0))
    # No adc_energy_model field -> flat -> ENOB does not touch conversion energy at all.
    assert lo.conversion_pj == hi.conversion_pj
    expected = dac_ops * 10.0 + adc_ops * 3.17
    assert lo.conversion_pj == pytest.approx(expected)


def test_explicit_flat_matches_default():
    feat = _feat()
    assert (estimate_op(feat, _profile()).conversion_pj
            == pytest.approx(estimate_op(feat, _profile(adc_energy_model="flat")).conversion_pj))


# --------------------------------------------------------------- coupled mode couples X to Z
def test_coupled_at_reference_enob_equals_flat():
    feat = _feat()
    flat = estimate_op(feat, _profile(enob_avail=ADC_ENERGY_REF_ENOB))
    coupled = estimate_op(feat, _profile(enob_avail=ADC_ENERGY_REF_ENOB, adc_energy_model="coupled"))
    assert coupled.conversion_pj == pytest.approx(flat.conversion_pj)


def test_coupled_converter_energy_tracks_enob():
    feat = _feat()
    dac_ops, adc_ops = _tiles(feat, _profile())
    low = estimate_op(feat, _profile(enob_avail=6.0, adc_energy_model="coupled"))
    high = estimate_op(feat, _profile(enob_avail=10.0, adc_energy_model="coupled"))
    # higher ENOB => strictly more converter energy per MAC (the economics cliff a flat coeff hides)
    assert high.converter_pj_per_mac > low.converter_pj_per_mac
    # and it matches the scaling law exactly on the ADC term
    exp_low = dac_ops * 10.0 + adc_ops * adc_energy_pj_for_enob(6.0, 3.17)
    assert low.conversion_pj == pytest.approx(exp_low)


def test_coupled_respects_custom_base_and_ref_enob():
    feat = _feat()
    dac_ops, adc_ops = _tiles(feat, _profile())
    c = estimate_op(feat, _profile(enob_avail=9.0, adc_energy_model="coupled",
                                   adc_energy_enob_base=4.0, adc_energy_ref_enob=8.0))
    exp = dac_ops * 10.0 + adc_ops * (3.17 * 4)  # 4^(9-8) = 4
    assert c.conversion_pj == pytest.approx(exp)


def test_coupled_without_enob_is_a_loud_error():
    fields = {k: ProfileField(value=v, redaction="exact_ok") for k, v in {
        "array_rows": 256, "array_cols": 256, "weight_bits": 4, "input_bits": 4,
        "mac_energy_pj": 0.04, "adc_energy_pj": 3.17, "dac_energy_pj": 10.0,
        "mem_energy_pj_per_byte": 0.1, "digital_mac_energy_pj": 0.07,
        "adc_energy_model": "coupled",
    }.items()}
    with pytest.raises(ValueError, match="enob_avail"):
        estimate_op(_feat(), HardwareProfile(name="no_enob", fields=fields))
