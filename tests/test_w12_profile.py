"""W12 · F4 — the AIMC profiles declare an `adc_full_scale` (the fixed ADC's full-scale as a fraction
of the per-tensor signal peak; a fraction < 1 means the peaks saturate). Uses the W11 professional
schema (unit/provenance)."""
from analog_ready.profiles import load_profile


def test_pcm_profile_declares_adc_full_scale():
    p = load_profile("aimc_pcm_4bit")
    assert "adc_full_scale" in p.fields
    fld = p.fields["adc_full_scale"]
    assert 0.0 < fld.value <= 1.0            # a fraction of the signal peak the fixed ADC covers
    assert fld.provenance in ("literature", "measured", "estimate")
