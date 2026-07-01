"""W11 · F6 — professional profile schema. A ProfileField gains optional provenance metadata (unit,
source, uncertainty, provenance, calibration_date) alongside its existing value/redaction — what a
hardware team needs to trust a coefficient. All optional with defaults, so every existing profile and
test still loads unchanged."""
import pytest

from analog_ready.core.profile import ProfileField
from analog_ready.profiles import load_profile


def test_field_carries_professional_metadata():
    f = ProfileField(value=3.17, redaction="bucket", unit="pJ",
                     source="LightCode arXiv:2509.16443", uncertainty="+-50%",
                     provenance="literature", calibration_date="2026-06-30")
    assert f.unit == "pJ"
    assert f.source.startswith("LightCode")
    assert f.uncertainty == "+-50%"
    assert f.provenance == "literature"
    assert f.calibration_date == "2026-06-30"


def test_metadata_is_optional_and_backward_compatible():
    f = ProfileField(value=1.0)                     # the pre-W11 style: value only
    assert f.redaction == "hidden"
    assert f.unit == "" and f.source == "" and f.uncertainty == ""
    assert f.provenance == "literature" and f.calibration_date == ""


def test_provenance_is_validated():
    with pytest.raises(ValueError):
        ProfileField(value=1.0, provenance="totally-made-up")


def test_existing_profiles_load_and_expose_the_schema():
    p = load_profile("aimc_pcm_4bit")
    fld = p.fields["adc_energy_pj"]
    for attr in ("unit", "source", "uncertainty", "provenance", "calibration_date"):
        assert hasattr(fld, attr)


def test_a_profile_declares_measurement_provenance_on_at_least_one_field():
    # the schema must be demonstrated, not just declarable: some AIMC-profile field carries a unit
    # and a provenance so a reader sees the intended usage.
    p = load_profile("aimc_pcm_4bit")
    assert any(f.unit and f.provenance in ("literature", "measured", "estimate")
               for f in p.fields.values())
