"""W2 ships named AIMC profiles (+ a photonic demo profile) that carry the cost coefficients the
break-even model needs, with the per-field redaction from W1 preserved."""
import pytest

PROFILES = ["aimc_pcm_4bit", "aimc_reram_6enob", "aimc_sram_8bit", "photonic_clements_mzi"]
REQUIRED = [
    "weight_bits", "input_bits", "enob_avail", "array_rows", "array_cols",
    "mac_energy_pj", "adc_energy_pj", "dac_energy_pj", "mem_energy_pj_per_byte",
    "digital_mac_energy_pj", "peak_flops", "peak_bw_bytes",
]


@pytest.mark.parametrize("name", PROFILES)
def test_profile_loads_with_required_cost_fields(name):
    from analog_ready.profiles import load_profile

    p = load_profile(name)
    assert p.name == name
    for k in REQUIRED:
        assert k in p.fields, f"{name} missing cost field {k}"
        assert p.fields[k].value is not None


def test_profile_redaction_hides_sensitive_cost():
    from analog_ready.profiles import load_profile

    p = load_profile("aimc_pcm_4bit")
    red = p.redacted()
    assert "mem_energy_pj_per_byte" not in red          # declared hidden -> key omitted entirely
    # a bucketed coefficient is coarsened, never the exact figure
    assert red["mac_energy_pj"] != p.fields["mac_energy_pj"].value


def test_unknown_profile_raises():
    from analog_ready.profiles import load_profile

    with pytest.raises((FileNotFoundError, ValueError, KeyError)):
        load_profile("does_not_exist")
