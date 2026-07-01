"""W7 · F3 — the AIMC profiles carry device-noise coefficients (programming + read), and because a
vendor's noise profile is crown-jewel IP they are `hidden`: never emitted by redacted()."""
import json

from analog_ready.profiles import load_profile

_AIMC = ("aimc_pcm_4bit", "aimc_sram_8bit", "aimc_reram_6enob")


def test_aimc_profiles_carry_program_and_read_noise_coeffs():
    for name in _AIMC:
        p = load_profile(name)
        assert "prog_noise_sigma" in p.fields, f"{name} missing prog_noise_sigma"
        assert "read_noise_sigma" in p.fields, f"{name} missing read_noise_sigma"
        assert p.fields["prog_noise_sigma"].value > 0.0
        assert p.fields["read_noise_sigma"].value >= 0.0


def test_noise_coeffs_are_hidden_and_never_leak_in_redacted():
    p = load_profile("aimc_pcm_4bit")
    assert p.fields["prog_noise_sigma"].redaction == "hidden"
    assert p.fields["read_noise_sigma"].redaction == "hidden"
    red = p.redacted()
    assert "prog_noise_sigma" not in red
    assert "read_noise_sigma" not in red
    # the raw values must not appear anywhere in the redacted blob (values or repr)
    blob = json.dumps(red)
    assert str(p.fields["prog_noise_sigma"].value) not in blob
    assert str(p.fields["read_noise_sigma"].value) not in blob


def test_hidden_noise_coeff_does_not_leak_via_repr():
    p = load_profile("aimc_pcm_4bit")
    assert str(p.fields["prog_noise_sigma"].value) not in repr(p.fields["prog_noise_sigma"])
