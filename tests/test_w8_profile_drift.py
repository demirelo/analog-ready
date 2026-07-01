"""W8 · F3 — drift coefficients in the AIMC profiles. The mean drift exponent `drift_nu` is a broadly
published materials constant (~0.06 for PCM), so it is `bucket`-shareable; the device-to-device
`drift_sigma_nu` variability (what actually degrades accuracy) is characterisation IP, so `hidden`.
SRAM does not drift, so it declares no drift coefficients."""
import json

from analog_ready.profiles import load_profile


def test_pcm_and_reram_carry_drift_coeffs():
    for name in ("aimc_pcm_4bit", "aimc_reram_6enob"):
        p = load_profile(name)
        assert "drift_nu" in p.fields, f"{name} missing drift_nu"
        assert "drift_sigma_nu" in p.fields, f"{name} missing drift_sigma_nu"
        assert p.fields["drift_nu"].value > 0.0
        assert p.fields["drift_sigma_nu"].value >= 0.0


def test_drift_nu_is_bucketed_and_sigma_is_hidden():
    p = load_profile("aimc_pcm_4bit")
    assert p.fields["drift_nu"].redaction == "bucket"
    assert p.fields["drift_sigma_nu"].redaction == "hidden"
    red = p.redacted()
    assert "drift_nu" in red                     # bucket -> a coarsened band is shareable
    assert "drift_sigma_nu" not in red           # hidden -> omitted entirely
    assert str(p.fields["drift_sigma_nu"].value) not in json.dumps(red)


def test_sram_has_negligible_or_no_drift():
    # SRAM analog cells are not conductance-drift devices; if a drift_nu is present it must be ~0.
    p = load_profile("aimc_sram_8bit")
    if "drift_nu" in p.fields:
        assert p.fields["drift_nu"].value == 0.0
