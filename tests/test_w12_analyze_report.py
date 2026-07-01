"""W12 · F3/F4 — the analyze report surfaces the REALISTIC fixed-range ADC (fidelity + saturation)
alongside the best-case auto number from W9, driven by the profile's `adc_full_scale`. Both are local
artifacts, excluded from the shareable view."""
from analog_ready.analyze import analyze
from analog_ready.profiles import load_profile
from analog_ready.report import render_html
from analog_ready.zoo import build, example_inputs


def _report(profile="aimc_pcm_4bit"):
    m, x = build("mlp"), example_inputs("mlp")
    return analyze(m, load_profile(profile), inputs=x)


def test_analyze_surfaces_fixed_range_adc_and_saturation():
    rep = _report()
    d = rep.to_dict()
    assert "adc_fixed_fidelity" in d
    assert "adc_saturation" in d
    assert 0.0 <= d["adc_fixed_fidelity"] <= 1.0
    assert 0.0 <= d["adc_saturation"] <= 1.0
    # the realistic fixed-range fidelity is never better than the best-case auto ADC fidelity
    assert d["adc_fixed_fidelity"] <= d["adc_fidelity"] + 1e-6


def test_report_shows_a_fixed_range_or_saturation_caveat():
    html = render_html(_report(), redact=False).lower()
    assert "fixed" in html or "saturat" in html


def test_fixed_range_fields_excluded_from_shareable_view():
    red = _report().redacted()
    assert "adc_fixed_fidelity" not in red
    assert "adc_saturation" not in red
