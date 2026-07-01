"""W9 · F3/F4 — analyze() drives the ADC readout quantization from the profile's own enob_avail
(already a real field, used by the precision gate) and surfaces a fidelity number in the report.
Redaction-safe: the derived adc_fidelity is a local artifact, absent from the shareable view."""
from analog_ready.analyze import analyze
from analog_ready.profiles import load_profile
from analog_ready.report import render_html
from analog_ready.zoo import build, example_inputs


def test_analyze_surfaces_adc_fidelity_from_profile_enob():
    model, inputs = build("mlp"), example_inputs("mlp")
    rep = analyze(model, load_profile("aimc_pcm_4bit"), inputs=inputs)
    d = rep.to_dict()
    assert "adc_fidelity" in d
    assert 0.0 <= d["adc_fidelity"] <= 1.0
    html = render_html(rep, redact=False)
    assert "readout" in html.lower()          # an ADC readout-precision section is rendered


def test_adc_fidelity_absent_without_inputs():
    model = build("mlp")
    rep = analyze(model, load_profile("aimc_pcm_4bit"))
    assert "adc_fidelity" not in rep.to_dict()


def test_adc_fidelity_excluded_from_shareable_view():
    model, inputs = build("mlp"), example_inputs("mlp")
    rep = analyze(model, load_profile("aimc_pcm_4bit"), inputs=inputs)
    assert "adc_fidelity" not in rep.redacted()
