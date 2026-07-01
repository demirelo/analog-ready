"""W5 · F2 — coverage in the report. The CoverageReport (what instrumentation actually touched)
already exists but never reaches analyze()'s output. A buyer needs to see "architecture-broad, here
is what we replaced/skipped", not infer it. analyze(inputs=...) attaches it; the HTML shows it.
Without inputs, analyze stays backward-compatible."""
from __future__ import annotations

from analog_ready.zoo import build, example_inputs
from analog_ready.profiles import load_profile
from analog_ready.analyze import analyze
from analog_ready.report import render_html


def test_analyze_attaches_coverage_when_inputs_given():
    m, x, p = build("mlp"), example_inputs("mlp"), load_profile("aimc_pcm_4bit")
    cov = analyze(m, p, inputs=x).to_dict().get("coverage")
    assert cov, "coverage missing"
    assert cov["modules_total"] > 0
    assert cov["linear_replaced"] > 0
    assert "coverage_confidence" in cov
    assert "attention_projection_match_rate" in cov


def test_report_html_shows_coverage_section():
    m, x, p = build("mlp"), example_inputs("mlp"), load_profile("aimc_pcm_4bit")
    html = render_html(analyze(m, p, inputs=x))
    assert "overage" in html                 # "Coverage" heading
    assert "replaced" in html.lower()


def test_analyze_without_inputs_has_no_coverage():
    m, p = build("mlp"), load_profile("aimc_pcm_4bit")
    assert not analyze(m, p).to_dict().get("coverage")
