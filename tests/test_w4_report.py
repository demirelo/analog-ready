"""W4 report — the sales artifact. analyze() already produces the data; report.render_html turns it
into a SELF-CONTAINED HTML page (no remote CSS/JS/asset fetches — a local-only run must render
behind a firewall with the network off), and render_json is the machine-readable twin. Both honour
redaction: with redact=True a `hidden` profile coefficient never appears, and the raw coefficient
block is dropped. The page always carries the cost-model spine (compute/conversion/DRAM energy), the
break-even verdict, the sensitivity flip, and an honest 'Limits of this estimate' section."""
import json

import torch.nn as nn


def _report(profile="aimc_sram_8bit"):
    from analog_ready.analyze import analyze
    from analog_ready.profiles import load_profile

    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1024))
    return analyze(model, load_profile(profile))


def test_render_html_is_a_standalone_document():
    from analog_ready.report import render_html

    html = render_html(_report())
    assert isinstance(html, str)
    low = html.lower()
    assert "<html" in low and "</html>" in low
    # SELF-CONTAINED: no remote resource loads (inline CSS/JS only). External <a> links are fine
    # (navigation, not a fetch); resource-loading attributes pointed at the network are not.
    assert 'src="http' not in low
    assert "<link" not in low
    assert "@import" not in low


def test_html_carries_the_cost_model_spine_and_limits():
    from analog_ready.report import render_html

    rep = _report()
    html = render_html(rep)
    low = html.lower()
    # the spine: the three energy components are shown
    assert "compute" in low and "conversion" in low and "dram" in low
    # the buyer-facing verdict + where it flips
    assert "break-even" in low or "break even" in low
    assert "sensitivity" in low or "flip" in low
    # honesty section is mandatory
    assert "limits" in low
    # one visible row per analysed op
    assert html.count("<tr") >= len(rep.ops)


def test_render_json_round_trips():
    from analog_ready.report import render_json

    rep = _report()
    obj = json.loads(render_json(rep))
    assert obj["model"] == rep.model
    assert obj["profile"] == rep.profile_name
    assert len(obj["ops"]) == len(rep.ops)


def test_redacted_outputs_never_leak_a_hidden_coefficient():
    from analog_ready.report import render_html, render_json

    rep = _report("aimc_pcm_4bit")  # this profile has a hidden mem-energy coefficient
    for blob in (render_html(rep, redact=True), render_json(rep, redact=True)):
        assert "mem_energy_pj_per_byte" not in blob
        assert "coefficients" not in blob


def test_full_report_may_show_coefficients_but_redacted_must_not():
    """Non-redacted is the internal view (coefficients allowed); redact=True is the shareable view."""
    from analog_ready.report import render_json

    rep = _report("aimc_pcm_4bit")
    full = render_json(rep, redact=False)
    shared = render_json(rep, redact=True)
    assert len(shared) < len(full) or "coefficients" not in shared
    assert "coefficients" not in shared
