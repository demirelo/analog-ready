"""W11 · F3 — the stale "reported separately, not composed" caveat. W10 added the End-of-life section
that DOES compose drift + programming noise + ADC, so the blanket "not composed" claim in the Limits
list is now false and must be corrected."""
from analog_ready.analyze import analyze
from analog_ready.profiles import load_profile
from analog_ready.report import render_html
from analog_ready.zoo import build, example_inputs


def _report_html():
    m, x = build("mlp"), example_inputs("mlp")
    return render_html(analyze(m, load_profile("aimc_pcm_4bit"), inputs=x), redact=False)


def test_stale_not_composed_blanket_caveat_is_gone():
    assert "reported separately, not composed" not in _report_html()


def test_report_acknowledges_the_composed_end_of_life_view():
    html = _report_html().lower()
    # the report both shows per-effect sections AND a composed one; the caveats should reflect that.
    assert "combined" in html          # the "End-of-life — all effects combined" section
    assert "end-of-life" in html
