"""Post-W12 read-only-review regression guard.

W12's review found a FALSE universal claim — "a fixed-range ADC is equal-or-worse than auto-ranging"
— surviving in generated report text (and, worse, in the redacted *shareable* HTML). It is false: on
multi-axis outputs a single per-tensor fixed range can BEAT a per-sample auto range for the same bit
budget (e.g. gpt2_block: fixed 0.9899 > auto 0.9862). The realistic fixed-range number is USUALLY
below the best-case auto one, but that is not a hard bound. This guard keeps the false bound from
re-appearing in either the local or the shareable report.
"""
from analog_ready.analyze import analyze
from analog_ready.profiles import load_profile
from analog_ready.report import render_html
from analog_ready.zoo import build, example_inputs


def _report(profile="aimc_pcm_4bit"):
    m, x = build("mlp"), example_inputs("mlp")
    return analyze(m, load_profile(profile), inputs=x)


def test_no_universal_fixed_le_auto_bound_in_either_html_view():
    rep = _report()
    for redact in (False, True):
        html = render_html(rep, redact=redact).lower()
        assert "equal-or-worse" not in html, f"false ADC bound shipped (redact={redact})"
        assert "equal or worse" not in html, f"false ADC bound shipped (redact={redact})"


def test_the_counterexample_that_makes_the_bound_false_still_holds():
    # The bound is not merely un-stated — it is genuinely violable: a per-tensor fixed range beats a
    # per-sample auto range on a 3-D transformer-block output. If this ever stops reproducing, the
    # prose hedges are over-cautious and can be revisited (but must never re-assert a hard bound).
    m, x = build("gpt2_block"), example_inputs("gpt2_block")
    d = analyze(m, load_profile("aimc_pcm_4bit"), inputs=x).to_dict()
    assert d["adc_fixed_fidelity"] > d["adc_fidelity"]
