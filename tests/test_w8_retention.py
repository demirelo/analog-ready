"""W8 · F2/F4 — the retention curve (fidelity vs elapsed time under drift) and its wiring into the
analyze report from the profile's own drift coefficients. Tests the real analyze entrypoint, not
just the primitive."""
import torch

from analog_ready.drift import retention_curve
from analog_ready.analyze import analyze
from analog_ready.profiles import load_profile
from analog_ready.report import render_html
from analog_ready.zoo import build, example_inputs


def test_retention_curve_shape_and_degradation():
    model, inputs = build("mlp"), example_inputs("mlp")
    curve = retention_curve(model, inputs, nu=0.06, sigma_nu=0.05,
                            times=[1.0, 3600.0, 86400.0], t0=1.0, draws=4, seed=0)
    assert [p["t"] for p in curve] == [1.0, 3600.0, 86400.0]
    assert curve[0]["fidelity"] > 0.99                       # t == t0 -> ~perfect
    assert curve[-1]["fidelity"] < curve[0]["fidelity"]      # fidelity decays over time


def test_retention_curve_reproducible_and_non_mutating():
    model, inputs = build("mlp"), example_inputs("mlp")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    a = retention_curve(model, inputs, nu=0.06, sigma_nu=0.05, times=[86400.0], t0=1.0,
                        draws=3, seed=0)
    b = retention_curve(model, inputs, nu=0.06, sigma_nu=0.05, times=[86400.0], t0=1.0,
                        draws=3, seed=0)
    assert a[0]["fidelity"] == b[0]["fidelity"]
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_analyze_surfaces_retention_from_the_profile_drift():
    model, inputs = build("mlp"), example_inputs("mlp")
    rep = analyze(model, load_profile("aimc_pcm_4bit"), inputs=inputs)
    d = rep.to_dict()
    assert "retention" in d and len(d["retention"]) >= 2          # curve attached
    assert all("t" in p and "fidelity" in p for p in d["retention"])
    html = render_html(rep, redact=False)
    assert "etention" in html                                    # a Retention section is rendered


def test_retention_is_not_in_the_shareable_view():
    # the retention curve is a derived local artifact (like accuracy) — it must not appear in the
    # redacted, shareable report.
    model, inputs = build("mlp"), example_inputs("mlp")
    rep = analyze(model, load_profile("aimc_pcm_4bit"), inputs=inputs)
    assert "retention" not in rep.redacted()
    assert "etention" not in render_html(rep, redact=True)
