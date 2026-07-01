"""W5 · F3 — recovery recipe in the report. The product promises "what recovers accuracy", but
recover() is only reachable via `validate`. recommend_recovery(model, inputs, profile) picks the
most-fragile linear layers, captures each layer's real input activations, and returns a per-layer
recipe whose calibrated fidelity never regresses against the naive baseline. analyze() surfaces it
when given inputs; the HTML shows it."""
from __future__ import annotations

from analog_ready.zoo import build, example_inputs
from analog_ready.profiles import load_profile
from analog_ready.analyze import analyze
from analog_ready.report import render_html


def test_recommend_recovery_returns_nonregressing_recipes():
    from analog_ready.recovery import recommend_recovery
    m, x, p = build("mlp"), example_inputs("mlp"), load_profile("aimc_pcm_4bit")
    recs = recommend_recovery(m, x, p, seed=0)
    assert len(recs) >= 1
    for r in recs:
        assert "name" in r and "bits" in r
        assert "degraded_fidelity" in r and "recovered_fidelity" in r
        assert r["recovered_fidelity"] >= r["degraded_fidelity"] - 1e-6   # never worse than naive


def test_recommend_recovery_is_deterministic():
    from analog_ready.recovery import recommend_recovery
    m, x, p = build("mlp"), example_inputs("mlp"), load_profile("aimc_pcm_4bit")
    assert recommend_recovery(m, x, p, seed=0) == recommend_recovery(m, x, p, seed=0)


def test_analyze_surfaces_recovery_in_report():
    m, x, p = build("mlp"), example_inputs("mlp"), load_profile("aimc_pcm_4bit")
    rep = analyze(m, p, inputs=x)
    recovery = rep.to_dict().get("recovery")
    assert recovery and len(recovery) >= 1
    assert "ecover" in render_html(rep)               # "Recovery" section present


def test_analyze_without_inputs_has_no_recovery():
    m, p = build("mlp"), load_profile("aimc_pcm_4bit")
    assert not analyze(m, p).to_dict().get("recovery")
