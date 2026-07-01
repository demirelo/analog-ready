"""W10 · the capstone — compose ALL modeled analog effects into one end-of-life number.

At elapsed time t on a given profile the weights have DRIFTED and carry PROGRAMMING noise, and every
readout goes through the ADC at its effective bits. `end_of_life_fidelity` / `end_of_life_curve`
apply drift(t) + programming-noise (weights, and drift on the analog bias) + ADC readout
quantization (outputs) together and report the combined output fidelity vs the clean baseline — the
honest "what does my model actually produce on this chip after a year" answer. Composing strictly
MORE degradation than any single effect; deterministic; non-mutating.
"""
import torch

from analog_ready.compose import end_of_life_curve, end_of_life_fidelity
from analog_ready.sweeps import profile_adc_fidelity
from analog_ready.analyze import analyze
from analog_ready.profiles import load_profile
from analog_ready.report import render_html
from analog_ready.zoo import build, example_inputs

_DAY = 86400.0
_YEAR = 3.15e7


def test_end_of_life_degrades_and_is_finite():
    model, inputs = build("mlp"), example_inputs("mlp")
    f = end_of_life_fidelity(model, inputs, load_profile("aimc_pcm_4bit"), t_s=1.0, draws=4, seed=0)
    assert -1.001 <= f <= 1.001
    assert f < 0.9999          # even at t0 (no drift), programming noise + ADC readout degrade it


def test_end_of_life_deterministic_and_non_mutating():
    model, inputs = build("mlp"), example_inputs("mlp")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    a = end_of_life_fidelity(model, inputs, load_profile("aimc_pcm_4bit"), t_s=_DAY, draws=4, seed=0)
    b = end_of_life_fidelity(model, inputs, load_profile("aimc_pcm_4bit"), t_s=_DAY, draws=4, seed=0)
    assert a == b
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before)


def test_composition_is_worse_than_adc_alone():
    # the whole point of composing: end-of-life (drift + programming noise + ADC) must be strictly
    # more degraded than the ADC readout effect on its own.
    model, inputs = build("mlp"), example_inputs("mlp")
    prof = load_profile("aimc_pcm_4bit")
    eol_year = end_of_life_fidelity(model, inputs, prof, t_s=_YEAR, draws=4, seed=0)
    adc_only = profile_adc_fidelity(model, inputs, prof)
    assert eol_year < adc_only


def test_end_of_life_curve_decays_over_time():
    model, inputs = build("mlp"), example_inputs("mlp")
    curve = end_of_life_curve(model, inputs, load_profile("aimc_pcm_4bit"),
                              times=[1.0, _DAY, _YEAR], draws=4, seed=0)
    assert [p["t"] for p in curve] == [1.0, _DAY, _YEAR]
    assert curve[-1]["fidelity"] <= curve[0]["fidelity"]     # drift accumulates over time
    assert curve[0]["fidelity"] < 1.0                        # noise + ADC present even at t0


def test_analyze_surfaces_end_of_life_and_excludes_from_shareable_view():
    model, inputs = build("mlp"), example_inputs("mlp")
    rep = analyze(model, load_profile("aimc_pcm_4bit"), inputs=inputs)
    d = rep.to_dict()
    assert "end_of_life" in d and len(d["end_of_life"]) >= 2
    assert all("t" in p and "fidelity" in p for p in d["end_of_life"])
    assert "combined" in render_html(rep, redact=False).lower()   # a combined-effects section renders
    assert "end_of_life" not in rep.redacted()                    # local artifact, not shareable
