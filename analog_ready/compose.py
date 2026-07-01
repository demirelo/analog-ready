"""W10 — the capstone: compose every modeled analog effect into one end-of-life fidelity.

Rungs W7-W9 each modeled ONE effect in isolation (programming noise, conductance drift, ADC readout
quantization). A real deployment suffers them together: at elapsed time t the weights have drifted
AND carry programming noise, and every layer's output is read back through the ADC. `end_of_life_*`
applies all three from a single profile and reports the combined output fidelity vs the clean
baseline — the honest "what does my model actually produce on this chip after time t" number.

Composition order (per eligible Linear/Conv2d): the weight is PROGRAMMED once at t0 (write noise),
then AGES (drift to t); the analog bias drifts likewise; then the output is read back through the
ADC. Programming noise and drift are RE-DRAWN every draw AND from DISTINCT seed offsets, so the two
Gaussian sources are decorrelated rather than reusing one draw.

HONESTY: this INHERITS every caveat of its components — literature-default (not measured) noise/
drift/ENOB, the best-case AUTO-RANGED ADC (real fixed-reference ADCs are usually worse), and per-sample
rather than per-column ADC granularity. It is a noisy mean-over-draws estimate — read it
DIRECTIONALLY (a best-case-leaning composite), not as a calibrated guarantee or a rigorous bound.
"""
from __future__ import annotations

import copy

import torch

from analog_ready.accuracy import _logits
from analog_ready.adc import ADCQuant, FixedADCQuant
from analog_ready.device_noise import program_noise
from analog_ready.drift import drift_weight
from analog_ready.instrument.replace import instrument
from analog_ready.sweeps import _baseline, _eligible, _profile_val, fidelity

# Distinct seed offsets per Gaussian source: program_noise and drift_weight both draw from
# fork_rng()+manual_seed(seed), so passing the same seed would make their draws bit-identical
# (correlated). Offsetting decorrelates the three sources while staying fully deterministic.
_DRIFT_W_SEED = 1_000_003
_DRIFT_B_SEED = 2_000_003


def _coeffs(profile) -> dict:
    """The composite's parameters from the profile (0 / None where the profile is silent, so a
    profile missing an effect simply drops that term rather than erroring)."""
    return {
        "prog": float(_profile_val(profile, "prog_noise_sigma") or 0.0),
        "read": float(_profile_val(profile, "read_noise_sigma") or 0.0),
        "nu": float(_profile_val(profile, "drift_nu") or 0.0),
        "sigma_nu": float(_profile_val(profile, "drift_sigma_nu") or 0.0),
        "enob": _profile_val(profile, "enob_avail"),
    }


def _composite_variant(model, c: dict, t_s: float, draw_seed: int, adc_full_scale=None):
    """A deep copy of `model` with the full composite applied for ONE draw: program-then-drift on
    each eligible weight (physical order — programmed once at t0, then ages), drift on the analog
    bias, then the ADC readout instrumented on outputs. Distinct seed offsets keep the programming
    and drift Gaussian sources decorrelated. Never touches the caller's model.

    `adc_full_scale=None` uses the best-case AUTO-ranged ADC (W9). A fraction in (0, 1] uses the
    realistic per-tensor FIXED-range ADC (W12), so outlier activations saturate — the honest choice
    when comparing against measured silicon."""
    variant = copy.deepcopy(model).eval()
    with torch.no_grad():
        for m in _eligible(variant):
            w = program_noise(m.weight.detach().clone(), prop_sigma=c["prog"], read_sigma=c["read"],
                              seed=draw_seed)
            w = drift_weight(w, nu=c["nu"], t_s=t_s, t0=1.0, sigma_nu=c["sigma_nu"],
                             seed=draw_seed + _DRIFT_W_SEED)
            m.weight.copy_(w)
            if m.bias is not None:
                m.bias.copy_(drift_weight(m.bias.detach().clone(), nu=c["nu"], t_s=t_s, t0=1.0,
                                          sigma_nu=c["sigma_nu"], seed=draw_seed + _DRIFT_B_SEED))
        if c["enob"] is not None:
            adc = (FixedADCQuant(c["enob"], float(adc_full_scale)) if adc_full_scale is not None
                   else ADCQuant(c["enob"]))
            instrument(variant, adc)   # ADC readout quant on every eligible output
    return variant


def _combined_fidelity(model, baseline, inputs, c: dict, t_s: float, draws: int, seed: int,
                       adc_full_scale=None) -> float:
    fids = []
    for i in range(max(draws, 1)):
        with torch.no_grad():
            out = _composite_variant(model, c, t_s, seed + i, adc_full_scale)(inputs)
        fids.append(fidelity(baseline, out))
    return sum(fids) / len(fids)


def _combined_accuracy(model, inputs, labels, c: dict, t_s: float, draws: int, seed: int,
                       adc_full_scale=None) -> float:
    accs = []
    for i in range(max(draws, 1)):
        with torch.no_grad():
            preds = _logits(_composite_variant(model, c, t_s, seed + i, adc_full_scale)(inputs)).argmax(-1)
        accs.append(float((preds == labels).float().mean()))
    return sum(accs) / len(accs)


def end_of_life_curve(model, inputs, profile, *, times: list, draws: int = 8, seed: int = 0) -> list:
    """[{"t": t, "fidelity": f}, ...] — combined-effect output fidelity vs the clean baseline, for
    each elapsed time in `times`. Deterministic given `seed`; never mutates the caller's model."""
    c = _coeffs(profile)
    baseline = _baseline(model, inputs)
    return [{"t": t, "fidelity": _combined_fidelity(model, baseline, inputs, c, t, draws, seed)}
            for t in times]


def end_of_life_fidelity(model, inputs, profile, *, t_s: float = 86400.0, draws: int = 8,
                         seed: int = 0) -> float:
    """Combined-effect output fidelity at a single elapsed time `t_s` (default 1 day)."""
    return end_of_life_curve(model, inputs, profile, times=[t_s], draws=draws, seed=seed)[0]["fidelity"]


def end_of_life_accuracy(model, inputs, labels, profile, *, t_s: float = 86400.0, draws: int = 8,
                         seed: int = 0, adc_full_scale=None) -> float:
    """Mean top-1 ACCURACY under the composed effects at elapsed time `t_s` — the accuracy twin of
    end_of_life_fidelity, used by the pilot to predict a top-1 drop comparable to a measured one.
    `adc_full_scale=None` uses the best-case auto-ranged ADC; a fraction in (0, 1] uses the realistic
    per-tensor fixed-range ADC (W12). Deterministic; never mutates the caller's model."""
    return _combined_accuracy(model, inputs, labels, _coeffs(profile), t_s, draws, seed,
                              adc_full_scale)
