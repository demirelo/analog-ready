"""W3 — degradation sweeps. Instrument a model with the W1 module-replacement walker, run it under
a profile-derived noise/quant model, and measure a fidelity metric vs the noiseless baseline across
a swept parameter. Noise is stochastic, so a curve is the MEAN fidelity over `draws` seeded draws;
fixing the seed makes the whole curve reproducible."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from analog_ready.adc import ADCQuant, FixedADCQuant, adc_saturation_fraction
from analog_ready.device_noise import program_noise
from analog_ready.instrument.replace import instrument
from analog_ready.noise import GaussianNoise
from analog_ready.quant import QuantSpec, fake_quant


def fidelity(ref: torch.Tensor, other: torch.Tensor) -> float:
    """Mean per-sample cosine similarity. Scale-invariant; in [-1, 1]; fidelity(y, y) == 1.0
    (zero-norm rows reproduced exactly count as 1.0, not 0 — a perfectly reproduced dead-ReLU /
    masked output is not a degradation)."""
    a = ref.reshape(ref.shape[0], -1) if ref.ndim > 1 else ref.reshape(1, -1)
    b = other.reshape(other.shape[0], -1) if other.ndim > 1 else other.reshape(1, -1)
    cos = F.cosine_similarity(a, b, dim=1)
    both_zero = (a.norm(dim=1) < 1e-12) & (b.norm(dim=1) < 1e-12)
    cos = torch.where(both_zero, torch.ones_like(cos), cos)
    # A genuinely zero row reproduced exactly is handled above (-> 1.0). Any REMAINING non-finite
    # cosine means a NaN/inf leaked in from a model output — surface it loudly rather than emitting a
    # silent NaN fidelity into a JSON report (invalid JSON) or a misleading metric.
    if not torch.isfinite(cos).all():
        raise ValueError(
            "fidelity: non-finite cosine — a baseline or variant output contains NaN/inf")
    cos = cos.clamp(-1.0, 1.0)   # cosine is mathematically in [-1, 1]; FP can drift to 1.0000003
    return float(cos.mean())


def _eligible(model):
    return [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]


def _apply_to_weights(model, fn) -> None:
    """Apply fn(weight)->weight in-place to every eligible Linear/Conv2d weight (a deterministic
    weight transform, e.g. fake-quant)."""
    with torch.no_grad():
        for m in _eligible(model):
            m.weight.copy_(fn(m.weight))


def _mean_fidelity(baseline, variant, inputs, draws: int, seed: int) -> float:
    """Mean fidelity for a variant whose STOCHASTICITY lives in the forward pass (e.g. an
    instrumented GaussianNoise wrapper that re-samples every call). Deterministic weight transforms
    also route here — every draw is then identical, which is correct (no variance to average)."""
    fids = []
    with torch.random.fork_rng():           # reproducible per draw WITHOUT clobbering the caller's RNG
        for i in range(max(draws, 1)):
            torch.manual_seed(seed + i)
            with torch.no_grad():
                out = variant(inputs)
            fids.append(fidelity(baseline, out))
    return sum(fids) / len(fids)


def _mean_perturbed_fidelity(model, baseline, inputs, transform, *, draws: int, seed: int,
                             params=("weight",)) -> float:
    """Mean fidelity over `draws` seeded draws of a deep copy of `model` whose eligible-module
    params (each name in `params`, e.g. 'weight' and/or 'bias') are RE-PERTURBED per draw by
    transform(pristine_param, seed=seed+i). Re-drawing every draw is essential — baking a STOCHASTIC
    transform in once makes `draws` a silent no-op (every forward identical) that reports one
    lucky/unlucky realization as a variance-reduced mean. Never mutates the caller's model."""
    variant = copy.deepcopy(model).eval()
    origs = []
    for m in _eligible(variant):
        for name in params:
            p = getattr(m, name, None)
            if p is not None:
                origs.append((p, p.detach().clone()))
    fids = []
    for i in range(max(draws, 1)):
        with torch.no_grad():
            for p, p0 in origs:
                p.copy_(transform(p0, seed=seed + i))
            out = variant(inputs)
        fids.append(fidelity(baseline, out))
    return sum(fids) / len(fids)


def _mean_weight_noise_fidelity(model, baseline, inputs, *, prop_sigma: float, read_sigma: float,
                                draws: int, seed: int) -> float:
    """Mean fidelity under weight-programming noise, re-drawn per draw (see _mean_perturbed_fidelity)."""
    return _mean_perturbed_fidelity(
        model, baseline, inputs,
        lambda w0, *, seed: program_noise(w0, prop_sigma=prop_sigma, read_sigma=read_sigma, seed=seed),
        draws=draws, seed=seed, params=("weight",))


def _profile_val(profile, key):
    fld = getattr(profile, "fields", {}).get(key)
    return fld.value if fld is not None else None


def _baseline(model, inputs):
    """The noiseless baseline output of a fresh eval() deep copy (never mutates the caller)."""
    base = copy.deepcopy(model).eval()
    with torch.no_grad():
        return base(inputs)


def _instrumented_fidelity(model, baseline, inputs, transform, *, draws, seed):
    """Mean fidelity of a deep copy of `model` instrumented with an OUTPUT `transform`
    (GaussianNoise, ADCQuant, ...) vs `baseline`."""
    variant = copy.deepcopy(model).eval()
    instrument(variant, transform)
    return _mean_fidelity(baseline, variant, inputs, draws, seed)


def profile_program_noise_fidelity(model, inputs, profile, *, draws: int = 8,
                                   seed: int = 0) -> float | None:
    """Mean fidelity of `model` under the PROFILE's own characterised weight-programming noise
    (prog_noise_sigma proportional term + read_noise_sigma floor), vs the noiseless baseline. This
    is what makes the profile's per-device coefficients actually drive a reported number instead of
    sitting as unused metadata. Returns None if the profile declares no programming-noise
    coefficients (e.g. the photonic MZI profile). The hidden coefficients never leave the tool — only
    this derived fidelity does, and only in the non-redacted local view."""
    prop = _profile_val(profile, "prog_noise_sigma")
    read = _profile_val(profile, "read_noise_sigma")
    if prop is None and read is None:
        return None
    baseline = _baseline(model, inputs)
    return _mean_weight_noise_fidelity(model, baseline, inputs, prop_sigma=float(prop or 0.0),
                                       read_sigma=float(read or 0.0), draws=draws, seed=seed)


def profile_adc_fidelity(model, inputs, profile, *, draws: int = 1, seed: int = 0) -> float | None:
    """Mean fidelity of `model` with every eligible layer's OUTPUT quantized through a simulated ADC
    at the PROFILE's own `enob_avail` effective bits, vs the noiseless baseline. This is what makes
    the profile's ENOB — already priced by the cost model's precision gate in score.py — actually
    drive a reported signal-fidelity number instead of sitting as cost-model metadata alone.
    `adc_quantize` has no RNG, so `draws` only matters if a caller wants parity with the other
    profile-driven fidelity helpers; the result is identical regardless. Returns None if the profile
    declares no `enob_avail` field."""
    bits = _profile_val(profile, "enob_avail")
    if bits is None:
        return None
    baseline = _baseline(model, inputs)
    return _instrumented_fidelity(model, baseline, inputs, ADCQuant(float(bits)),
                                  draws=draws, seed=seed)


def profile_adc_fixed(model, inputs, profile, *, draws: int = 1,
                      seed: int = 0) -> tuple[float | None, float | None]:
    """(fixed_fidelity, saturation_fraction) under the profile's FIXED-range ADC — the realistic
    counterpart to profile_adc_fidelity's best-case auto-ranging (usually worse, but not a hard
    bound: a per-tensor range can occasionally beat a wide per-sample one on multi-axis outputs). The
    fixed ADC covers `adc_full_scale` (a fraction) of each output's per-tensor peak, so peaks
    saturate. Saturation is reported on the clean output tensor. Returns (None, None) if the profile
    declares no `adc_full_scale` or `enob_avail`."""
    frac = _profile_val(profile, "adc_full_scale")
    bits = _profile_val(profile, "enob_avail")
    if frac is None or bits is None:
        return None, None
    baseline = _baseline(model, inputs)
    fixed = _instrumented_fidelity(model, baseline, inputs,
                                   FixedADCQuant(float(bits), float(frac)), draws=draws, seed=seed)
    saturation = adc_saturation_fraction(baseline, float(frac) * float(baseline.abs().max()))
    return fixed, saturation


def degradation_curve(model, inputs, *, param: str, values: list, draws: int = 8,
                      seed: int = 0) -> list:
    """[{"value": v, "fidelity": f}, ...]. param="sigma": instrument with GaussianNoise(v);
    param="weight_bits": fake-quant eligible weights to v bits; param="prog_sigma": apply analog
    weight-programming noise (proportional term at level v, no read floor) to every eligible weight,
    RE-DRAWN per draw; param="adc_bits": instrument with ADCQuant(v), quantizing every eligible
    layer's OUTPUT to v effective ADC bits (deterministic — no draw variance). Fidelity is the mean
    over `draws` seeded draws against the model's own noiseless baseline. Does not mutate the
    caller's model."""
    baseline = _baseline(model, inputs)
    curve = []
    for v in values:
        if param == "sigma":
            fid = _instrumented_fidelity(model, baseline, inputs, GaussianNoise(float(v)),
                                         draws=draws, seed=seed)
        elif param == "weight_bits":
            variant = copy.deepcopy(model).eval()
            _apply_to_weights(variant, lambda w: fake_quant(w, QuantSpec(bits=int(v))))
            fid = _mean_fidelity(baseline, variant, inputs, draws, seed)
        elif param == "prog_sigma":
            fid = _mean_weight_noise_fidelity(model, baseline, inputs, prop_sigma=float(v),
                                              read_sigma=0.0, draws=draws, seed=seed)
        elif param == "adc_bits":
            fid = _instrumented_fidelity(model, baseline, inputs, ADCQuant(int(v)),
                                         draws=draws, seed=seed)
        else:
            raise ValueError(f"unknown sweep param {param!r}")
        curve.append({"value": v, "fidelity": fid})
    return curve
