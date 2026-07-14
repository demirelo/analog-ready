"""W8 · F1/F2 — conductance drift over time.

PCM/ReRAM analog memory cells do not hold a programmed conductance forever: the conductance decays
as a power law over the elapsed time since programming,

    G(t) = G0 * (t / t0) ** (-nu)

(Joshi et al. 2020, arXiv:1906.03138; Le Gallo et al. 2023, arXiv:2212.02872; IBM aihwkit's
`PCMLikeNoiseModel`, which uses the same power-law form). `nu` is the drift EXPONENT — for PCM a
broadly published materials constant, roughly 0.05-0.1.

THE KEY PHYSICS: a UNIFORM drift (identical nu for every programmed value) scales the whole layer —
weights AND the analog bias — by one global factor. To first order that is a pure scale: cosine
fidelity is scale-invariant, and it is exactly what an analog system's periodic drift-compensation /
re-calibration step corrects for (we drift the bias too, `params=('weight','bias')`, so this holds
in the model and a uniform drift is close to invisible). It is not, by itself, the accuracy-relevant
failure mode.

What actually degrades accuracy is DEVICE-TO-DEVICE VARIABILITY in the drift exponent: real cells
across a crossbar do not share one exact nu — each drifts at its own rate, `nu_i = nu + sigma_nu *
eps_i`, `eps_i ~ N(0, 1)`. That per-device spread turns a compensatable global rescale into a
non-uniform perturbation recalibration cannot fully undo. `sigma_nu` is therefore what drives the
retention curve down over time — not `nu` itself.

SIMPLIFICATIONS — be honest about what this does NOT model: a LITERATURE-DEFAULT power-law drift
model (same functional form as aihwkit's `PCMLikeNoiseModel`), not measured silicon or a per-device
characterised drift table. It does not itself model the recalibration step it refers to, cycling
effects, or read-noise-on-top-of-drift (see `device_noise.py` for the separate programming-noise
model); the two are not composed here. Bias is drifted on the assumption it is stored as an analog
quantity; a design with a digital bias would drift only the weights.

SIGNED-WEIGHT vs DIFFERENTIAL-PAIR: real PCM crossbars (HERMES, Joshi) encode a signed weight as a
CONDUCTANCE PAIR, w = G+ - G-, and each conductance drifts with its OWN exponent. This model drifts
the SIGNED weight directly — so a near-zero weight (G+ ~ G-, both large) stays near zero here, while
on real silicon its pair's unequal nu_i would drift it AWAY from zero. The signed-weight
approximation therefore UNDERSTATES drift-induced error for small-magnitude weights; a differential-
pair drift model is a future refinement.
"""
from __future__ import annotations

import copy
import math

import torch

from analog_ready.sweeps import _mean_perturbed_fidelity


def drift_weight(weight: torch.Tensor, *, nu: float, t_s: float, t0: float = 1.0,
                 sigma_nu: float = 0.0, seed: int = 0) -> torch.Tensor:
    """Return a NEW tensor: `weight` after power-law conductance drift from `t0` to `t_s`.

    Per-weight exponent `nu_i = clamp(nu + sigma_nu * eps_i, min=0)`, `eps_i ~ N(0, 1)` shaped like
    `weight`, drawn via `torch.random.fork_rng()` + `torch.manual_seed(seed)` so this call does NOT
    advance/clobber the caller's global torch RNG. `nu_i` is clamped to >= 0 because a real drift
    exponent is non-negative — a device relaxes toward (does not grow away from) its drift track, so
    conductance magnitude only decays with time, never grows.

    factor_i = (t_s / t0) ** (-nu_i), applied elementwise: out = weight * factor.

    At t_s == t0 the factor is 1 for every weight (no time elapsed — identity). With sigma_nu == 0
    every weight shares the same factor. Never mutates `weight`; deterministic given `seed`.
    Raises ValueError for non-finite or negative drift parameters, t0 <= 0, or non-positive elapsed
    time t_s. The power-law model is only defined for positive time ratios."""
    if (not math.isfinite(float(t0)) or t0 <= 0):
        raise ValueError(f"t0 (reference time) must be finite and > 0 (got {t0})")
    if (not math.isfinite(float(t_s)) or t_s <= 0):
        raise ValueError(f"t_s (elapsed time) must be finite and > 0 (got {t_s})")
    if (not math.isfinite(float(nu)) or nu < 0):
        raise ValueError(f"nu (drift exponent) must be finite and >= 0 (got {nu})")
    if (not math.isfinite(float(sigma_nu)) or sigma_nu < 0):
        raise ValueError(f"sigma_nu must be finite and >= 0 (got {sigma_nu})")
    if t_s == t0:
        return weight.clone()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        eps = (torch.randn(weight.shape, dtype=weight.dtype, device=weight.device)
               if sigma_nu else None)
    if sigma_nu:
        nu_i = nu + sigma_nu * eps
    else:
        nu_i = torch.full(weight.shape, float(nu), dtype=weight.dtype, device=weight.device)
    nu_i = nu_i.clamp(min=0.0)   # non-negative drift exponent: conductance decays, never grows
    ratio = torch.as_tensor(t_s / t0, dtype=weight.dtype, device=weight.device)
    factor = ratio ** (-nu_i)
    return weight * factor


def retention_curve(model, inputs, *, nu: float, sigma_nu: float, times: list, t0: float = 1.0,
                    draws: int = 8, seed: int = 0) -> list:
    """[{"t": t, "fidelity": f}, ...] — mean output fidelity (over `draws` seeded draws) of a
    drifted copy of `model` vs the noiseless baseline, for each elapsed time in `times`. Drifts both
    the weight and the (analog) bias of eligible Linear/Conv2d modules. The nu spread is RE-DRAWN
    every draw (seed + i) — baking it in once would make `draws` a no-op. At t == t0, fidelity is
    ~1.0; it decays for larger t as device-to-device nu spread accumulates. Deterministic given
    `seed`; never mutates the caller's model."""
    base_model = copy.deepcopy(model).eval()
    with torch.no_grad():
        baseline = base_model(inputs)
    curve = []
    for t in times:
        fid = _mean_perturbed_fidelity(
            model, baseline, inputs,
            lambda p0, *, seed, t=t: drift_weight(p0, nu=nu, t_s=t, t0=t0, sigma_nu=sigma_nu,
                                                  seed=seed),
            draws=draws, seed=seed, params=("weight", "bias"))
        curve.append({"t": t, "fidelity": fid})
    return curve
