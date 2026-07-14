"""W7 · F1 — faithful analog weight-programming noise.

Replaces the W1 arbitrary additive-OUTPUT sigma (`noise.py::GaussianNoise`) with the standard
analog-inference WEIGHT-noise decomposition used across the PIM/AIMC literature and toolkits
(e.g. IBM aihwkit's `RPUConfig` device models; Joshi et al. 2020, arXiv:1906.03138; Le Gallo et
al. 2023, arXiv:2212.02872):

    programmed_weight = weight + prop_sigma * |weight| * eps1 + read_sigma * eps2,   eps ~ N(0, 1)

- `prop_sigma` — PROPORTIONAL (conductance-dependent) programming noise: analog memory devices
  (PCM/ReRAM/SRAM cells) hit a target conductance with an error that scales with the target itself,
  so a large-magnitude weight is perturbed more (in absolute terms) than a near-zero one.
- `read_sigma` — a magnitude-INDEPENDENT programming/read floor (write granularity + a thermal /
  quantisation floor).

SIMPLIFICATIONS — be honest about what this does NOT model, so a hardware engineer isn't misled:
  * `read_sigma` here is a STATIC per-run weight perturbation. True readout noise is re-sampled on
    every MVM (per inference) on the OUTPUT; folding it once into the weight is a first-order
    approximation, not a per-inference readout model.
  * Conductance DRIFT (the dominant temporal effect in PCM/ReRAM: a systematic, non-zero-mean
    power-law decay G(t) = G0*(t/t0)^-nu) is NOT modeled — there is no time argument here. Drift is
    a separate, later rung; do not read this as capturing it.
  * Also NOT modeled: 1/f noise, cycle-to-cycle vs device-to-device variability, asymmetric
    conductance response, and any device-specific calibration.

`prop_sigma`/`read_sigma` are literature-plausible DEFAULTS, not measured silicon (see the report's
"Limits of this estimate" section). A vendor with characterised hardware should override them.
"""
from __future__ import annotations

import math

import torch


def program_noise(weight: torch.Tensor, *, prop_sigma: float, read_sigma: float,
                  seed: int = 0) -> torch.Tensor:
    """Return a NEW tensor: weight perturbed by proportional programming noise (prop_sigma) plus a
    magnitude-independent floor (read_sigma). Never mutates `weight`. Deterministic given `seed`;
    uses `torch.random.fork_rng()` so it does not advance/clobber the caller's global torch RNG
    (same hygiene as `sweeps.py`/`recovery.py`). Identity when both sigmas are 0.

    Raises ValueError on a negative or non-finite sigma (a noise LEVEL must be finite and >= 0) and
    on a non-floating-point weight (Gaussian perturbation is undefined for integer/quantised
    storage)."""
    if (not math.isfinite(prop_sigma) or not math.isfinite(read_sigma)
            or prop_sigma < 0.0 or read_sigma < 0.0):
        raise ValueError(f"noise sigmas must be >= 0 (got prop_sigma={prop_sigma}, "
                         f"read_sigma={read_sigma})")
    if not weight.is_floating_point():
        raise ValueError(f"program_noise needs a floating-point weight, got {weight.dtype}")
    if prop_sigma == 0.0 and read_sigma == 0.0:
        return weight.clone()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        out = weight.clone()
        if prop_sigma:
            out = out + prop_sigma * weight.abs() * torch.randn_like(weight)
        if read_sigma:
            out = out + read_sigma * torch.randn_like(weight)
        return out
