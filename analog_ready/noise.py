"""Pluggable noise callables injected by the instrumentation wrappers. At sigma=0 the noise is a
no-op, so a sigma=0 wrapper reproduces the base module exactly."""
from __future__ import annotations

import math

import torch


class GaussianNoise:
    """Additive i.i.d. Gaussian noise. A placeholder/default — the credible signal-dependent,
    transfer-matrix-aware models live in the backends; this is the wrapper-level hook."""

    def __init__(self, sigma: float = 0.0):
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError(f"sigma must be a finite value >= 0 (got {sigma})")
        self.sigma = float(sigma)

    def __call__(self, x):
        if self.sigma == 0.0:
            return x
        return x + self.sigma * torch.randn_like(x)
