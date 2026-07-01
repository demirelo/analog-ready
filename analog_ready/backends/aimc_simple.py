"""Pure-PyTorch AIMC (analog in-memory compute) fallback — the DEFAULT analog backend, so the
product path runs without aihwkit. Models the three first-order non-idealities: uniform weight
quantization, additive Gaussian noise, and a power-law conductance drift (identity at t=0)."""
from __future__ import annotations

import torch

from analog_ready.core.backend import NoiseQuantBackend


class AIMCSimpleBackend(NoiseQuantBackend):
    name = "aimc_simple"

    def quantize_weights(self, w, bits: int = 8, **kw):
        levels = 2 ** int(bits)
        lo, hi = w.min(), w.max()
        if levels <= 1 or hi <= lo:
            return w.clone()
        step = (hi - lo) / (levels - 1)
        return torch.round((w - lo) / step) * step + lo

    def apply_noise(self, x, sigma: float = 0.05, **kw):
        if sigma == 0.0:
            return x
        return x + sigma * torch.randn_like(x)

    def drift(self, w, t: float = 0.0, exponent: float = 0.1, **kw):
        """Power-law conductance drift; returns w unchanged at t == 0."""
        return w * (1.0 + float(t)) ** (-float(exponent))

    def forward_matmul(self, x, w, **kw):
        return x @ w
