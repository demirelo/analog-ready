"""QuantSpec + a pure-PyTorch fake-quant default. Quantization is a cross-cutting policy (not a
backend), so it works with neither Brevitas nor torchao installed."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantSpec:
    bits: int = 8
    scheme: str = "uniform"


def fake_quant(x: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """Uniform fake-quantization to at most 2**bits levels over [min, max]. High bit-width is
    near-identity; low bit-width coarsens to a bounded number of levels."""
    levels = 2 ** int(spec.bits)
    lo = x.min()
    hi = x.max()
    if levels <= 1 or hi <= lo:
        return x.clone()
    step = (hi - lo) / (levels - 1)
    return torch.round((x - lo) / step) * step + lo
