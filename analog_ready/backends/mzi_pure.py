"""Pure-PyTorch Clements/Reck MZI engine — the DEFAULT photonic backend (zero third-party deps),
so the demo always runs even if torchonn fails to install. Builds an N x N unitary (any N,
including non-power-of-2) as a product of 2x2 MZI rotations embedded in a rectangular mesh."""
from __future__ import annotations

import math

import torch

from analog_ready.core.backend import NoiseQuantBackend


def _mzi_2x2(theta: float, phi: float) -> torch.Tensor:
    """A 2x2 MZI transfer matrix: a real rotation composed with an input phase shifter. Unitary by
    construction (product of two unitaries)."""
    c, s = math.cos(theta), math.sin(theta)
    e = complex(math.cos(phi), math.sin(phi))
    return torch.tensor([[c * e, -s], [s * e, c]], dtype=torch.cfloat)


def mzi_unitary(n: int, params=None) -> torch.Tensor:
    """An n x n unitary from a Clements-style rectangular mesh of n(n-1)/2 MZI blocks."""
    if n < 1:
        raise ValueError("n must be >= 1")
    nblocks = n * (n - 1) // 2
    if params is None:
        angles = torch.rand(max(nblocks, 1), 2) * (2 * math.pi)
    else:
        angles = params if torch.is_tensor(params) else torch.tensor(params, dtype=torch.float)
        angles = angles.reshape(-1, 2)
        if angles.shape[0] < nblocks:
            raise ValueError(
                f"need {nblocks} angle rows for n={n}, got {angles.shape[0]}")

    u = torch.eye(n, dtype=torch.cfloat)
    b = 0
    for layer in range(n):
        i = layer % 2
        while i + 1 < n and b < nblocks:
            theta = float(angles[b, 0])
            phi = float(angles[b, 1])
            b += 1
            t2 = _mzi_2x2(theta, phi)
            t = torch.eye(n, dtype=torch.cfloat)
            t[i, i], t[i, i + 1] = t2[0, 0], t2[0, 1]
            t[i + 1, i], t[i + 1, i + 1] = t2[1, 0], t2[1, 1]
            u = t @ u
            i += 2
        if b >= nblocks:
            break
    return u


class MZIPureBackend(NoiseQuantBackend):
    name = "mzi_pure"

    def apply_noise(self, x, sigma: float = 0.01, **kw):
        if sigma == 0.0:
            return x
        return x + sigma * torch.randn_like(x)

    def quantize_weights(self, w, bits: int = 8, **kw):
        levels = 2 ** int(bits)
        lo, hi = w.min(), w.max()
        if levels <= 1 or hi <= lo:
            return w.clone()
        step = (hi - lo) / (levels - 1)
        return torch.round((w - lo) / step) * step + lo

    def forward_matmul(self, x, w, **kw):
        return x @ w
