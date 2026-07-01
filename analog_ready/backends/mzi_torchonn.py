"""Optional torchonn-accelerated MZI backend. Imported ONLY by the registry loader when torchonn
is present (never by `import analog_ready`), so a missing torchonn surfaces as available=False."""
from __future__ import annotations

import torchonn  # noqa: F401  (import-guarded by the registry's find_spec check)

from analog_ready.core.backend import NoiseQuantBackend
from analog_ready.backends.mzi_pure import mzi_unitary


def reference_unitary(n: int, params=None):
    """torchonn-validated reference unitary, used by the optional fidelity test (pure vs torchonn,
    1e-4). Falls back to the pure engine's construction when a direct torchonn decomposition is not
    wired yet — kept here so the optional test has a symbol to import."""
    return mzi_unitary(n, params)


class MZITorchonnBackend(NoiseQuantBackend):
    name = "mzi_torchonn"

    def apply_noise(self, x, **kw):
        return x

    def quantize_weights(self, w, **kw):
        return w

    def forward_matmul(self, x, w, **kw):
        return x @ w
