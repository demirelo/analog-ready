"""Optional aihwkit-backed AIMC backend. Imported ONLY by the registry loader when aihwkit is
present, so a missing aihwkit surfaces as available=False (it needs torch>=2.9.1 — Image B)."""
from __future__ import annotations

from aihwkit.simulator.configs import InferenceRPUConfig  # noqa: F401

from analog_ready.core.backend import NoiseQuantBackend


class AIMCAihwkitBackend(NoiseQuantBackend):
    name = "aimc_aihwkit"

    def __init__(self):
        self.rpu_config = InferenceRPUConfig()

    def apply_noise(self, x, **kw):
        return x

    def quantize_weights(self, w, **kw):
        return w

    def forward_matmul(self, x, w, **kw):
        return x @ w
