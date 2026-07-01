"""W5 · F1 — per-layer sensitivity. "Which layers break first." For each eligible module
(nn.Linear / nn.Conv2d), inject noise on THAT module's output ONLY — scaled to the layer's own
output magnitude (a RELATIVE perturbation: `sigma * output.std()`), so the ranking reflects true
sensitivity rather than output scale — and measure whole-model output fidelity vs the noiseless
baseline. Layers are returned most-fragile-first (ascending fidelity). Deterministic given `seed`
via `torch.random.fork_rng()`; the caller's model is left exactly as found (training mode restored,
no hooks left behind)."""
from __future__ import annotations

import torch
import torch.nn as nn

from analog_ready.sweeps import fidelity

_ELIGIBLE = (nn.Linear, nn.Conv2d)


def _relative_noise_hook(sigma: float):
    """Add noise scaled to the layer's output RMS, so `sigma` is a fraction of the signal — a
    magnitude-fair perturbation. A constant (zero-variance) output gets no noise."""
    def hook(module, inputs, output):  # noqa: ANN001
        scale = output.detach().std()
        return output + sigma * scale * torch.randn_like(output)
    return hook


def rank_layers(model: nn.Module, inputs, *, sigma: float = 0.1, draws: int = 4,
                seed: int = 0) -> list[dict]:
    """[{"name", "fidelity", "fidelity_drop"}, ...] sorted ascending by fidelity (most fragile
    first). Each eligible module (nn.Linear/nn.Conv2d via named_modules()) is perturbed ONE AT A
    TIME with output-relative noise; every other module is left exact. `fidelity_drop` is
    `max(0, 1 - fidelity)`. Does not mutate the caller's model (training mode restored, every hook
    removed) and never clobbers the global RNG (fork_rng)."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            baseline = model(inputs)

        targets = [(name, module) for name, module in model.named_modules()
                   if isinstance(module, _ELIGIBLE)]

        results = []
        for i, (name, module) in enumerate(targets):
            handle = module.register_forward_hook(_relative_noise_hook(sigma))
            try:
                fids = []
                with torch.random.fork_rng():        # reproducible, never clobbers the caller's RNG
                    for d in range(max(draws, 1)):
                        torch.manual_seed(seed + i * 1009 + d)   # decorrelate (layer, draw)
                        with torch.no_grad():
                            noised = model(inputs)
                        fids.append(fidelity(baseline, noised))
                fid = sum(fids) / len(fids)
            finally:
                handle.remove()
            results.append({"name": name, "fidelity": fid, "fidelity_drop": max(0.0, 1.0 - fid)})

        results.sort(key=lambda r: r["fidelity"])
        return results
    finally:
        model.train(was_training)
