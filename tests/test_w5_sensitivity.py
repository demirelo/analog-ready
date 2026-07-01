"""W5 · F1 — per-layer sensitivity. The product promises "which layers break first"; this is the
ranked answer. `rank_layers(model, inputs, *, sigma, seed)` perturbs ONE eligible layer at a time
(GaussianNoise on its output) and measures the whole-model output fidelity vs the noiseless run,
returning the layers ordered most-fragile-first. Deterministic. analyze() surfaces it when given
inputs; without inputs it stays backward-compatible (no per-layer block)."""
from __future__ import annotations

import torch
import torch.nn as nn

from analog_ready.zoo import build, example_inputs
from analog_ready.profiles import load_profile


def _two_path():
    """A model where `fragile` drives the output and `robust` is scaled to ~0, so noise on `fragile`
    hurts ~1000x more — a deterministic, unambiguous fragility ordering."""
    class TwoPath(nn.Module):
        def __init__(self):
            super().__init__()
            self.fragile = nn.Linear(8, 8)
            self.robust = nn.Linear(8, 8)

        def forward(self, x):
            return self.fragile(x) + 1e-3 * self.robust(x)

    torch.manual_seed(0)
    return TwoPath()


def test_rank_layers_orders_most_fragile_first():
    from analog_ready.sensitivity import rank_layers
    torch.manual_seed(0)
    m, x = _two_path(), torch.randn(4, 8)
    ranked = rank_layers(m, x, sigma=1.0, seed=0)
    names = [r["name"] for r in ranked]
    assert set(names) == {"fragile", "robust"}
    assert names[0] == "fragile"                       # most fragile first
    assert ranked[0]["fidelity"] <= ranked[1]["fidelity"]


def test_rank_layers_is_deterministic():
    from analog_ready.sensitivity import rank_layers
    torch.manual_seed(0)
    m, x = _two_path(), torch.randn(4, 8)
    assert rank_layers(m, x, sigma=0.5, seed=0) == rank_layers(m, x, sigma=0.5, seed=0)


def test_rank_layers_fidelity_in_range_and_sorted():
    from analog_ready.sensitivity import rank_layers
    m, x = build("mlp"), example_inputs("mlp")
    ranked = rank_layers(m, x, sigma=0.3, seed=0)
    assert len(ranked) >= 2
    for r in ranked:
        assert -1.0 - 1e-6 <= r["fidelity"] <= 1.0 + 1e-6
        assert abs(r["fidelity_drop"] - (1.0 - r["fidelity"])) < 1e-6
    fids = [r["fidelity"] for r in ranked]
    assert fids == sorted(fids)                         # most-fragile-first => non-decreasing fidelity


def test_rank_layers_no_eligible_layers_is_empty():
    from analog_ready.sensitivity import rank_layers
    assert rank_layers(nn.Sequential(nn.ReLU()), torch.randn(2, 4), sigma=0.1, seed=0) == []


def test_analyze_surfaces_layer_sensitivity_only_when_inputs_given():
    from analog_ready.analyze import analyze
    m, x, p = build("mlp"), example_inputs("mlp"), load_profile("aimc_pcm_4bit")
    d = analyze(m, p, inputs=x).to_dict()
    assert "layers" in d and len(d["layers"]) >= 2
    assert d["layers"][0]["name"]                       # ranked, named
    assert analyze(m, p).to_dict().get("layers", []) == []   # backward-compatible without inputs
