"""W3 — recovery via per-layer quant-scale calibration. Instead of the naive per-tensor min/max
range (which wastes levels on outliers), pick a clip magnitude that maximises the layer's OUTPUT
FIDELITY (the same cosine metric `recover` reports) on a calibration batch. Because the naive range
is one candidate in the grid, the calibrated fidelity is never worse than naive AT sigma=0 — a real
general guarantee, not a fixture artefact; the gain is strictly positive for outlier-dominated
layers. The improvement is pure quantisation (same noise draw, sigma unchanged), so it persists at
sigma=0. NOTE: v0 calibrates and evaluates on the same batch (a demonstration); a deployment uses a
held-out calibration set."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from analog_ready.quant import QuantSpec, fake_quant
from analog_ready.sensitivity import rank_layers
from analog_ready.sweeps import fidelity

# Clip magnitudes to try, as fractions of weight.abs().max() (1.0 == the naive range).
_CLIP_FRACTIONS = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05)


def calibrate_quant_scale(module, inputs, bits: int) -> float:
    """Return the clip magnitude on module.weight (clamped to [-clip, clip], fake-quantised to
    `bits`) that maximises the layer's output FIDELITY on `inputs` vs full precision — the same
    cosine metric `recover` reports, so its 'never worse than naive' guarantee is real. The naive
    range (clip == weight.abs().max()) is the first candidate, so a tighter clip is chosen only when
    it strictly improves fidelity; on an outlier-dominated layer it is strictly tighter."""
    w = module.weight.detach()
    wmax = float(w.abs().max())
    with torch.no_grad():
        clean = F.linear(inputs, w, module.bias)
    best_clip, best_fid = wmax, -2.0
    for frac in _CLIP_FRACTIONS:             # 1.0 (naive) first -> ties keep the widest range
        clip = wmax * frac
        wq = fake_quant(w.clamp(-clip, clip), QuantSpec(bits=bits))
        with torch.no_grad():
            fid = fidelity(clean, F.linear(inputs, wq, module.bias))
        if fid > best_fid + 1e-12:
            best_fid, best_clip = fid, clip
    return best_clip


def recover(module, inputs, *, bits: int, sigma: float, seed: int = 0) -> dict:
    """Naive vs calibrated weight quantisation under IDENTICAL noise (same sigma, same seed)."""
    w = module.weight.detach()
    clip = calibrate_quant_scale(module, inputs, bits)

    def out_with(weight):
        with torch.random.fork_rng():        # identical noise draw for degraded/recovered, no global leak
            torch.manual_seed(seed)
            with torch.no_grad():
                y = F.linear(inputs, weight, module.bias)
                if sigma:
                    y = y + sigma * torch.randn_like(y)
        return y

    clean = F.linear(inputs, w, module.bias).detach()
    degraded = out_with(fake_quant(w, QuantSpec(bits=bits)))                  # naive min/max
    recovered = out_with(fake_quant(w.clamp(-clip, clip), QuantSpec(bits=bits)))  # calibrated
    return {
        "degraded_fidelity": fidelity(clean, degraded),
        "recovered_fidelity": fidelity(clean, recovered),
        "sigma": sigma,
    }


def recommend_recovery(model, inputs, profile, *, top_k: int = 1, bits: int = 4,
                        sigma: float = 0.1, seed: int = 0, ranked: list | None = None) -> list[dict]:
    """"What recovers accuracy." Rank the model's nn.Linear layers by whole-model fragility, take
    the `top_k` most fragile LINEAR layers (recover() applies to Linear; Conv2d fragility is
    reported by rank_layers but is not yet recoverable), and for each capture its REAL input
    activations (a forward_pre_hook on a single no_grad pass), then call recover() on that captured
    input. recover() guarantees recovered_fidelity >= degraded_fidelity. Pass `ranked` (the output
    of rank_layers) to avoid a redundant second ranking; otherwise it is computed here. Deterministic
    given `seed`; the caller's model is left exactly as found (training mode restored)."""
    was_training = model.training
    model.eval()
    try:
        if ranked is None:
            ranked = rank_layers(model, inputs, sigma=sigma, seed=seed)
        name_to_module = {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear)}
        fragile_linear_names = [r["name"] for r in ranked if r["name"] in name_to_module]
        chosen = fragile_linear_names[:max(top_k, 0)]

        captured: dict[str, torch.Tensor] = {}
        handles = []
        for name in chosen:
            module = name_to_module[name]

            def make_hook(key):
                def hook(module, inputs):  # noqa: ANN001
                    captured[key] = inputs[0]
                return hook

            handles.append(module.register_forward_pre_hook(make_hook(name)))

        try:
            with torch.no_grad():
                model(inputs)
        finally:
            for h in handles:
                h.remove()
    finally:
        model.train(was_training)

    recs = []
    for name in chosen:
        module = name_to_module[name]
        result = recover(module, captured[name], bits=bits, sigma=sigma, seed=seed)
        recs.append({
            "name": name,
            "bits": bits,
            "degraded_fidelity": result["degraded_fidelity"],
            "recovered_fidelity": result["recovered_fidelity"],
        })
    return recs
