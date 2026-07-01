"""W6 · F1 — task-accuracy harness. The product's headline is a REAL accuracy number, not a proxy:
top-1 on real labels, clean vs under-noise. `accuracy_under_noise` instruments a deep-copied model
with the W1 GaussianNoise wrapper (so the caller's model is never mutated) and measures mean top-1
over `draws` seeded draws inside `torch.random.fork_rng()` — reproducible, never clobbers the
caller's global RNG. At sigma=0 the wrapper is a no-op, so it reproduces the clean accuracy exactly."""
from __future__ import annotations

import copy
import math

import torch

from analog_ready.instrument.replace import instrument
from analog_ready.noise import GaussianNoise


def _logits(out):
    """The class-logit tensor from a model output. Tolerates the common transformer/HF return
    shapes (a `.logits` attribute, or a tuple/list whose first element is the logits) so accuracy
    works beyond bare-tensor CNNs."""
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def top1_accuracy(model, inputs, labels) -> float:
    """Fraction of `inputs` where the model's class logits argmax == labels. Runs under
    eval()/no_grad and RESTORES the caller's original training mode afterwards, regardless of
    model.training on entry."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            preds = _logits(model(inputs)).argmax(-1)
        return float((preds == labels).float().mean())
    finally:
        model.train(was_training)


def accuracy_under_noise(model, inputs, labels, *, sigma: float, draws: int = 4,
                         seed: int = 0) -> float:
    """Mean top-1 accuracy of a NOISE-INSTRUMENTED deep copy of `model` over `draws` seeded draws.
    The caller's model is never mutated (we instrument copy.deepcopy(model)). At sigma=0 the
    GaussianNoise wrapper is a no-op, so this reproduces top1_accuracy(model, inputs, labels)
    exactly. Deterministic given `seed`; uses torch.random.fork_rng() so the caller's global RNG
    state is left untouched. Raises ValueError on sigma < 0 (a noise magnitude is non-negative)."""
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma must be a finite value >= 0 (got {sigma})")
    noisy_model, _ = instrument(copy.deepcopy(model), GaussianNoise(float(sigma)))
    was_training = noisy_model.training
    noisy_model.eval()
    try:
        accs = []
        with torch.random.fork_rng():
            for i in range(max(draws, 1)):
                torch.manual_seed(seed + i)
                with torch.no_grad():
                    preds = _logits(noisy_model(inputs)).argmax(-1)
                accs.append(float((preds == labels).float().mean()))
        return sum(accs) / len(accs)
    finally:
        noisy_model.train(was_training)


def accuracy_report(model, inputs, labels, *, sigma: float, draws: int = 4,
                    seed: int = 0) -> dict:
    """{"clean", "under_noise", "drop": clean-under_noise, "sigma"}."""
    clean = top1_accuracy(model, inputs, labels)
    under_noise = accuracy_under_noise(model, inputs, labels, sigma=sigma, draws=draws, seed=seed)
    return {"clean": clean, "under_noise": under_noise, "drop": clean - under_noise, "sigma": sigma}
