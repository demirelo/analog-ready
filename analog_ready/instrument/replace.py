"""Module-replacement instrumentation — the architecture-agnostic spine. We recurse over
named_children() and swap nn.Linear / nn.Conv2d (and name-matched attention projections) for noisy
wrappers that COMPOSE the original module (so dtype/device/grad are preserved). This needs no graph
capture, so it covers ResNet/ViT/transformers/LLMs uniformly. A linear-like module that is NOT an
nn.Linear (e.g. GPT-2's fused-QKV Conv1D) is reported as unhandled rather than mis-wrapped."""
from __future__ import annotations

import torch.nn as nn

from analog_ready.noise import GaussianNoise
from analog_ready.instrument.coverage import CoverageReport

# Final-segment names that denote an attention projection across common architectures.
ATTENTION_PROJ_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj",   # Llama / Mistral / Qwen
    "c_attn", "c_proj",                        # GPT-2 (c_attn is fused QKV)
    "query", "key", "value", "dense",          # BERT
}

# Names that signal a FUSED qkv projection (one matrix holding Q,K,V).
_FUSED_NAMES = {"c_attn", "qkv", "qkv_proj", "in_proj"}


class NoisyLinear(nn.Module):
    """Wraps an nn.Linear: runs the real op, then injects noise. `base` is a submodule, so gradients
    flow to the original parameters; at sigma=0 the output is identical to the base."""

    def __init__(self, base: nn.Module, noise=None):
        super().__init__()
        self.base = base
        self.noise = noise or GaussianNoise(0.0)

    def forward(self, x):
        return self.noise(self.base(x))


class NoisyConv2d(nn.Module):
    def __init__(self, base: nn.Module, noise=None):
        super().__init__()
        self.base = base
        self.noise = noise or GaussianNoise(0.0)

    def forward(self, x):
        return self.noise(self.base(x))


def _is_linear_like(module: nn.Module) -> bool:
    """A module that performs a matmul projection but is not an nn.Linear (e.g. GPT-2 Conv1D, which
    exposes a 2-D `weight`). We detect it structurally so the walker can disclose it."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        return False
    w = getattr(module, "weight", None)
    return w is not None and getattr(w, "ndim", 0) == 2


def instrument(model: nn.Module, noise=None):
    """Recursively replace eligible modules in-place; return (model, CoverageReport)."""
    cov = CoverageReport(model=type(model).__name__.removeprefix("_"))
    attn_seen = 0     # attention-named nn.Linear modules encountered
    attn_matched = 0  # ...of those, the ones we wrapped

    def walk(module: nn.Module, prefix: str) -> None:
        nonlocal attn_seen, attn_matched
        for name, child in list(module.named_children()):
            if isinstance(child, (NoisyLinear, NoisyConv2d)):
                continue  # idempotent: never re-wrap an already-instrumented module
            qualified = f"{prefix}.{name}" if prefix else name
            is_attn_name = name in ATTENTION_PROJ_NAMES
            if isinstance(child, nn.Linear):
                cov.modules_total += 1
                cov.linear_replaced += 1
                if is_attn_name:
                    attn_seen += 1
                    attn_matched += 1
                    if name in _FUSED_NAMES and cov.fused_qkv_policy == "none":
                        # a fused projection that happens to be a plain nn.Linear: we wrap it
                        # as a single matrix (no per-Q/K/V split yet — that is v0.1).
                        cov.fused_qkv_policy = "wrapped_as_linear_v0"
                setattr(module, name, NoisyLinear(child, noise))
            elif isinstance(child, nn.Conv2d):
                cov.modules_total += 1
                cov.conv_replaced += 1
                setattr(module, name, NoisyConv2d(child, noise))
            elif is_attn_name and _is_linear_like(child):
                # an attention projection we cannot safely wrap (e.g. fused-QKV Conv1D) — disclose it
                cov.unhandled_linear_like.append(qualified)
                if name in _FUSED_NAMES:
                    cov.fused_qkv_policy = "detected_unhandled"
                # do not recurse into it (it is a leaf projection)
            else:
                walk(child, qualified)

    walk(model, "")
    cov.attention_projection_match_rate = (attn_matched / attn_seen) if attn_seen else 1.0
    cov.coverage_confidence = "high" if not cov.unhandled_linear_like else "medium"
    return model, cov
