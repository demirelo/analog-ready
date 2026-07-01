"""W4 zoo — a small, pure-PyTorch, network-free collection of stand-in workloads. No torchvision /
timm / transformers downloads; everything is built from nn primitives with a fixed seed so state
dicts are BYTE-IDENTICAL across calls.

All four members are chosen to exercise distinct parts of the analysis + instrumentation pipeline:
  mlp           — Linear stack; exercises the baseline MLP path.
  tiny_resnet   — Conv2d + shortcut + Linear; exercises the conv-replacement walker.
  gpt2_block    — GPT-2-style self-attention block with c_attn / c_proj named Linears so the W1
                  attention-projection name matcher actually fires.
  synthetic_gemm — Single Linear; the reference workload for the validate / sensitivity curve.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- helpers

def _g(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _reset_params(module: nn.Module, seed: int) -> None:
    """Re-initialise every parameter tensor with a fixed seed so builds are deterministic."""
    g = _g(seed)
    for p in module.parameters():
        with torch.no_grad():
            p.data = torch.randn(p.shape, generator=g)


# --------------------------------------------------------------------------- model definitions

class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        _reset_params(self, seed=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class _TinyResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 4)
        _reset_params(self, seed=2)

    def forward(self, x):
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h) + h)   # residual shortcut
        h = self.pool(h).flatten(1)
        return self.fc(h)


class _CausalSelfAttention(nn.Module):
    """A minimal GPT-2-style self-attention with c_attn + c_proj named exactly as GPT-2."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        # Fused QKV — named c_attn so the W1 attention-projection matcher fires.
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)                           # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)
        # reshape to multi-head view
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        scale = self.d_head ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class _GPT2Block(nn.Module):
    """A single GPT-2 transformer block: LayerNorm → Attention → LayerNorm → FFN."""

    def __init__(self, d_model: int = 64, n_head: int = 4, seq: int = 16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        _reset_params(self, seed=3)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _SyntheticGEMM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 256)
        _reset_params(self, seed=4)

    def forward(self, x):
        return self.linear(x)


# --------------------------------------------------------------------------- public API

def _seeded_input(shape: tuple, seed: int) -> torch.Tensor:
    """A DETERMINISTIC, non-degenerate input. randn (not zeros): with zero inputs a layer emits only
    its bias, so a sweep would characterise noise on a constant bias vector — not a representative
    activation distribution. A fixed Generator keeps validate byte-identical; the sigma=0 point is
    still exactly 1.0 (no noise) regardless of input."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(*shape, generator=g)


_REGISTRY: dict[str, tuple] = {
    "mlp":           (_MLP,          lambda: _seeded_input((4, 128), 101)),
    "tiny_resnet":   (_TinyResnet,   lambda: _seeded_input((2, 3, 8, 8), 102)),
    "gpt2_block":    (_GPT2Block,    lambda: _seeded_input((2, 16, 64), 103)),
    "synthetic_gemm": (_SyntheticGEMM, lambda: _seeded_input((8, 256), 104)),
}


def list_models() -> list[str]:
    """Names of all zoo members."""
    return list(_REGISTRY.keys())


def build(name: str) -> nn.Module:
    """Return a freshly constructed, DETERMINISTIC nn.Module. Unknown name raises KeyError."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown zoo model {name!r}; available: {list(_REGISTRY)}")
    cls, _ = _REGISTRY[name]
    return cls()


def example_inputs(name: str) -> torch.Tensor:
    """A batch of DETERMINISTIC seeded-random inputs (see `_seeded_input` — deliberately NOT zeros,
    which would exercise only the bias) whose shape is compatible with build(name).eval()(…)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown zoo model {name!r}; available: {list(_REGISTRY)}")
    _, inp_fn = _REGISTRY[name]
    return inp_fn()
