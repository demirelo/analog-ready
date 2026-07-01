"""Lightweight torch-only mock models for the acceptance oracle — no torchvision/transformers
dependency, so the core gate runs with just torch installed."""
import torch
import torch.nn as nn


class Conv1DLike(nn.Module):
    """A GPT-2-style fused-QKV projection: holds weight as (in, 3*in) like HF transformers' Conv1D.
    It is linear-like but NOT an nn.Linear — the walker must NOT wrap it, and must report it as
    unhandled_linear_like."""

    def __init__(self, n_in: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_in, 3 * n_in))
        self.bias = nn.Parameter(torch.zeros(3 * n_in))

    def forward(self, x):
        return x @ self.weight + self.bias


class LlamaStyleAttention(nn.Module):
    """Separate q/k/v/o projections by name — all nn.Linear (should all be wrapped)."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class BertStyleAttention(nn.Module):
    """BERT-style names: query/key/value + a dense output projection — all nn.Linear."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dense = nn.Linear(dim, dim)


class GPT2StyleBlock(nn.Module):
    """A transformer block with a FUSED-QKV c_attn (Conv1DLike, NOT nn.Linear) + an nn.Linear
    c_proj output projection + an MLP."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.c_attn = Conv1DLike(dim)          # fused QKV, linear-like but not nn.Linear
        self.c_proj = nn.Linear(dim, dim)      # output projection, a real nn.Linear
        self.mlp_fc = nn.Linear(dim, 4 * dim)
        self.mlp_proj = nn.Linear(4 * dim, dim)


class TinyConvNet(nn.Module):
    """A small CNN to exercise nn.Conv2d replacement and nested traversal (a ResNet-18 stand-in)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.block = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
        )
        self.head = nn.Linear(8, 10)


def count_modules(model: nn.Module, types) -> int:
    return sum(1 for m in model.modules() if isinstance(m, types))
