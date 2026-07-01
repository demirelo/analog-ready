"""Per-op feature extraction — the model-intrinsic input the cost model and score consume.
We turn each eligible op (nn.Linear / nn.Conv2d) into shape/FLOP/contraction numbers via pure
arithmetic on its weights + a nominal workload (no forward pass, fully deterministic). A Conv2d is
modelled as its im2col GEMM so it shares one cost path with Linear."""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class Workload:
    """A nominal workload shape. M (the number of input vectors) is what amortises converters and
    weight programming, so it is a first-class knob.

    The defaults are deliberate, illustrative middle-of-the-road values (NOT a calibration to any
    specific deployment) — override them to match the real inference shape:
      * tokens=128     — a typical transformer/attention sequence length (rows per Linear pass).
      * batch=1        — single-stream, in-order inference (the converter-hostile case).
      * conv_positions=196 — a 14x14 output feature map, a mid-network CNN spatial size."""
    tokens: int = 128          # sequence/token count for a Linear pass
    batch: int = 1
    conv_positions: int = 196  # output spatial positions for a Conv2d (e.g. 14x14)


@dataclass
class OpFeature:
    name: str
    op_type: str        # "linear" | "conv2d"
    M: int              # input vectors processed (rows)
    K: int              # contraction depth (in_features, or in_ch*kh*kw)
    N: int              # outputs (out_features / out_channels)
    macs: int
    flops: int
    weight_numel: int
    act_in_numel: int
    act_out_numel: int

    @property
    def contraction_dim(self) -> int:
        return self.K

    @property
    def weight_reuse(self) -> float:
        """MACs amortising one weight load = M for a single GEMM pass."""
        return self.macs / self.weight_numel if self.weight_numel else 0.0


def _feat(name: str, op_type: str, M: int, K: int, N: int) -> OpFeature:
    macs = M * K * N
    return OpFeature(name=name or op_type, op_type=op_type, M=M, K=K, N=N,
                     macs=macs, flops=2 * macs, weight_numel=K * N,
                     act_in_numel=M * K, act_out_numel=M * N)


def extract_op_features(model: nn.Module, workload: Workload | None = None) -> list[OpFeature]:
    wl = workload or Workload()
    feats: list[OpFeature] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            feats.append(_feat(name, "linear", wl.tokens * wl.batch, mod.in_features, mod.out_features))
        elif isinstance(mod, nn.Conv2d):
            kh, kw = (mod.kernel_size if isinstance(mod.kernel_size, tuple)
                      else (mod.kernel_size, mod.kernel_size))
            # im2col contraction depth: each output channel contracts over its GROUP's input
            # channels only — PyTorch's weight is (out_ch, in_ch/groups, kh, kw), so omitting
            # /groups overcounts weights and MACs by `groups`x (16x for a depthwise conv).
            # Dilation is deliberately ABSENT here: it spreads the taps spatially but adds no
            # weights and no MACs — receptive field is not contraction depth.
            K = (mod.in_channels // mod.groups) * kh * kw
            feats.append(_feat(name, "conv2d", wl.conv_positions * wl.batch, K, mod.out_channels))
    return feats
