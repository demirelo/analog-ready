"""W9 · F1 — ADC readout quantization. The analog dot-product accumulates on the array in the analog
domain and is read back to digital through an ADC with only a few EFFECTIVE bits (ENOB) — a dominant
precision bottleneck the cost model already prices via the `enob_avail` field + the precision gate
in score.py, but until now nothing actually applied that limit to the signal itself (LightCode,
arXiv:2509.16443 — ADC/readout precision is a dominant analog-inference bottleneck).

`adc_quantize` uniformly quantizes a layer's OUTPUT tensor over its PER-SAMPLE (dim-0 = batch)
dynamic range using a SYMMETRIC mid-tread quantizer with 2**b - 1 levels (codes -qmax..+qmax,
qmax = 2**(b-1) - 1) — the standard signed symmetric design, NOT a full 2**b-code ADC. So "b bits"
delivers log2(2**b - 1) effective bits: a negligible difference at b >= 6, but materially fewer at
low b (b=2 gives 3 levels ~ 1.58 bits) — always in the CONSERVATIVE direction (more degradation
reported, never less). Deterministic (no RNG at all); non-mutating; high bits (e.g. 12) is
~identity; an all-zero row is returned as zeros with no NaN.

OPTIMISM / SIMPLIFICATIONS — be honest, a hardware engineer will check these:
  * This AUTO-RANGES: each sample is quantized against its OWN peak magnitude — a best-case ADC whose
    full-scale reference is retrospectively perfect for every input. A REAL ADC has a FIXED
    full-scale reference, so real fidelity is USUALLY WORSE — often much worse for outlier / high-
    dynamic-range activations (one large value forces the small ones toward 0). This is the typical
    case, NOT a hard bound: a single per-tensor fixed range can occasionally BEAT a per-sample auto
    range on multi-axis outputs (e.g. a transformer block's 3-D output). Treat the auto number as an
    optimistic reference point, not a guaranteed floor on ADC degradation.
  * It quantizes per-sample (per batch row); a crossbar ADC digitizes per output column
    (per-output-channel) partial sums — a secondary granularity mismatch (v0.1 simplification).
  * ENOB is used directly as a clean quantizer bit-count; a measured ENOB already folds in noise, so
    this is a coarse mapping, not a calibrated one."""
from __future__ import annotations

import math

import torch


def adc_quantize(x: torch.Tensor, *, bits: float, full_scale: float | None = None) -> torch.Tensor:
    """Uniform quantization of `x` to `bits` effective bits. `bits` may be a float ENOB — FLOORED to
    an integer level count (conservative: fewer bits => more degradation). Returns a NEW tensor;
    never mutates `x`.

    full_scale=None (W9 default): AUTO-RANGE to each sample's own per-sample peak — the best-case ADC
    (see the module docstring). full_scale=V: a FIXED reference — one range [-V, V] shared across the
    whole tensor; values beyond V SATURATE (the realistic case). Clamping the quantizer CODE to
    ±(qmax) already saturates out-of-range inputs to +-full_scale.

    `bits` must be >= 1 (bits == 1 is a sign comparator: sign(x)*clip, not uniform quantization) and
    `full_scale`, when given, must be > 0. Both raise ValueError otherwise — a physically invalid ADC
    is a loud error, never silently masked to garbage."""
    if not math.isfinite(bits) or bits < 1:
        raise ValueError(f"bits must be a finite value >= 1 (got {bits})")
    if full_scale is not None and (not math.isfinite(float(full_scale)) or float(full_scale) <= 0):
        raise ValueError(f"full_scale must be a finite value > 0 (got {full_scale})")
    bits_i = math.floor(bits)
    if bits_i >= 24:
        return x.clone()  # ~lossless at this magnitude; also avoids a huge 2**(bits-1)

    if full_scale is not None:
        fs = float(full_scale)         # validated > 0 above
        xq = x.clamp(-fs, fs)          # SATURATE beyond the fixed reference (new tensor; x unchanged)
        clip = torch.as_tensor(fs, dtype=x.dtype)   # FIXED range, per-tensor
    else:
        xq = x
        if x.ndim > 1:
            clip = x.abs().amax(dim=tuple(range(1, x.ndim)), keepdim=True)
        else:
            clip = x.abs().amax()      # 1-D output: treat as one sample / single range, not per-element
        clip = clip.clamp_min(1e-12)   # avoid div-by-zero on an all-zero row (x/scale is still 0 there)

    if bits_i <= 1:
        return torch.sign(xq) * clip   # a 1-bit ADC is a sign comparator: 2 levels, distinct from 2-bit
    qmax = 2 ** (bits_i - 1) - 1        # bits>=2 -> qmax>=1, a genuinely distinct level count per bit
    scale = clip / qmax
    # Codes land in [-qmax, qmax] on BOTH paths (|xq| <= clip by construction), i.e. 2**b - 1
    # symmetric levels — the -(qmax+1) code is unreachable except via FP rounding at the negative
    # peak; the clamp is a defensive bound, not a load-bearing code.
    return torch.round(xq / scale).clamp(-(qmax + 1), qmax) * scale


def adc_saturation_fraction(x: torch.Tensor, full_scale: float) -> float:
    """Fraction of `x` whose magnitude exceeds a fixed ADC full-scale (i.e. clips). In [0, 1]."""
    return float((x.abs() > float(full_scale)).float().mean())


class ADCQuant:
    """The output-transform callable matching the `noise.py::GaussianNoise` interface, so
    `instrument(model, ADCQuant(bits))` quantizes every eligible layer's output through a simulated
    ADC of `bits` effective bits."""

    def __init__(self, bits: float):
        if not math.isfinite(bits) or bits < 1:
            raise ValueError(f"bits must be a finite value >= 1 (got {bits})")
        self.bits = bits

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return adc_quantize(x, bits=self.bits)


class FixedADCQuant:
    """A per-tensor FIXED-range ADC output transform (a more realistic counterpart to the best-case
    `ADCQuant`): the full-scale reference covers `full_scale_fraction` of each output tensor's OWN
    peak (per-tensor, not per-sample), so the top of the range saturates when the fraction is < 1.
    This is a per-tensor CLIPPED-ADC stress, NOT an absolute calibration — the full-scale is still
    derived from the signal's own peak, not a measured device reference — but it drops the per-sample
    auto-ranging optimism and models saturation. USUALLY worse than `ADCQuant`, but NOT a hard bound:
    on higher-rank per-sample outputs a tighter per-tensor range can occasionally beat a wide
    per-sample one."""

    def __init__(self, bits: float, full_scale_fraction: float = 0.9):
        if not math.isfinite(bits) or bits < 1:
            raise ValueError(f"bits must be a finite value >= 1 (got {bits})")
        frac = float(full_scale_fraction)
        if not math.isfinite(frac) or not 0.0 < frac <= 1.0:
            raise ValueError(
                f"full_scale_fraction must be a finite value in (0, 1] (got {full_scale_fraction})")
        self.bits = bits
        self.frac = frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        fs = max(self.frac * float(x.abs().max()), 1e-12)
        return adc_quantize(x, bits=self.bits, full_scale=fs)
