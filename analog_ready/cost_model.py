"""The system-level cost model — the spine. Energy splits into compute (analog MACs) + ADC/DAC
conversion + DRAM movement (LightCode framing: converters + memory movement, not the analog MACs,
dominate system energy). With LightCode's own coefficients (0.04 pJ analog MAC, 3.17 pJ 8-bit ADC,
10 pJ DAC) the CONVERTER share alone, DERIVED here, ranges from ~1.3x the analog compute at 256-wide
arrays to ~10x at 32-wide (adding DRAM movement: ~2.6x / ~11.7x for the same square shapes) — ratios
we compute from the coefficients per op shape/precision/array size, not figures LightCode reports
directly. Every coefficient ships its
{value, unit, source, uncertainty} so a reviewer can audit each digit.

Modelling assumptions (conservative, stated so a reviewer can check them):
- Conversions assume NO converter sharing between spatial tiles: each tile drives its own input
  DACs (DAC count replicates across column-tiles, n_tiles) and reads its own partial sums (ADC
  count replicates across row-tiles, k_tiles). An optimistic shared-DAC design lower-bounds DAC
  conversions at M*K.
- DRAM is a single-pass lower bound (weights once + activations); it excludes partial-sum
  re-streaming and weight re-loads for non-resident tiles (a v0.1 refinement)."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Coefficient:
    value: float
    unit: str
    source: str
    uncertainty: str

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit,
                "source": self.source, "uncertainty": self.uncertainty}


# provenance for each profile coefficient the estimate consumes: (unit, source, uncertainty).
_PROVENANCE = {
    "mac_energy_pj": ("pJ/MAC", "LightCode arXiv:2509.16443 (analog MAC ~0.04 pJ)", "+/-50%"),
    "adc_energy_pj": ("pJ/conversion", "LightCode arXiv:2509.16443 (8b ADC ~3.17 pJ)", "+/-50%"),
    "dac_energy_pj": ("pJ/conversion", "LightCode arXiv:2509.16443 (DAC ~10 pJ)", "+/-50%"),
    "mem_energy_pj_per_byte": ("pJ/byte", "DRAM movement, LightCode system model", "+/-2x"),
    "digital_mac_energy_pj": ("pJ/MAC", "digital MAC baseline this op would replace", "+/-50%"),
}


@dataclass
class OpCost:
    name: str
    macs: int
    compute_pj: float
    conversion_pj: float
    dram_pj: float
    total_pj: float
    latency_ns: float
    converter_pj_per_mac: float
    coefficients: dict   # name -> Coefficient

    def to_dict(self, *, with_coefficients: bool = True) -> dict:
        d = {
            "name": self.name, "macs": self.macs,
            "compute_pj": self.compute_pj, "conversion_pj": self.conversion_pj,
            "dram_pj": self.dram_pj, "total_pj": self.total_pj,
            "latency_ns": self.latency_ns, "converter_pj_per_mac": self.converter_pj_per_mac,
        }
        if with_coefficients:
            d["coefficients"] = {k: c.to_dict() for k, c in self.coefficients.items()}
        return d


@dataclass
class CostReport:
    ops: list
    total_pj: float


def _coeff(profile, key: str) -> Coefficient:
    unit, source, unc = _PROVENANCE[key]
    return Coefficient(float(profile.fields[key].value), unit, source, unc)


def _val(profile, key: str) -> float:
    return float(profile.fields[key].value)


def dram_bytes(feat, weight_bits: float, input_bits: float, output_bits: float = 8.0) -> float:
    """Single-pass lower bound on bytes moved: weights once + activations. Excludes partial-sum
    re-streaming and weight re-loads for non-resident tiles (a v0.1 refinement). Shared by the cost
    model and the score so the byte accounting can never drift between them.

    `output_bits` defaults to 8.0 — a standard digitised-activation width — and is NOT read from the
    profile (an ENOB below 8 would round up to a byte-aligned store anyway); with all builtin
    profiles at enob_avail <= 8 the default slightly OVERestimates output traffic, i.e. errs
    conservative. Override explicitly for a wider-output design."""
    return (feat.weight_numel * weight_bits + feat.act_in_numel * input_bits
            + feat.act_out_numel * output_bits) / 8.0


_REQUIRED_FIELDS = ("array_rows", "array_cols", "weight_bits", "input_bits",
                    "mac_energy_pj", "adc_energy_pj", "dac_energy_pj", "mem_energy_pj_per_byte",
                    "digital_mac_energy_pj")


def estimate_op(feat, profile) -> OpCost:
    # Fail with a clear, actionable message on a malformed/incomplete profile (e.g. a vendor's
    # custom YAML) instead of a bare KeyError from deep inside the arithmetic.
    missing = [k for k in _REQUIRED_FIELDS if k not in profile.fields]
    if missing:
        raise ValueError(f"profile {getattr(profile, 'name', '?')!r} is missing required "
                         f"cost-model field(s): {', '.join(missing)}")
    rows, cols = int(_val(profile, "array_rows")), int(_val(profile, "array_cols"))
    wbits, ibits = _val(profile, "weight_bits"), _val(profile, "input_bits")
    mac_e, adc_e, dac_e = _val(profile, "mac_energy_pj"), _val(profile, "adc_energy_pj"), _val(profile, "dac_energy_pj")
    mem_e = _val(profile, "mem_energy_pj_per_byte")

    k_tiles = math.ceil(feat.K / rows)
    n_tiles = math.ceil(feat.N / cols)
    # No-sharing tile model (see module docstring): each spatial tile drives its own input DACs, so
    # DAC conversions replicate across column-tiles; each tile reads its own partial sums, so ADC
    # conversions replicate across row-tiles. (Optimistic shared-DAC lower bound would be M*K.)
    dac_ops = feat.M * feat.K * n_tiles
    adc_ops = feat.M * feat.N * k_tiles

    compute_pj = feat.macs * mac_e
    conversion_pj = dac_ops * dac_e + adc_ops * adc_e
    dram_pj = dram_bytes(feat, wbits, ibits) * mem_e
    total_pj = compute_pj + conversion_pj + dram_pj

    # PLACEHOLDER latency (does not enter any verdict): M analog passes across the tiled mesh.
    latency_ns = float(feat.M * k_tiles * n_tiles)
    converter_pj_per_mac = conversion_pj / feat.macs if feat.macs else 0.0
    coeffs = {k: _coeff(profile, k) for k in _PROVENANCE}
    return OpCost(feat.name, feat.macs, compute_pj, conversion_pj, dram_pj, total_pj,
                  latency_ns, converter_pj_per_mac, coeffs)


def estimate(feats, profile) -> CostReport:
    ops = [estimate_op(f, profile) for f in feats]
    return CostReport(ops=ops, total_pj=sum(o.total_pj for o in ops))
