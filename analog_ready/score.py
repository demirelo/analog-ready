"""The analog-friendliness score, DERIVED from the cost-model features (not a standalone rubric):
a hard ENOB precision gate x a weighted geometric mean of soft features (roofline intensity,
converter amortisation, weight reuse, array tiling). It emits the full feature table + the dominant
limiter, never a bare 'score = 82'. Conservative mode only in W2; an `empirical` mode (calibrated to
vendor data) is v0.1."""
from __future__ import annotations

import math
from dataclasses import dataclass

from analog_ready.cost_model import estimate_op, dram_bytes
from analog_ready.breakeven import enob_required, reuse_threshold

# weighted geometric mean — one near-zero feature drags the whole score down (matches the physics).
_WEIGHTS = {"intensity": 0.30, "amortization": 0.30, "reuse": 0.25, "tiling": 0.15}


@dataclass
class Gate:
    name: str
    status: str          # "pass" | "fail"
    confidence: str      # e.g. "heuristic_high"
    rule: str
    requires_empirical_validation: bool

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "confidence": self.confidence,
                "rule": self.rule, "requires_empirical_validation": self.requires_empirical_validation}


@dataclass
class OpScore:
    name: str
    favorability: float
    features: dict        # raw soft features in [0,1]
    contributions: dict   # weighted ln contributions (auditable)
    dominant_limiter: str
    gates: list

    def to_dict(self) -> dict:
        return {"name": self.name, "favorability": self.favorability,
                "features": self.features, "contributions": self.contributions,
                "dominant_limiter": self.dominant_limiter,
                "gates": [g.to_dict() for g in self.gates]}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _val(profile, key: str) -> float:
    return float(profile.fields[key].value)


def score_op(feat, profile) -> OpScore:
    rows, cols = int(_val(profile, "array_rows")), int(_val(profile, "array_cols"))
    ibits, wbits = _val(profile, "input_bits"), _val(profile, "weight_bits")
    enob_avail = _val(profile, "enob_avail")
    peak_flops, peak_bw = _val(profile, "peak_flops"), _val(profile, "peak_bw_bytes")
    digital_mac = _val(profile, "digital_mac_energy_pj")
    cost = estimate_op(feat, profile)

    # roofline arithmetic intensity vs the ridge point (memory-bound ops can't use the analog MAC)
    bytes_moved = dram_bytes(feat, wbits, ibits)
    ai = feat.flops / bytes_moved if bytes_moved else 0.0
    ridge = peak_flops / peak_bw if peak_bw else 1.0
    f_intensity = _clamp01(ai / ridge)
    # converter amortisation: how far below the digital MAC the converter cost sits
    f_amort = _clamp01(digital_mac / cost.converter_pj_per_mac) if cost.converter_pj_per_mac > 0 else 1.0
    # weight reuse vs the (profile-derived) amortisation threshold Y; saturates at 2*Y
    y = reuse_threshold(profile)
    f_reuse = _clamp01(feat.weight_reuse / (2 * y)) if y else 1.0
    # array utilisation (partial tiles waste the crossbar)
    f_tiling = _clamp01((min(feat.K, rows) * min(feat.N, cols)) / (rows * cols))
    features = {"intensity": f_intensity, "amortization": f_amort, "reuse": f_reuse, "tiling": f_tiling}

    enob_req = enob_required(feat, profile)
    precision_pass = enob_req < enob_avail
    gates = [
        Gate("precision", "pass" if precision_pass else "fail", "heuristic_high",
             "ENOB_req = input_bits + 0.5*log2(K) < enob_avail", True),
        Gate("intensity", "pass" if f_intensity > 0 else "fail", "heuristic_medium",
             "arithmetic intensity vs roofline ridge (peak_flops/peak_bw)", True),
    ]

    if feat.macs == 0:  # a no-op does no compute — it cannot be analog-favorable
        return OpScore(feat.name, 0.0, features, {}, "no-op (zero MACs)", gates)

    if not precision_pass:  # hard veto: the op fails the conservative precision envelope
        limiter = f"precision (ENOB_req {enob_req:.1f} >= enob_avail {enob_avail:.1f})"
        return OpScore(feat.name, 0.0, features, {}, limiter, gates)

    contributions, log_sum = {}, 0.0
    for key, w in _WEIGHTS.items():
        contributions[key] = w * math.log(max(features[key], 1e-6))
        log_sum += contributions[key]
    favorability = _clamp01(math.exp(log_sum))
    dominant = min(contributions, key=lambda k: contributions[k])  # most-negative log term
    return OpScore(feat.name, favorability, features, contributions, dominant, gates)
