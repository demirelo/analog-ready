"""analyze() — the W2 report surface. It joins per-op features + cost + score + break-even, runs a
default ENOB sensitivity sweep (so every report shows WHERE the verdict flips), and exposes a
redacted view that carries the qualitative verdicts but never the raw cost coefficients or a
`hidden` profile field."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from analog_ready.features import extract_op_features, Workload
from analog_ready.cost_model import estimate_op
from analog_ready.score import score_op
from analog_ready.breakeven import break_even, enob_required, _with_field
from analog_ready.sensitivity import rank_layers
from analog_ready.recovery import recommend_recovery
from analog_ready.instrument.replace import instrument
from analog_ready.accuracy import accuracy_report
from analog_ready.sweeps import (_profile_val, profile_adc_fidelity, profile_adc_fixed,
                                 profile_program_noise_fidelity)
from analog_ready.drift import retention_curve
from analog_ready.compose import end_of_life_curve

# Standard retention horizons surfaced when a profile declares a drift exponent: 1s / 1hr / 1day /
# 1month / 1year, relative to the profile's own t0=1.0 reference time.
_RETENTION_TIMES = [1.0, 3600.0, 86400.0, 2.6e6, 3.15e7]


@dataclass
class AnalysisReport:
    model: str
    profile_name: str
    ops: list                 # list of dicts: name, cost, score, break_even
    total_pj: float
    sweep_param: str
    sweep: list               # list of {value, n_favorable}
    _profile_redacted: dict
    coverage: dict = field(default_factory=dict)
    layers: list = field(default_factory=list)
    recovery: list = field(default_factory=list)
    accuracy: dict | None = None
    program_noise_fidelity: float | None = None   # fidelity under the profile's characterised
    #                                                weight-programming noise (prog/read sigma)
    retention: list = field(default_factory=list)   # fidelity-vs-time under the profile's own
    #                                                  conductance-drift coefficients (drift_nu /
    #                                                  drift_sigma_nu); a derived LOCAL artifact,
    #                                                  never in the shareable/redacted view.
    adc_fidelity: float | None = None   # fidelity under the profile's own enob_avail readout
    #                                     precision (W9, best-case auto-ranged); a derived LOCAL
    #                                     artifact, never in the shareable/redacted view.
    adc_fixed_fidelity: float | None = None   # W12: fidelity under a realistic FIXED-range ADC
    adc_saturation: float | None = None       # W12: fraction of the clean output that saturates
    end_of_life: list = field(default_factory=list)   # W10 capstone: fidelity-vs-time with ALL
    #                                                    effects (drift + programming noise + ADC)
    #                                                    composed; local artifact, never shareable.

    def to_dict(self) -> dict:
        d = {"model": self.model, "profile": self.profile_name,
             "total_pj": self.total_pj, "ops": self.ops,
             "sensitivity": {"param": self.sweep_param, "points": self.sweep}}
        if self.coverage:
            d["coverage"] = self.coverage
        if self.layers:
            d["layers"] = self.layers
        if self.recovery:
            d["recovery"] = self.recovery
        if self.accuracy:
            d["accuracy"] = self.accuracy
        if self.program_noise_fidelity is not None:
            d["program_noise_fidelity"] = self.program_noise_fidelity
        if self.retention:
            d["retention"] = self.retention
        if self.adc_fidelity is not None:
            d["adc_fidelity"] = self.adc_fidelity
        if self.adc_fixed_fidelity is not None:
            d["adc_fixed_fidelity"] = self.adc_fixed_fidelity
        if self.adc_saturation is not None:
            d["adc_saturation"] = self.adc_saturation
        if self.end_of_life:
            d["end_of_life"] = self.end_of_life
        return d

    def has_verdict_flip(self) -> bool:
        counts = {p["n_favorable"] for p in self.sweep}
        return len(counts) > 1

    def redacted(self) -> dict:
        """Shareable view: qualitative per-op verdicts + the redacted profile, no raw coefficients."""
        return {
            "model": self.model, "profile": self.profile_name,
            "profile_redacted": self._profile_redacted,
            "ops": [{"name": o["name"],
                     "favorability": round(o["score"]["favorability"], 2),
                     "favorable": o["break_even"]["favorable"],
                     "dominant_limiter": o["score"]["dominant_limiter"]} for o in self.ops],
            "verdict_flip": self.has_verdict_flip(),
        }


def _sweep_enob(feats, profile) -> list:
    if not feats:
        return []
    reqs = [enob_required(f, profile) for f in feats]
    lo, hi = math.floor(min(reqs)) - 1, math.ceil(max(reqs)) + 2
    points = []
    for v in range(lo, hi + 1):
        swept = _with_field(profile, "enob_avail", float(v))
        n_fav = sum(break_even(f, swept).favorable for f in feats)
        points.append({"value": float(v), "n_favorable": n_fav})
    return points


def analyze(model, profile, workload: Workload | None = None, *, inputs=None, labels=None,
            sigma: float = 0.1, seed: int = 0, accuracy_source: str | None = None) -> AnalysisReport:
    feats = extract_op_features(model, workload or Workload())
    ops, total = [], 0.0
    for f in feats:
        cost = estimate_op(f, profile)
        total += cost.total_pj
        ops.append({
            "name": f.name,
            "cost": cost.to_dict(),
            "score": score_op(f, profile).to_dict(),
            "break_even": break_even(f, profile).to_dict(),
        })

    layers, coverage, recovery = [], {}, []
    program_noise_fidelity = None
    retention: list = []
    adc_fidelity = None
    adc_fixed_fidelity = adc_saturation = None
    end_of_life: list = []
    if inputs is not None:
        layers = rank_layers(model, inputs, sigma=sigma, seed=seed)
        _, cov_report = instrument(copy.deepcopy(model))
        coverage = cov_report.to_dict()
        # reuse the ranking we just computed — recommend_recovery would otherwise re-rank the model.
        recovery = recommend_recovery(model, inputs, profile, sigma=sigma, seed=seed, ranked=layers)
        # make the profile's HIDDEN programming-noise coefficients actually drive a reported number.
        program_noise_fidelity = profile_program_noise_fidelity(model, inputs, profile, seed=seed)
        # make the profile's own drift coefficients drive a reported retention curve (W8).
        drift_nu = _profile_val(profile, "drift_nu")
        if drift_nu is not None:
            drift_sigma_nu = _profile_val(profile, "drift_sigma_nu") or 0.0
            retention = retention_curve(model, inputs, nu=float(drift_nu),
                                        sigma_nu=float(drift_sigma_nu), times=_RETENTION_TIMES,
                                        t0=1.0, seed=seed)
            # W10 capstone: compose drift + programming noise + ADC into one end-of-life curve. Gated
            # on drift, because "end-of-life over time" only means something when a time-dependent
            # effect exists — a driftless profile (e.g. photonic) would show a misleading flat curve.
            # draws=8 matches retention/program-noise, so the composite mean is no noisier than the
            # components it is read against.
            end_of_life = end_of_life_curve(model, inputs, profile, times=_RETENTION_TIMES, draws=8,
                                            seed=seed)
        # make the profile's own ENOB (enob_avail) drive a reported ADC readout-precision fidelity
        # number (W9) instead of sitting as cost-model-only metadata.
        adc_fidelity = profile_adc_fidelity(model, inputs, profile, seed=seed)
        # W12: the realistic FIXED-range ADC (saturation + per-tensor range) alongside the best-case
        # auto number above, driven by the profile's adc_full_scale.
        adc_fixed_fidelity, adc_saturation = profile_adc_fixed(model, inputs, profile, seed=seed)

    accuracy = None
    if inputs is not None and labels is not None:
        accuracy = accuracy_report(model, inputs, labels, sigma=sigma, seed=seed)
        # provenance of the eval data, so the report can label a synthetic fixture honestly and a
        # real --data-dir run truthfully.
        accuracy["source"] = accuracy_source or "the supplied labels"

    return AnalysisReport(
        model=type(model).__name__.removeprefix("_"), profile_name=profile.name, ops=ops, total_pj=total,
        sweep_param="enob_avail", sweep=_sweep_enob(feats, profile),
        _profile_redacted=profile.redacted(),
        coverage=coverage, layers=layers, recovery=recovery, accuracy=accuracy,
        program_noise_fidelity=program_noise_fidelity, retention=retention,
        adc_fidelity=adc_fidelity, adc_fixed_fidelity=adc_fixed_fidelity,
        adc_saturation=adc_saturation, end_of_life=end_of_life,
    )
