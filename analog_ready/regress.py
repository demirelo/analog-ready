"""W3 — regress: the recurring-CI primitive. Summarise a model+profile run into a small
deterministic record, compare a current record against a baseline, and emit a redacted envelope a
hardware vendor can share. Gates on real regressions; refuses to pass vacuously; the envelope reuses
the profile's own redaction so a `hidden` field never leaves the tool."""
from __future__ import annotations

from dataclasses import dataclass, field

from analog_ready.analyze import analyze
from analog_ready.sweeps import degradation_curve


@dataclass
class RegressionResult:
    passed: bool
    deltas: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)


def summarize(model, profile, inputs=None, *, ref_sigma: float = 0.1, draws: int = 8,
              seed: int = 0) -> dict:
    """A deterministic, comparable record of a model+profile run."""
    rep = analyze(model, profile)
    pjs = [o["cost"]["total_pj"] for o in rep.ops]
    favs = [o["score"]["favorability"] for o in rep.ops]
    total = sum(pjs)
    if total > 0:                                                     # energy-weighted
        favorability = sum(f * p for f, p in zip(favs, pjs)) / total
    else:                                                            # all-zero energy -> honest unweighted mean
        favorability = sum(favs) / len(favs) if favs else 0.0
    favorable = any(o["break_even"]["favorable"] for o in rep.ops)
    fidelity_at_ref = 1.0
    if inputs is not None:
        fidelity_at_ref = degradation_curve(
            model, inputs, param="sigma", values=[ref_sigma], draws=draws, seed=seed)[0]["fidelity"]
    return {
        "model": type(model).__name__.removeprefix("_"),
        "profile": profile.name,
        "favorability": favorability,
        "favorable": favorable,
        "total_pj": rep.total_pj,
        "verdict_flip": rep.has_verdict_flip(),
        "fidelity": fidelity_at_ref,
    }


def compare(current: dict, baseline: dict, *, tol_fav: float = 0.05,
            tol_fid: float = 0.05) -> RegressionResult:
    """Fail if favorability drops > tol_fav, `favorable` flips True->False, or fidelity drops >
    tol_fid. Raise ValueError if the baseline is missing/empty or there is no comparable metric."""
    if not baseline:
        raise ValueError("missing or empty baseline — nothing to compare against")
    deltas: dict = {}
    failures: list = []
    comparable = False
    # Fail closed on schema mismatch: a gating metric present in only one record cannot be verified,
    # so a real regression in it would otherwise pass silently. Treat the asymmetry as a failure.
    for key in ("favorability", "fidelity", "favorable"):
        if (key in current) != (key in baseline):
            comparable = True
            where = "current" if key in current else "baseline"
            failures.append(f"{key!r} present only in {where} — cannot verify (treating as regression)")
    if "favorability" in current and "favorability" in baseline:
        comparable = True
        d = current["favorability"] - baseline["favorability"]
        deltas["favorability"] = d
        if d < -tol_fav:
            failures.append(f"favorability dropped {-d:.3f} (> tol {tol_fav})")
    if "fidelity" in current and "fidelity" in baseline:
        comparable = True
        d = current["fidelity"] - baseline["fidelity"]
        deltas["fidelity"] = d
        if d < -tol_fid:
            failures.append(f"fidelity dropped {-d:.3f} (> tol {tol_fid})")
    if "favorable" in current and "favorable" in baseline:
        comparable = True
        if baseline["favorable"] and not current["favorable"]:
            failures.append("break-even flipped favorable -> unfavorable")
    if not comparable:
        raise ValueError("no comparable metric (favorability/fidelity/favorable) in both records")
    return RegressionResult(passed=not failures, deltas=deltas, failures=failures)


def envelope(result: RegressionResult, profile, *, model=None, redact: bool = False) -> dict:
    """A shareable verdict. Never includes a `hidden` profile field's raw value (reuses
    profile.redacted()). With redact=True, the model identity is omitted."""
    env = {
        "passed": result.passed,
        "failures": list(result.failures),
        "deltas": dict(result.deltas),
        "profile_redacted": profile.redacted(),
    }
    if model is not None and not redact:
        env["model"] = model
    return env
