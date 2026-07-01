"""W11 · F5 — the pilot: wire the stored MEASURED silicon references into an executable check.

For each cited measured reference we compute the tool's PREDICTED analog top-1 drop by running the
end-of-life composite (programming noise + drift + realistic fixed-range ADC) on an offline labelled
STAND-IN at the reference's device profile, and compare it to the reference's measured drop. Because
the exact reference architecture + dataset (the A0 baseline reproduction) is deferred, every record's
status is honestly `INDICATIVE` — never a claimed validated pass.

Operating points are REFERENCE-SPECIFIC (they are not the same physical comparison):
  * `software_to_analog` (e.g. HERMES) — a t0 MAPPING drop: clean software accuracy minus the
    post-mapping analog accuracy (programming noise + ADC, NO drift). Drift is identity at t_s == t0.
  * `retention` (e.g. Joshi) — a drift-RETENTION drop over the reference's horizon: post-mapping
    analog accuracy at t0 minus the same at the horizon. Programming + ADC cancel between the two
    terms, isolating the drift-only loss the reference actually measures.

Redaction: `predicted_drop`, `delta`, and `indicative_within_tolerance` are DERIVED from the
profile's HIDDEN coefficients through the composite, so — like the report's derived fidelities — they
are LOCAL-ONLY. `run_pilot(redact=True)` (CLI `pilot --redact`) drops them, leaving only published
reference facts + the redacted profile; no hidden coefficient leaves the tool in either mode.
"""
from __future__ import annotations

from analog_ready.accuracy import top1_accuracy
from analog_ready.compose import end_of_life_accuracy
from analog_ready.datasets import trained_classifier
from analog_ready.profiles import load_profile
from analog_ready.validation import REFERENCES

# Both stored references are PCM devices; the tool's PCM profile drives the stand-in prediction.
_STANDIN_PROFILE = "aimc_pcm_4bit"
_STANDIN = "an offline trained classifier on a synthetic separable task"
_T0_S = 1.0            # drift t0: drift_weight uses t0=1.0, so t_s==t0 is the no-drift (mapping) point
_HORIZON_S = 86400.0   # default ~1-day retention horizon if a reference declares none


def run_pilot(*, draws: int = 8, seed: int = 0, redact: bool = False) -> list:
    """One record per MEASURED reference: predicted vs measured drop at the reference's OWN operating
    point (mapping vs retention), an honest `INDICATIVE` status, and the redacted profile. With
    `redact=True` the coefficient-derived predictions are omitted (the shareable view)."""
    model, inputs, labels = trained_classifier(seed=seed)
    profile = load_profile(_STANDIN_PROFILE)
    fs_field = profile.fields.get("adc_full_scale")
    adc_fs = fs_field.value if fs_field is not None else None   # realistic fixed-range ADC if present

    clean = top1_accuracy(model, inputs, labels)
    # Post-mapping analog accuracy (programming + fixed-range ADC, no drift): the t0 operating point
    # shared by both the mapping drop and the retention baseline.
    acc_t0 = end_of_life_accuracy(model, inputs, labels, profile, t_s=_T0_S, draws=draws, seed=seed,
                                  adc_full_scale=adc_fs)

    records = []
    for ref in REFERENCES:
        if ref.get("kind") != "measured":
            continue
        comparison = ref.get("comparison", "software_to_analog")
        if comparison == "retention":
            horizon = float(ref.get("horizon_s", _HORIZON_S))
            acc_h = end_of_life_accuracy(model, inputs, labels, profile, t_s=horizon, draws=draws,
                                         seed=seed, adc_full_scale=adc_fs)
            predicted_drop = acc_t0 - acc_h        # drift-only retention drop from the mapped baseline
        else:                                      # software_to_analog
            horizon = _T0_S
            predicted_drop = clean - acc_t0        # t0 mapping drop (programming + ADC), no drift

        baseline = ref.get("software_baseline", ref["reference_value"])
        measured_drop = baseline - ref["reference_value"]
        delta = abs(predicted_drop - measured_drop)

        record = {
            "name": ref["name"],
            "citation": ref["citation"],
            "kind": ref["kind"],
            "reference_model": ref.get("model", ""),
            "operating_point": ref.get("operating_point", ""),
            "comparison": comparison,             # software_to_analog | retention
            "horizon_s": horizon,
            "measured_value": ref["reference_value"],
            # NEUTRAL, self-describing baseline: `software_baseline` would be a lie for the Joshi
            # record (its value is post-mapping HARDWARE, pre-drift). `baseline_kind` names exactly
            # what `measured_drop` is measured DOWN FROM, per reference.
            "baseline_value": baseline,
            "baseline_kind": ref.get("baseline_kind", "unspecified"),
            "measured_drop": measured_drop,
            "tolerance": ref["tolerance"],
            "status": (f"INDICATIVE — predicted on {_STANDIN} at the {_STANDIN_PROFILE} profile, "
                       f"NOT {ref.get('model', 'the reference architecture')} on its own dataset. A "
                       f"rigorous validation needs the exact architecture + dataset (deferred A0 "
                       f"baseline reproduction), not this stand-in."),
            "profile_redacted": profile.redacted(),
        }
        if not redact:
            # DERIVED from hidden profile coefficients via the composite → local-only, like the
            # report's derived fidelities. The name/status carry the INDICATIVE caveat so a consumer
            # that filters on the boolean alone can't misread it as a validated match.
            record["predicted_drop"] = predicted_drop
            record["delta"] = delta
            record["indicative_within_tolerance"] = delta <= ref["tolerance"] + 1e-12
        records.append(record)
    return records
