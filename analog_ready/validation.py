"""W6 · F4 — measured-hardware validation harness. A predicted degradation is only credible if it
can be checked against a real, cited reference. This module hardcodes published analog-accelerator
operating points and reports whether a simulator's prediction lands within a documented band.
Honest about provenance: `kind` is "measured" (real fabricated silicon) or "published_simulation"
(a simulated/noise-modelled device, no claim of measured hardware)."""
from __future__ import annotations

# REFERENCES[0]: IBM "HERMES Project Chip" — a 64-core mixed-signal in-memory-compute chip on
# phase-change memory (PCM), fabricated in 14nm CMOS. MEASURED on real silicon (not simulated).
#
#   Le Gallo, M., Khaddam-Aljameh, R., Stanisavljevic, M., et al. "A 64-core mixed-signal
#   in-memory compute chip based on phase-change memory for deep neural network inference."
#   Nature Electronics 6, 680-693 (2023). arXiv:2212.02872.
#
# Quote (paper, p.5, Fig. 3c discussion): "The CIFAR-10 test accuracy results for ODP and TDP
# weight programming are shown in Fig. 3c. We achieve a hardware accuracy of 92.81% with TDP,
# which is less than 1% below the software baseline of 93.67%, whereas ODP achieves 92.23%."
#
# We take the TDP (two-device-per-cell) measured hardware accuracy, 92.81%, as the reference value.
# Tolerance is set to cover the spread between the two reported on-chip programming schemes
# (ODP 92.23% vs TDP 92.81%, i.e. ~0.6 points) plus a small margin, so a small fixed Linear-layer
# variant of the same hardware-aware-trained ResNet-9/CIFAR-10 setup that lands anywhere in
# [91.8%, 93.8%] is considered consistent with the published silicon measurement.
REFERENCES = [
    {
        "name": "IBM HERMES Project Chip — ResNet-9/CIFAR-10 (PCM, TDP)",
        "citation": "Le Gallo, M. et al. \"A 64-core mixed-signal in-memory compute chip based on "
                    "phase-change memory for deep neural network inference.\" Nature Electronics 6, "
                    "680-693 (2023). arXiv:2212.02872.",
        "model": "ResNet-9 (1,866,536 synaptic weights)",
        "metric": "top-1 test accuracy",
        "operating_point": "CIFAR-10 test set; TDP (two-device-per-cell) PCM weight programming; "
                            "8-bit input/output MVM, 4-phase high-precision read",
        "reference_value": 0.9281,   # measured hardware accuracy, TDP programming (paper Fig. 3c)
        "software_baseline": 0.9367,  # the paper's FP32 software baseline (measured drop ~0.86pt)
        "baseline_kind": "software_fp32",  # 0.9367 IS a genuine FP32 software baseline
        "tolerance": 0.01,           # ~1pt: covers the ODP/TDP spread (92.23-92.81%) + margin
        "kind": "measured",          # explicitly measured on fabricated 14nm CMOS silicon
        "relation": "exact",         # 0.9281 is a point measurement — compared symmetrically
        "comparison": "software_to_analog",  # a t0 MAPPING drop (programming + ADC), NOT retention
    },
    {
        "name": "IBM PCM inference — ResNet-32/CIFAR-10 (1-day drift RETENTION)",
        "citation": "Joshi, V. et al. \"Accurate deep neural network inference using computational "
                    "phase-change memory.\" Nature Communications 11, 2473 (2020). arXiv:1906.03138.",
        "model": "ResNet-32 (~0.46M weights)",
        "metric": "top-1 test accuracy (retention over time, not a software->analog drop)",
        "operating_point": "CIFAR-10; weights programmed on real PCM devices; a RETENTION reference — "
                           "accuracy right after programming vs after ~1 day of conductance drift. The "
                           ">93.5% one-day figure is COMPENSATION-ASSISTED: the paper applies Global "
                           "Drift Compensation (GDC, a per-layer scaling correction) and Adaptive "
                           "Batch-norm Statistics (AdaBS, recalibrating BN running mean/variance); "
                           "`pilot` models RAW uncompensated drift and NEITHER technique, so its "
                           "indicative prediction is expected to sit BELOW this compensated reference.",
        # Abstract states verbatim: 93.7% "after mapping the trained weights to PCM" and accuracy
        # "above 93.5% retained over a one day period". Both numbers are MEASURED-hardware, so the
        # baseline here is the post-mapping hardware accuracy (pre-drift), NOT an FP software
        # baseline; the "drop" is the 1-day drift-retention drop. NOTE: the >93.5% one-day figure is
        # compensation-assisted — the paper reports GDC (global drift compensation, a per-layer
        # scaling correction) and AdaBS (adaptive batch-norm-statistic recalibration), NEITHER of
        # which `pilot` models, so pilot's uncompensated prediction should UNDERSHOOT it. This nuance
        # rides along in `operating_point`, attached to every pilot record.
        "reference_value": 0.935,    # LOWER BOUND: abstract "above 93.5%" retained over ~1 day
        "software_baseline": 0.937,   # post-PCM-mapping hardware accuracy (abstract "93.7%"), pre-drift
        "baseline_kind": "post_mapping_hardware_pre_drift",  # NOT software: measured HW right after mapping
        "tolerance": 0.015,          # the 93.5% floor is a bound, so allow margin on the retained value
        "kind": "measured",          # real PCM silicon in the loop (not a pure noise-model sim)
        "relation": "lower_bound",   # abstract: accuracy stays ABOVE 93.5% — a one-sided (>=) bound
        "comparison": "retention",   # a 1-day drift-RETENTION drop from the post-mapping baseline
        "horizon_s": 86400.0,        # ~1 day, matching the abstract's retention window
    },
]


def validate_against_reference(predicted: float, ref: dict) -> dict:
    """Compare a predicted metric value against a published reference operating point.

    `relation` (default "exact") sets the band shape:
      * "exact"       — a point measurement; within_band iff |predicted - reference| <= tolerance.
      * "lower_bound" — the reference is a floor the paper reports the metric stays ABOVE (e.g.
                        Joshi's ">93.5% over one day"). A value at or above the floor is consistent,
                        and we allow `tolerance` slack BELOW it; there is no upper edge. Rendering a
                        symmetric band here would wrongly reject a healthy prediction comfortably
                        above the floor.
    Returns {"within_band", "predicted", "reference", "delta", "tolerance", "relation",
    "citation", "kind"}.
    """
    reference_value = ref["reference_value"]
    tolerance = ref["tolerance"]
    relation = ref.get("relation", "exact")
    delta = predicted - reference_value
    if relation == "lower_bound":
        within_band = delta >= -(tolerance + 1e-12)        # at/above the floor, or within slack below
    else:
        within_band = abs(delta) <= tolerance + 1e-12      # inclusive band edge, float-safe
    return {
        "within_band": within_band,
        "predicted": predicted,
        "reference": reference_value,
        "delta": delta,
        "tolerance": tolerance,
        "relation": relation,
        "citation": ref["citation"],
        "kind": ref["kind"],
    }
