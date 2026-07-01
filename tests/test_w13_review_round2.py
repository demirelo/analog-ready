"""Round-2 read-only-review fixes (post-W12 / PR #15). Acceptance oracles.

Covers: per-reference pilot operating points + fixed-range ADC in the composite (P1-a);
a redacted pilot mode that drops coefficient-derived predictions (P1-b); custom-YAML-path
profile loading that preserves the W11 professional schema (P1-c); a one-sided band for a
`lower_bound` reference (P2-a); and a non-stale committed sample report (P2-c).
"""
import json
from pathlib import Path

from analog_ready.accuracy import top1_accuracy
from analog_ready.compose import end_of_life_accuracy
from analog_ready.datasets import trained_classifier
from analog_ready.pilot import run_pilot
from analog_ready.profiles import load_profile, load_profile_from_path
from analog_ready.validation import REFERENCES, validate_against_reference

_REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------- P1-c: profile schema
def test_custom_profile_path_preserves_professional_metadata(tmp_path):
    y = tmp_path / "vendor.yaml"
    y.write_text(
        "name: vendor_x\n"
        "fields:\n"
        "  enob_avail:\n"
        "    value: 7.0\n"
        "    redaction: bucket\n"
        "    unit: bits\n"
        "    source: vendor datasheet\n"
        "    uncertainty: '+/-0.3'\n"
        "    provenance: measured\n"
        "    calibration_date: '2026-05-01'\n"
    )
    from analog_ready.cli import _load_profile_arg
    # BOTH the library path-loader and the CLI arg-loader must preserve the full schema
    for prof in (load_profile_from_path(str(y)), _load_profile_arg(str(y))):
        f = prof.fields["enob_avail"]
        assert f.unit == "bits"
        assert f.source == "vendor datasheet"
        assert f.uncertainty == "+/-0.3"
        assert f.provenance == "measured"
        assert f.calibration_date == "2026-05-01"


# --------------------------------------------------------------- P2-a: lower-bound band
def test_lower_bound_reference_band_is_one_sided():
    joshi = next(r for r in REFERENCES if r.get("relation") == "lower_bound")
    rv, tol = joshi["reference_value"], joshi["tolerance"]
    # comfortably ABOVE a ">=" bound is consistent (a symmetric band would wrongly reject it)
    assert validate_against_reference(rv + 5 * tol, joshi)["within_band"] is True
    # within tolerance BELOW the bound is still accepted
    assert validate_against_reference(rv - 0.5 * tol, joshi)["within_band"] is True
    # far below the bound is not
    assert validate_against_reference(rv - 2 * tol, joshi)["within_band"] is False
    assert validate_against_reference(rv, joshi)["relation"] == "lower_bound"


def test_exact_reference_band_stays_symmetric():
    hermes = next(r for r in REFERENCES if r.get("relation", "exact") == "exact")
    rv, tol = hermes["reference_value"], hermes["tolerance"]
    assert validate_against_reference(rv + 0.5 * tol, hermes)["within_band"] is True
    assert validate_against_reference(rv + 2 * tol, hermes)["within_band"] is False
    assert validate_against_reference(rv - 2 * tol, hermes)["within_band"] is False


# --------------------------------------------------- P1-a: per-reference operating points
def test_references_declare_distinct_comparison_operating_points():
    comps = {r.get("comparison") for r in REFERENCES if r.get("kind") == "measured"}
    assert "software_to_analog" in comps   # HERMES: a t0 software->analog mapping drop
    assert "retention" in comps            # Joshi: a 1-day drift-retention drop


def test_pilot_uses_reference_specific_operating_points():
    # The fix: HERMES (t0 mapping) and Joshi (1-day retention) must NOT share one 1-day
    # prediction. Recompute each operating point deterministically and pin it exactly.
    m, x, y = trained_classifier(seed=0)
    prof = load_profile("aimc_pcm_4bit")
    fs = prof.fields["adc_full_scale"].value
    clean = top1_accuracy(m, x, y)
    acc_t0 = end_of_life_accuracy(m, x, y, prof, t_s=1.0, draws=8, seed=0, adc_full_scale=fs)

    recs = list(run_pilot(draws=8, seed=0))
    assert len(recs) >= 2
    for r in recs:
        # invariant the immutable W11 oracle depends on
        assert abs(r["delta"] - abs(r["predicted_drop"] - r["measured_drop"])) < 1e-9
        if r["comparison"] == "software_to_analog":
            assert abs(r["predicted_drop"] - (clean - acc_t0)) < 1e-9
        elif r["comparison"] == "retention":
            acc_h = end_of_life_accuracy(m, x, y, prof, t_s=r["horizon_s"], draws=8, seed=0,
                                         adc_full_scale=fs)
            assert abs(r["predicted_drop"] - (acc_t0 - acc_h)) < 1e-9
        else:
            raise AssertionError(f"unexpected comparison {r['comparison']!r}")


def test_composite_accuracy_supports_fixed_range_adc():
    m, x, y = trained_classifier(seed=0)
    prof = load_profile("aimc_pcm_4bit")
    auto = end_of_life_accuracy(m, x, y, prof, t_s=1.0, draws=2, seed=0)
    fixed = end_of_life_accuracy(m, x, y, prof, t_s=1.0, draws=2, seed=0, adc_full_scale=0.9)
    assert 0.0 <= auto <= 1.0
    assert 0.0 <= fixed <= 1.0
    # deterministic (no RNG divergence from the extra kwarg)
    assert fixed == end_of_life_accuracy(m, x, y, prof, t_s=1.0, draws=2, seed=0, adc_full_scale=0.9)


# --------------------------------------------------------------- P1-b: redacted pilot mode
def test_pilot_redacted_mode_drops_coefficient_derived_fields():
    red = run_pilot(draws=2, redact=True)
    assert len(red) >= 2
    for rec in red:
        for derived in ("predicted_drop", "delta", "indicative_within_tolerance"):
            assert derived not in rec, f"{derived} is coefficient-derived; must not be shareable"
        for kept in ("name", "citation", "measured_drop", "tolerance", "status"):
            assert kept in rec
    assert "mem_energy_pj_per_byte" not in json.dumps(red)   # no hidden coeff leaks either


def test_pilot_default_mode_still_carries_predictions():
    assert all("predicted_drop" in r for r in run_pilot(draws=2))


# --------------------------------------------------------------- P2-c: sample report fresh
def test_committed_sample_report_has_the_newer_physics_sections():
    html = (_REPO / "docs" / "sample_report.html").read_text().lower()
    assert "adc readout" in html          # W9/W12 ADC section present
    assert "fixed-range" in html          # W12 realistic fixed-range ADC
    assert "end-of-life" in html          # W10 composite
