"""W6 · F4 — the measured-hardware validation harness. A predicted degradation is only credible if
it can be checked against a real, cited reference. This harness encodes published analog-accelerator
operating points (with citations) and reports whether the simulator's prediction lands within a
documented band. Honest about kind: 'measured' vs 'published_simulation'."""
from __future__ import annotations


def test_references_are_well_formed_and_cited():
    from analog_ready.validation import REFERENCES
    assert len(REFERENCES) >= 1
    required = ("name", "citation", "model", "metric", "operating_point",
                "reference_value", "tolerance", "kind")
    for ref in REFERENCES:
        for key in required:
            assert key in ref, f"reference missing {key!r}"
        assert isinstance(ref["citation"], str) and ref["citation"].strip()
        assert ref["kind"] in ("measured", "published_simulation")
        assert ref["tolerance"] > 0


def test_validate_against_reference_band_logic():
    from analog_ready.validation import REFERENCES, validate_against_reference
    ref = REFERENCES[0]
    rv, tol = ref["reference_value"], ref["tolerance"]

    inside = validate_against_reference(rv + 0.5 * tol, ref)
    assert inside["within_band"] is True
    assert abs(inside["delta"] - 0.5 * tol) < 1e-9
    assert inside["citation"] == ref["citation"]

    outside = validate_against_reference(rv + 2.0 * tol, ref)
    assert outside["within_band"] is False
    assert abs(outside["predicted"] - (rv + 2.0 * tol)) < 1e-9
