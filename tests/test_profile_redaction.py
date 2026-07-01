"""HardwareProfile carries per-field redaction (exact_ok | bucket | hidden). The redacted view
must expose exact values only for exact_ok, never leak a hidden field's raw value, and coarsen a
bucket field to a non-exact representation."""


def _profile():
    from analog_ready.core.profile import HardwareProfile, ProfileField

    return HardwareProfile(
        name="vendor_private",
        visibility="private",
        fields={
            "adc_bits": ProfileField(value=8, redaction="exact_ok"),
            "enob_avail": ProfileField(value=7.2, redaction="bucket"),
            "output_noise_sigma": ProfileField(value=0.031, redaction="hidden"),
        },
    )


def test_exact_ok_preserves_value():
    red = _profile().redacted()
    assert red["adc_bits"] == 8


def test_hidden_never_leaks_raw_value():
    red = _profile().redacted()
    blob = repr(red)
    assert "0.031" not in blob, "a hidden field's raw value must not appear in the redacted view"


def test_bucket_is_coarsened_not_exact_not_hidden():
    red = _profile().redacted()
    assert "enob_avail" in red
    val = red["enob_avail"]
    assert val != 7.2, "a bucket field must not expose the exact value"
    assert val is not None and str(val) != "", "a bucket field must still be summarized, not dropped"


def test_redaction_values_are_validated():
    from analog_ready.core.profile import ProfileField
    import pytest

    with pytest.raises((ValueError, AssertionError)):
        ProfileField(value=1, redaction="not_a_real_mode")
