"""W14 — hardening & honesty round (from the full-codebase review). Acceptance oracle.

Theme: silent garbage becomes a loud error. Unphysical inputs (bits < 1, full_scale <= 0,
negative sigma, out-of-range ADC fraction, non-finite fidelity) must RAISE, not mask; CLI verbs
must return an exit code + stderr message, never an uncaught traceback; and the doctor / vision
trust surfaces must not overclaim network posture.
"""
import pytest
import torch

from analog_ready.accuracy import accuracy_under_noise
from analog_ready.adc import ADCQuant, FixedADCQuant, adc_quantize
from analog_ready.cli import main
from analog_ready.noise import GaussianNoise
from analog_ready.sweeps import fidelity


# ---------------------------------------------------------------- Phase 1: physics validation
def test_adc_quantize_rejects_sub_one_or_non_finite_bits():
    x = torch.randn(4, 8)
    for bad in (0, -1, 0.5, float("nan"), float("inf")):   # NaN/inf slip past a bare `< 1`
        with pytest.raises(ValueError):
            adc_quantize(x, bits=bad)
    adc_quantize(x, bits=1)   # a 1-bit sign comparator is legal


def test_adc_quantize_rejects_nonpositive_or_non_finite_full_scale():
    x = torch.randn(4, 8)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            adc_quantize(x, bits=6, full_scale=bad)
    adc_quantize(x, bits=6, full_scale=None)   # auto path unchanged
    adc_quantize(x, bits=6, full_scale=1.0)


def test_adcquant_and_fixedadcquant_validate_at_construction():
    for bad_bits in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ADCQuant(bad_bits)
    with pytest.raises(ValueError):
        FixedADCQuant(0, 0.9)                 # bad bits
    for bad in (0.0, -0.5, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            FixedADCQuant(6, bad)             # fraction must be a finite value in (0, 1]
    FixedADCQuant(6, 0.9)                      # valid
    FixedADCQuant(6, 1.0)


def test_gaussian_noise_and_accuracy_reject_negative_or_non_finite_sigma():
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            GaussianNoise(bad)
    GaussianNoise(0.0)
    m = torch.nn.Linear(4, 3)
    x, y = torch.randn(5, 4), torch.zeros(5, dtype=torch.long)
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            accuracy_under_noise(m, x, y, sigma=bad)


def test_fidelity_raises_on_non_finite_inputs():
    good = torch.randn(3, 4)
    assert fidelity(good, good) == pytest.approx(1.0, abs=1e-6)   # unchanged happy path
    nan = torch.tensor([[1.0, float("nan"), 2.0]])
    inf = torch.tensor([[1.0, float("inf"), 2.0]])
    ref = torch.tensor([[1.0, 0.5, 2.0]])
    for bad in (nan, inf):
        with pytest.raises(ValueError):
            fidelity(ref, bad)


def test_fidelity_still_handles_legitimate_zero_rows():
    # a dead-ReLU / all-zero row reproduced exactly must still score 1.0, not raise.
    z = torch.zeros(2, 4)
    assert fidelity(z, z) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- Phase 2: CLI error handling
def test_analyze_unknown_model_is_a_message_not_a_traceback(capsys):
    rc = main(["analyze", "--model", "no_such_model", "--profile", "aimc_pcm_4bit"])
    assert rc == 2
    assert "no_such_model" in capsys.readouterr().err.lower()


def test_analyze_missing_profile_path_is_handled(capsys):
    rc = main(["analyze", "--model", "mlp", "--profile", "/nope/missing.yaml"])
    assert rc == 2
    assert "profile" in capsys.readouterr().err.lower()


def test_sweep_unknown_model_is_handled(capsys):
    rc = main(["sweep", "--model", "no_such_model", "--param", "sigma", "--values", "0.1"])
    assert rc == 2
    assert capsys.readouterr().err.strip()


def test_sweep_rejects_non_finite_values(capsys):
    rc = main(["sweep", "--model", "mlp", "--param", "sigma", "--values", "inf"])
    assert rc == 2
    assert capsys.readouterr().err.strip()


def test_regress_missing_files_is_a_message_not_a_traceback(capsys):
    rc = main(["regress", "--baseline", "/nope/base.json", "--current", "/nope/cur.json"])
    assert rc == 2
    assert capsys.readouterr().err.strip()


# ---------------------------------------------------------------- Phase 3/4: trust + env
def test_doctor_reports_network_disabled_and_names_the_weight_download_exception(capsys):
    rc = main(["doctor"])                       # even WITHOUT --local-only, the tool is local
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "network" in out and "disabled" in out           # never claim 'enabled'
    assert "resnet18" in out or "weights" in out            # the one real exception, disclosed


def test_doctor_reports_torch_and_numpy_versions(capsys):
    main(["doctor", "--local-only"])
    out = capsys.readouterr().out.lower()
    assert "numpy" in out and "torch" in out                # env-consistency surface


def test_vision_docstring_discloses_the_weight_download():
    import analog_ready.vision as v
    doc = (v.__doc__ or "").lower()
    assert "download" in doc or "cache" in doc              # provenance of the ~45MB weights


# ---------------------------------------------------------------- Phase 5: clearer errors
def test_cost_model_missing_profile_field_is_a_clear_error():
    from analog_ready.cost_model import estimate_op
    from analog_ready.core.profile import HardwareProfile
    from analog_ready.features import Workload, extract_op_features
    from analog_ready.zoo import build
    feat = extract_op_features(build("mlp"), Workload())[0]
    with pytest.raises(ValueError):                          # not a bare KeyError
        estimate_op(feat, HardwareProfile(fields={}))


def test_breakeven_with_field_preserves_professional_metadata():
    from analog_ready.breakeven import _with_field
    from analog_ready.core.profile import HardwareProfile, ProfileField
    prof = HardwareProfile(fields={
        "enob_avail": ProfileField(value=6.0, redaction="bucket", unit="bits",
                                   provenance="measured", source="datasheet")})
    swept = _with_field(prof, "enob_avail", 7.0)
    f = swept.fields["enob_avail"]
    assert f.value == 7.0
    assert f.unit == "bits" and f.provenance == "measured" and f.source == "datasheet"
    assert f.redaction == "bucket"
