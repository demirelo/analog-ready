"""W16 - producer/gate UX, redaction, and CI wiring checks."""
import json

import pytest
import torch

from analog_ready.cli import main
from analog_ready.core.profile import HardwareProfile, ProfileField
from analog_ready.device_noise import program_noise
from analog_ready.drift import drift_weight
from analog_ready.adc import adc_saturation_fraction
from analog_ready.regress import compare, envelope


def test_summarize_is_deterministic_and_feeds_regress(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    args = ["summarize", "--model", "mlp", "--profile", "aimc_pcm_4bit"]
    assert main([*args, "--out", str(first)]) == 0
    assert main([*args, "--out", str(second)]) == 0
    assert first.read_text() == second.read_text()
    record = json.loads(first.read_text())
    assert {"model", "profile", "favorability", "favorable", "fidelity"} <= record.keys()
    assert main(["regress", "--baseline", str(first), "--current", str(second)]) == 0


def test_summarize_rejects_bad_parameters(capsys):
    assert main(["summarize", "--model", "mlp", "--profile", "aimc_pcm_4bit",
                 "--ref-sigma", "nan"]) == 2
    assert "finite" in capsys.readouterr().err
    assert main(["summarize", "--model", "mlp", "--profile", "aimc_pcm_4bit",
                 "--draws", "0"]) == 2
    assert "draws" in capsys.readouterr().err


def test_regress_rejects_malformed_json_records(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text("{}")
    current.write_text('{"model": "M"}')
    assert main(["regress", "--baseline", str(baseline), "--current", str(current)]) == 2
    assert "schema" in capsys.readouterr().err


def test_regress_refuses_to_overwrite_inputs(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    record = {"favorability": 0.5}
    baseline.write_text(json.dumps(record))
    current.write_text(json.dumps(record))
    assert main(["regress", "--baseline", str(baseline), "--current", str(current),
                 "--out", str(baseline)]) == 2
    assert "overwrite" in capsys.readouterr().err


def test_regress_rejects_unknown_profile_in_record(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    record = {"favorability": 0.5, "profile": "not_a_profile"}
    baseline.write_text(json.dumps(record))
    current.write_text(json.dumps(record))
    assert main(["regress", "--baseline", str(baseline), "--current", str(current)]) == 2
    assert "profile" in capsys.readouterr().err


def test_redacted_regression_drops_derived_numeric_deltas_and_failure_text():
    base = {"favorability": 0.8, "fidelity": 0.9, "favorable": True}
    current = {"favorability": 0.6, "fidelity": 0.9, "favorable": True}
    result = compare(current, base)
    profile = HardwareProfile(fields={
        "mem_energy_pj_per_byte": ProfileField(0.0314159, "hidden"),
        "enob_avail": ProfileField(6.0, "bucket"),
    })
    redacted = envelope(result, profile, model="SecretNet", redact=True)
    blob = json.dumps(redacted)
    assert "SecretNet" not in blob
    assert "deltas" not in redacted
    assert "0.2" not in blob
    assert redacted["failures"] == ["favorability regression"]
    assert "0.0314159" not in blob


def test_analyze_reports_missing_analysis_fields_as_a_cli_error(tmp_path, capsys):
    profile = tmp_path / "incomplete.yaml"
    profile.write_text("name: incomplete\nfields:\n  array_rows: {value: 64, redaction: exact_ok}\n")
    assert main(["analyze", "--model", "mlp", "--profile", str(profile)]) == 2
    assert "missing required analysis field" in capsys.readouterr().err


def test_program_noise_rejects_non_finite_sigmas():
    weight = torch.ones(2, 2)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            program_noise(weight, prop_sigma=bad, read_sigma=0.0)
        with pytest.raises(ValueError):
            program_noise(weight, prop_sigma=0.0, read_sigma=bad)


def test_drift_rejects_nonphysical_parameters():
    weight = torch.ones(2, 2)
    for kwargs in (
        {"nu": -0.1, "t_s": 1.0, "t0": 1.0, "sigma_nu": 0.0},
        {"nu": float("nan"), "t_s": 1.0, "t0": 1.0, "sigma_nu": 0.0},
        {"nu": 0.1, "t_s": 0.0, "t0": 1.0, "sigma_nu": 0.0},
        {"nu": 0.1, "t_s": 1.0, "t0": 1.0, "sigma_nu": float("inf")},
    ):
        with pytest.raises(ValueError):
            drift_weight(weight, **kwargs)


def test_saturation_fraction_rejects_invalid_signal_or_full_scale():
    signal = torch.tensor([1.0, -2.0])
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            adc_saturation_fraction(signal, bad)
    with pytest.raises(ValueError):
        adc_saturation_fraction(torch.tensor([float("nan")]), 1.0)


@pytest.mark.parametrize("path", ["action.yml", ".github/workflows/regress.yml"])
def test_ci_surfaces_do_not_interpolate_untrusted_inputs_into_run_commands(path):
    text = open(path).read()
    assert "${{ inputs." not in "\n".join(
        line for line in text.splitlines() if line.lstrip().startswith("run:")
    )
    assert "ANALOG_MODEL" in text
    assert "RUNNER_TEMP" in text or "runner.temp" in text
    assert "v0.1.0" not in text
