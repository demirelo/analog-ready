"""W3 acceptance oracle — regress (recurring-CI primitive): deterministic summary, real-regression
gating with no vacuous pass, and a leak-free redacted envelope."""
import json

import pytest
import torch
import torch.nn as nn

from analog_ready.regress import summarize, compare, envelope
from analog_ready.core.profile import HardwareProfile, ProfileField as PF
from analog_ready.profiles import load_profile


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))


def test_summarize_is_deterministic_and_has_keys():
    p = load_profile("aimc_pcm_4bit")
    torch.manual_seed(1)
    x = torch.randn(4, 16)
    a = summarize(_model(), p, x, seed=0)
    b = summarize(_model(), p, x, seed=0)
    assert a == b
    for k in ("favorability", "favorable", "fidelity"):
        assert k in a


def test_compare_passes_when_identical():
    cur = {"favorability": 0.6, "favorable": True, "fidelity": 0.9}
    res = compare(cur, dict(cur))
    assert res.passed and not res.failures


def test_compare_fails_on_favorability_drop():
    base = {"favorability": 0.6, "favorable": True, "fidelity": 0.9}
    cur = {"favorability": 0.4, "favorable": True, "fidelity": 0.9}
    res = compare(cur, base, tol_fav=0.05)
    assert not res.passed
    assert any("favorab" in f.lower() for f in res.failures)


def test_compare_fails_on_fidelity_drop():
    base = {"favorability": 0.6, "favorable": True, "fidelity": 0.9}
    cur = {"favorability": 0.6, "favorable": True, "fidelity": 0.7}
    res = compare(cur, base, tol_fid=0.05)
    assert not res.passed
    assert any("fidel" in f.lower() for f in res.failures)


def test_compare_fails_on_breakeven_flip():
    base = {"favorability": 0.6, "favorable": True, "fidelity": 0.9}
    cur = {"favorability": 0.6, "favorable": False, "fidelity": 0.9}
    res = compare(cur, base)
    assert not res.passed


def test_compare_refuses_vacuous_pass():
    with pytest.raises(ValueError):
        compare({"model": "x"}, {"model": "x"})          # no comparable metric


def test_compare_rejects_missing_baseline():
    with pytest.raises((ValueError, TypeError)):
        compare({"favorability": 0.6}, None)


def test_envelope_never_leaks_hidden_field():
    p = HardwareProfile(name="m", fields={
        "adc": PF(3.17, "bucket"), "mem_secret": PF(0.0314159, "hidden")})
    res = compare({"favorability": 0.6, "favorable": True, "fidelity": 0.9},
                  {"favorability": 0.6, "favorable": True, "fidelity": 0.9})
    blob = json.dumps(envelope(res, p))
    assert "0.0314159" not in blob
    assert "mem_secret" not in blob


def test_envelope_redact_strips_model_identity():
    p = HardwareProfile(name="m", fields={"adc": PF(3.17, "bucket")})
    res = compare({"favorability": 0.6, "favorable": True, "fidelity": 0.9},
                  {"favorability": 0.6, "favorable": True, "fidelity": 0.9})
    assert "SecretNet" in json.dumps(envelope(res, p, model="SecretNet", redact=False))
    assert "SecretNet" not in json.dumps(envelope(res, p, model="SecretNet", redact=True))


def test_cli_regress_fails_on_regression(tmp_path):
    from analog_ready.cli import main
    base, cur, out = tmp_path / "b.json", tmp_path / "c.json", tmp_path / "e.json"
    base.write_text(json.dumps({"favorability": 0.6, "favorable": True, "fidelity": 0.9,
                                "profile": "aimc_pcm_4bit", "model": "M"}))
    cur.write_text(json.dumps({"favorability": 0.3, "favorable": True, "fidelity": 0.9,
                               "profile": "aimc_pcm_4bit", "model": "M"}))
    rc = main(["regress", "--baseline", str(base), "--current", str(cur), "--out", str(out)])
    assert rc == 1
    assert out.exists()


def test_cli_regress_passes_when_no_regression(tmp_path):
    from analog_ready.cli import main
    base, cur, out = tmp_path / "b.json", tmp_path / "c.json", tmp_path / "e.json"
    s = {"favorability": 0.6, "favorable": True, "fidelity": 0.9,
         "profile": "aimc_pcm_4bit", "model": "M"}
    base.write_text(json.dumps(s))
    cur.write_text(json.dumps(s))
    rc = main(["regress", "--baseline", str(base), "--current", str(cur), "--out", str(out)])
    assert rc == 0
