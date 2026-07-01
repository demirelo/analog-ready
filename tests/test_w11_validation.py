"""W11 · F4/F5 — wire measured validation into the CLI.

F4: add a second cited MEASURED reference (Joshi et al. 2020, PCM ResNet-32/CIFAR-10) alongside
HERMES, and give references a `software_baseline` so a measured *drop* is defined.
F5: `analog-ready pilot` runs the stored references through the tool's composite on an offline
labelled stand-in and emits predicted/reference/delta/tolerance/status per reference. Because the
exact reference architecture+dataset (A0) is deferred, the status is honestly `indicative`, never a
claimed validated pass. Redaction-safe: no hidden coefficient leaves the tool.
"""
import json
import subprocess
import sys

from analog_ready.pilot import run_pilot
from analog_ready.validation import REFERENCES


def test_two_measured_references_incl_joshi():
    text = " ".join(str(r.get("name", "")) + str(r.get("citation", "")) for r in REFERENCES)
    assert "1906.03138" in text or "Joshi" in text          # the added Joshi reference
    assert "2212.02872" in text or "HERMES" in text          # the existing HERMES reference
    assert sum(1 for r in REFERENCES if r.get("kind") == "measured") >= 2


def test_references_have_a_software_baseline_above_the_measured_value():
    for r in REFERENCES:
        assert "software_baseline" in r, f"{r.get('name')} missing software_baseline"
        assert r["software_baseline"] >= r["reference_value"]   # analog measured <= software


def test_run_pilot_records_are_well_formed_and_indicative():
    recs = run_pilot()
    assert len(recs) >= 2
    for rec in recs:
        for k in ("name", "citation", "kind", "measured_drop", "predicted_drop", "delta",
                  "tolerance", "status"):
            assert k in rec, f"pilot record missing {k}"
        assert "indicative" in rec["status"].lower()   # never claims a validated pass pre-A0
        assert rec["measured_drop"] >= 0.0
        assert abs(rec["delta"] - abs(rec["predicted_drop"] - rec["measured_drop"])) < 1e-9


def test_pilot_cli_runs_and_leaks_no_hidden_coefficient(tmp_path):
    out = tmp_path / "pilot.json"
    rc = subprocess.call([sys.executable, "-m", "analog_ready.cli", "pilot", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert isinstance(data, list) and len(data) >= 2
    assert "mem_energy_pj_per_byte" not in json.dumps(data)   # a hidden coeff must not leak
