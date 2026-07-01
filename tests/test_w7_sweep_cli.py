"""W7 · F2 — the user-facing degradation sweep over programming-noise level, via the library AND the
`analog-ready sweep --param prog_sigma` CLI (test the real entrypoint, not just the function)."""
import json
import subprocess
import sys

from analog_ready.sweeps import degradation_curve
from analog_ready.zoo import build, example_inputs


def test_degradation_curve_prog_sigma_degrades_monotonically():
    model = build("mlp")
    inputs = example_inputs("mlp")
    curve = degradation_curve(model, inputs, param="prog_sigma", values=[0.0, 0.2, 0.5],
                              draws=4, seed=0)
    assert [p["value"] for p in curve] == [0.0, 0.2, 0.5]
    assert curve[0]["fidelity"] > 0.99             # zero programming noise -> ~perfect
    assert curve[-1]["fidelity"] < curve[0]["fidelity"]   # more programming noise -> lower fidelity


def test_degradation_curve_prog_sigma_is_reproducible():
    model = build("mlp")
    inputs = example_inputs("mlp")
    a = degradation_curve(model, inputs, param="prog_sigma", values=[0.3], draws=4, seed=0)
    b = degradation_curve(model, inputs, param="prog_sigma", values=[0.3], draws=4, seed=0)
    assert a[0]["fidelity"] == b[0]["fidelity"]


def test_degradation_curve_does_not_mutate_caller_model():
    model = build("mlp")
    inputs = example_inputs("mlp")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    degradation_curve(model, inputs, param="prog_sigma", values=[0.5], draws=2, seed=0)
    after = model.state_dict()
    assert all(__import__("torch").equal(before[k], after[k]) for k in before)


def test_sweep_cli_prog_sigma(tmp_path):
    out = tmp_path / "curve.json"
    rc = subprocess.call([sys.executable, "-m", "analog_ready.cli", "sweep", "--model", "mlp",
                          "--param", "prog_sigma", "--values", "0,0.2,0.5", "--out", str(out)])
    assert rc == 0
    curve = json.loads(out.read_text())
    assert len(curve) == 3
    assert curve[0]["fidelity"] > curve[-1]["fidelity"]
