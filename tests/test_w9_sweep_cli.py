"""W9 · F2 — the user-facing ADC-precision sweep, via the library AND the `analog-ready sweep
--param adc_bits` CLI (test the real entrypoint)."""
import json
import subprocess
import sys

import torch

from analog_ready.sweeps import degradation_curve
from analog_ready.zoo import build, example_inputs


def test_degradation_curve_adc_bits_improves_with_bits():
    model, inputs = build("mlp"), example_inputs("mlp")
    curve = degradation_curve(model, inputs, param="adc_bits", values=[2, 4, 8, 12], draws=1, seed=0)
    assert [p["value"] for p in curve] == [2, 4, 8, 12]
    assert curve[-1]["fidelity"] > curve[0]["fidelity"]   # more ADC bits -> higher fidelity
    assert curve[-1]["fidelity"] > 0.99                    # ~lossless readout at 12 bits


def test_degradation_curve_adc_reproducible_and_non_mutating():
    model, inputs = build("mlp"), example_inputs("mlp")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    a = degradation_curve(model, inputs, param="adc_bits", values=[4], draws=1, seed=0)
    b = degradation_curve(model, inputs, param="adc_bits", values=[4], draws=1, seed=0)
    assert a[0]["fidelity"] == b[0]["fidelity"]
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before)


def test_sweep_cli_adc_bits(tmp_path):
    out = tmp_path / "adc.json"
    rc = subprocess.call([sys.executable, "-m", "analog_ready.cli", "sweep", "--model", "mlp",
                          "--param", "adc_bits", "--values", "2,4,8", "--out", str(out)])
    assert rc == 0
    curve = json.loads(out.read_text())
    assert len(curve) == 3
    assert curve[-1]["fidelity"] > curve[0]["fidelity"]
