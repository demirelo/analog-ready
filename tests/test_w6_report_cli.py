"""W6 · F1 integration — accuracy reaches the report and a user-facing `eval` verb (the W5 lesson:
test the CLI path, not just the function). analyze(labels=...) attaches a real top-1 accuracy block;
the HTML shows it; `analog-ready eval --model classifier` runs the full clean→under-noise accuracy
report fully offline on the trained fixture."""
from __future__ import annotations

from analog_ready.zoo import build, example_inputs
from analog_ready.profiles import load_profile
from analog_ready.analyze import analyze
from analog_ready.report import render_html
from analog_ready.cli import main


def test_analyze_surfaces_accuracy_when_labels_given():
    from analog_ready.datasets import trained_classifier
    m, X, y = trained_classifier(seed=0)
    rep = analyze(m, load_profile("aimc_pcm_4bit"), inputs=X, labels=y, sigma=100.0, seed=0)
    d = rep.to_dict()
    assert "accuracy" in d
    assert d["accuracy"]["clean"] >= 0.9
    assert d["accuracy"]["drop"] > 0.0
    assert "ccuracy" in render_html(rep)              # an Accuracy section is rendered


def test_analyze_without_labels_has_no_accuracy():
    m = build("mlp")
    d = analyze(m, load_profile("aimc_pcm_4bit"), inputs=example_inputs("mlp")).to_dict()
    assert not d.get("accuracy")


def test_eval_cli_classifier_runs_offline(tmp_path):
    out = tmp_path / "acc.html"
    rc = main(["eval", "--model", "classifier", "--sigma", "100", "--out", str(out)])
    assert rc == 0
    html = out.read_text()
    assert "ccuracy" in html
    assert "%" in html


def test_eval_cli_resnet_without_datadir_fails_gracefully():
    # missing --data-dir (or absent torchvision) must return non-zero, not raise a traceback.
    rc = main(["eval", "--model", "resnet18"])
    assert rc != 0
