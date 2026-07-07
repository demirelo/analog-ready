"""W16 · the `summarize` CLI verb — the PRODUCER half of the recurring-CI convention.

`regress` compares a current run-summary against a committed baseline, but until now no verb emitted
that record, so the "commit a baseline, gate against it" convention (and the shipped GitHub Action)
had no way to produce a current summary. `summarize` closes that gap. Needs torch (zoo models), like
the rest of the CLI suite."""
import json

from analog_ready.cli import main

_RECORD_KEYS = ("favorability", "favorable", "fidelity", "model", "profile", "total_pj", "verdict_flip")


def test_summarize_emits_a_regress_record(tmp_path):
    out = tmp_path / "summary.json"
    assert main(["summarize", "--model", "mlp", "--profile", "aimc_pcm_4bit", "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    for key in _RECORD_KEYS:
        assert key in rec, f"summary is missing regress-required key {key!r}"


def test_summarize_is_deterministic_and_feeds_regress_clean(tmp_path):
    base = tmp_path / "base.json"
    cur = tmp_path / "cur.json"
    for dest in (base, cur):
        assert main(["summarize", "--model", "mlp", "--profile", "aimc_pcm_4bit",
                     "--out", str(dest)]) == 0
    # a repeatable summary is a precondition for a meaningful baseline gate
    assert base.read_text() == cur.read_text()
    # identical current vs baseline -> no regression -> exit 0 (the green-CI path)
    assert main(["regress", "--baseline", str(base), "--current", str(cur)]) == 0


def test_summarize_unknown_model_is_graceful():
    assert main(["summarize", "--model", "not_a_model", "--profile", "aimc_pcm_4bit"]) == 2


def test_summarize_unknown_profile_is_graceful():
    assert main(["summarize", "--model", "mlp", "--profile", "not_a_profile"]) == 2
