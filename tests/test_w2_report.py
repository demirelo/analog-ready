"""analyze() ties features + cost + score + break-even into one report: a per-op table, a model
roll-up, a sensitivity sweep that flips at least one verdict (the DoD), and a redacted view that
never leaks a hidden profile coefficient."""
import torch.nn as nn


def test_analyze_produces_per_op_table_and_verdict_flip():
    from analog_ready.profiles import load_profile
    from analog_ready.analyze import analyze

    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1024))
    rep = analyze(model, load_profile("aimc_sram_8bit"))
    d = rep.to_dict()
    assert len(d["ops"]) == 2
    for op in d["ops"]:
        assert "cost" in op and "score" in op and "break_even" in op
    assert "total_pj" in d
    # the headline: somewhere in the swept envelope a verdict flips
    assert rep.has_verdict_flip() is True


def test_report_redacted_hides_hidden_profile_fields():
    import json
    from analog_ready.profiles import load_profile
    from analog_ready.analyze import analyze

    p = load_profile("aimc_pcm_4bit")
    rep = analyze(nn.Linear(64, 64), p)
    blob = json.dumps(rep.redacted())
    # the hidden coefficient never appears (by key) anywhere in the shareable view,
    # and the redacted report does not dump raw cost coefficients
    assert "mem_energy_pj_per_byte" not in blob
    assert "coefficients" not in blob


def test_report_has_no_single_opaque_score():
    from analog_ready.profiles import load_profile
    from analog_ready.analyze import analyze

    rep = analyze(nn.Linear(128, 256), load_profile("aimc_sram_8bit"))
    op = rep.to_dict()["ops"][0]
    # the score is never a bare number — it always ships its feature decomposition + limiter
    assert isinstance(op["score"]["features"], dict) and op["score"]["features"]
    assert op["score"]["dominant_limiter"]
