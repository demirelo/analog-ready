"""The analog-friendliness score is DERIVED from the cost-model features (not a standalone rubric):
hard ENOB / roofline gates x a weighted geometric mean of soft features, with a dominant-limiter
string, per-gate confidence labels, and an auditable feature table — never a single opaque scalar."""
import torch.nn as nn


def _feat(k=16, n=64, tokens=128):
    from analog_ready.features import extract_op_features, Workload

    return extract_op_features(nn.Linear(k, n), Workload(tokens=tokens, batch=1))[0]


def test_favorability_in_unit_interval_with_feature_table():
    from analog_ready.profiles import load_profile
    from analog_ready.score import score_op

    s = score_op(_feat(), load_profile("aimc_sram_8bit"))
    assert 0.0 <= s.favorability <= 1.0
    # the decomposition is exposed — a reviewer can audit every term, not just a "score = 82"
    assert s.features and isinstance(s.features, dict)
    assert s.contributions and isinstance(s.contributions, dict)
    assert s.dominant_limiter


def test_enob_gate_hard_vetoes_deep_contraction():
    from analog_ready.profiles import load_profile
    from analog_ready.score import score_op

    deep = _feat(k=4096, n=64)   # ENOB_req = 4 + 0.5*log2(4096) = 4 + 6 = 10 >> enob_avail 6
    s = score_op(deep, load_profile("aimc_pcm_4bit"))
    assert s.favorability == 0.0
    assert any(g.name == "precision" and g.status == "fail" for g in s.gates)
    assert "precision" in s.dominant_limiter.lower() or "enob" in s.dominant_limiter.lower()


def test_memory_bound_op_scores_lower_than_compute_bound():
    from analog_ready.profiles import load_profile
    from analog_ready.score import score_op

    p = load_profile("aimc_sram_8bit")
    # tiny contraction = low arithmetic intensity (memory-bound); large = compute-bound
    mem_bound = score_op(_feat(k=8, n=8, tokens=8), p)
    compute_bound = score_op(_feat(k=512, n=512, tokens=256), p)
    assert mem_bound.features["intensity"] <= compute_bound.features["intensity"]


def test_gates_carry_confidence_and_empirical_flag():
    from analog_ready.profiles import load_profile
    from analog_ready.score import score_op

    s = score_op(_feat(), load_profile("aimc_sram_8bit"))
    assert s.gates
    for g in s.gates:
        assert g.confidence
        assert isinstance(g.requires_empirical_validation, bool)
