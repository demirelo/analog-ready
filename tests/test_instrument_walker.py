"""The instrumentation walker recursively replaces nn.Linear / nn.Conv2d and name-matched
attention projections with noisy variants, and emits a coverage report whose denominator is the
ELIGIBLE targets (not every module). A fused-QKV that is linear-like but not nn.Linear (GPT-2
Conv1D) must be reported as unhandled, not silently mis-wrapped."""
import torch.nn as nn

from helpers import (
    LlamaStyleAttention, BertStyleAttention, GPT2StyleBlock, TinyConvNet,
)


def _instrument(model):
    from analog_ready.instrument.replace import instrument

    return instrument(model)  # -> (model, CoverageReport)


def test_llama_projections_all_wrapped():
    from analog_ready.instrument.replace import NoisyLinear

    model, cov = _instrument(LlamaStyleAttention())
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert isinstance(getattr(model, name), NoisyLinear)
    assert cov.linear_replaced == 4
    assert cov.attention_projection_match_rate == 1.0


def test_bert_projections_all_wrapped():
    from analog_ready.instrument.replace import NoisyLinear

    model, cov = _instrument(BertStyleAttention())
    for name in ("query", "key", "value", "dense"):
        assert isinstance(getattr(model, name), NoisyLinear)
    assert cov.linear_replaced == 4


def test_fused_qkv_conv1d_is_unhandled_not_wrapped():
    from analog_ready.instrument.replace import NoisyLinear
    from helpers import Conv1DLike

    model, cov = _instrument(GPT2StyleBlock())
    # c_attn is a Conv1D-like fused QKV — must NOT be wrapped as a Linear
    assert isinstance(model.c_attn, Conv1DLike)
    assert any("c_attn" in u for u in cov.unhandled_linear_like)
    assert isinstance(cov.fused_qkv_policy, str) and cov.fused_qkv_policy
    # the real nn.Linear projections in the block ARE wrapped
    assert isinstance(model.c_proj, NoisyLinear)


def test_conv_and_nested_traversal_with_eligible_denominator():
    from analog_ready.instrument.replace import NoisyConv2d, NoisyLinear

    model, cov = _instrument(TinyConvNet())
    # nested Sequential is traversed; all conv2d wrapped
    assert isinstance(model.conv1, NoisyConv2d)
    assert isinstance(model.block[0], NoisyConv2d)
    assert isinstance(model.block[2], NoisyConv2d)
    assert isinstance(model.head, NoisyLinear)
    n_conv = 3
    n_linear = 1
    assert cov.conv_replaced == n_conv
    assert cov.linear_replaced == n_linear
    # modules_total counts ELIGIBLE targets only (conv+linear), not ReLU/containers
    assert cov.modules_total == n_conv + n_linear


def test_no_double_wrapping():
    from analog_ready.instrument.replace import NoisyLinear

    model, _ = _instrument(LlamaStyleAttention())
    inner = model.q_proj.base if hasattr(model.q_proj, "base") else None
    assert isinstance(model.q_proj, NoisyLinear)
    assert inner is not None and not isinstance(inner, NoisyLinear), "wrapper.base must be the original module"
    assert isinstance(inner, nn.Linear)
