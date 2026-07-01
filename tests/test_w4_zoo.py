"""W4 zoo — a small, pure-PyTorch, network-free set of stand-in workloads (no torchvision/timm/
transformers downloads). They exist to exercise the instrumentation + cost-model end to end
deterministically: a conv net, a GPT-2-style attention block (so the attention-projection name
matcher has something to match), an MLP, and a single GEMM (the synthetic MVM sensitivity curve).
Each model is buildable, deterministic, and runs forward on its own example inputs."""
import torch
import torch.nn as nn


def test_list_models_is_nonempty_and_buildable():
    from analog_ready import zoo

    names = zoo.list_models()
    assert isinstance(names, list) and names
    # the committed validation members the plan names
    for required in ("mlp", "tiny_resnet", "gpt2_block", "synthetic_gemm"):
        assert required in names
    for name in names:
        m = zoo.build(name)
        assert isinstance(m, nn.Module)


def test_example_inputs_drive_a_forward_pass():
    from analog_ready import zoo

    for name in zoo.list_models():
        m = zoo.build(name).eval()
        x = zoo.example_inputs(name)
        assert torch.is_tensor(x)
        with torch.no_grad():
            y = m(x)
        assert torch.is_tensor(y)


def test_build_is_deterministic():
    from analog_ready import zoo

    a = zoo.build("gpt2_block")
    b = zoo.build("gpt2_block")
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"non-deterministic param {k}"


def test_unknown_model_is_a_clear_error():
    from analog_ready import zoo

    try:
        zoo.build("does_not_exist")
    except (KeyError, ValueError):
        return
    raise AssertionError("building an unknown model must raise KeyError/ValueError")


def test_gpt2_block_exposes_attention_projection_names():
    """The point of shipping a GPT-2 block: its Linear layers carry c_attn/c_proj names so the
    W1 attention-projection matcher actually fires. Instrumenting it must replace >0 modules."""
    from analog_ready import zoo
    from analog_ready.instrument.replace import instrument
    from analog_ready.noise import GaussianNoise

    m = zoo.build("gpt2_block")
    named = dict(m.named_modules())
    assert any(("c_attn" in n or "c_proj" in n) and isinstance(mod, nn.Linear)
               for n, mod in named.items()), "gpt2_block must carry attention-projection Linears"
    _, coverage = instrument(m, GaussianNoise(0.01))
    assert coverage.linear_modules_replaced > 0


def test_tiny_resnet_has_conv_and_linear():
    from analog_ready import zoo

    mods = list(zoo.build("tiny_resnet").modules())
    assert any(isinstance(m, nn.Conv2d) for m in mods)
    assert any(isinstance(m, nn.Linear) for m in mods)


def test_zoo_analyzes_under_a_profile():
    """The zoo is the analyzer's fuel: every member yields a non-empty per-op report."""
    from analog_ready import zoo
    from analog_ready.analyze import analyze
    from analog_ready.profiles import load_profile

    rep = analyze(zoo.build("mlp"), load_profile("aimc_sram_8bit"))
    assert rep.ops
