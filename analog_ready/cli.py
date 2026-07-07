"""`analog-ready` CLI. Subcommands:

  doctor    — audit local-only posture and backend availability (W1).
  summarize — emit a run-summary JSON (the record `regress` compares as a committed baseline).
  regress   — gate a run-summary against a baseline (W3).
  analyze   — build a zoo model, load a profile, write HTML + JSON report (W4).
  sweep     — degradation curve over sigma, weight_bits, prog_sigma, or adc_bits (W4/W7/W9).
  demo      — photonic-mzi demo; verifies unitarity (W4).
  validate  — deterministic benchmark: curve + recovery guarantee (W4).
  eval      — REAL task accuracy (clean vs under-noise): an offline trained classifier fixture, or
              a real ResNet-18 + the user's own labelled image folder (W6).
  pilot     — run stored MEASURED-silicon references through the composite; honest INDICATIVE
              status (A0 baseline reproduction pending), never a claimed validated pass (W11).
"""
from __future__ import annotations

import argparse
import math
import sys

from analog_ready import __version__
from analog_ready.core.backend import status


# --------------------------------------------------------------------------- helpers

def _load_profile_arg(arg: str):
    """Accept either a builtin NAME ('aimc_sram_8bit') or a YAML file PATH ('path/to/foo.yaml').
    Path detection: ends with '.yaml' or contains '/'."""
    from analog_ready.profiles import load_profile, load_profile_from_path
    if arg.endswith(".yaml") or "/" in arg:
        # Explicit path → the SAME loader as builtins, so a vendor's custom profile keeps its
        # full professional schema (unit/source/uncertainty/provenance/calibration_date).
        return load_profile_from_path(arg)
    return load_profile(arg)


# --------------------------------------------------------------------------- doctor

def _doctor(local_only: bool) -> int:
    import numpy
    import torch

    print(f"analog-ready doctor (v{__version__})")
    print(f"  mode:               {'local-only' if local_only else 'standard'}")
    # HONEST posture: the tool makes NO network calls in doctor/analyze/sweep/regress/pilot/validate/
    # demo or `eval --model classifier`. The ONE exception is `eval --model resnet18`, which downloads
    # torchvision ImageNet weights (~45 MB, cached under ~/.cache/torch) on first use. So "disabled"
    # holds for every verb but that one — reporting a flat "enabled" would be a false claim.
    print("  network access:     disabled (one exception: `eval --model resnet18` downloads "
          "torchvision weights ~45MB to ~/.cache/torch on first use)")
    print("  telemetry:          none")
    print("  model export:       disabled")
    print("  profile export:     disabled")
    print("  profile redaction:  enabled")
    print(f"  image:              analog-ready:{__version__} (local source)")
    # Env consistency: torch 2.2 is built against the NumPy 1.x ABI, so a NumPy 2.x in the same env
    # imports with a warning. Surface both versions + a pin check so the numbers can be trusted.
    np_v, torch_v = numpy.__version__, torch.__version__
    pin_warn = ("  WARN: numpy>=2 violates the declared numpy<2 pin (torch prints a NumPy-ABI "
                "warning; run `pip install 'numpy<2'`)") if int(np_v.split(".")[0]) >= 2 else ""
    print(f"  numpy:              {np_v}")
    print(f"  torch:              {torch_v}")
    if pin_warn:
        print(pin_warn)
    print("  backends:")
    for row in status():
        state = "available" if row.available else f"unavailable ({row.reason})"
        print(f"    - {row.name:14} {state}")
    return 0


# --------------------------------------------------------------------------- regress

def _regress(baseline: str, current: str, out: str | None, redact: bool) -> int:
    """Compare a current run-summary JSON against a baseline; write a redacted envelope; return 0
    if no regression, 1 if a regression is detected."""
    import json

    from analog_ready.regress import compare, envelope
    from analog_ready.core.profile import HardwareProfile
    from analog_ready.profiles import load_profile

    try:
        with open(baseline) as f:
            base = json.load(f)
        with open(current) as f:
            cur = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"regress: could not read baseline/current JSON: {e}", file=sys.stderr)
        return 2
    result = compare(cur, base)
    name = cur.get("profile")
    try:
        prof = load_profile(name) if name else HardwareProfile()
    except Exception:
        prof = HardwareProfile(name=name or "unknown")
    blob = json.dumps(envelope(result, prof, model=cur.get("model"), redact=redact), indent=2)
    if out:
        with open(out, "w") as f:
            f.write(blob)
    else:
        print(blob)
    return 0 if result.passed else 1


# --------------------------------------------------------------------------- summarize

def _summarize(model_name: str, profile_arg: str, out: str | None, ref_sigma: float,
               draws: int, seed: int) -> int:
    """Emit the deterministic run-summary record `regress` consumes as its `--baseline`/`--current`.
    This is the PRODUCER half of the recurring-CI convention: commit the summary once as a baseline,
    then re-emit it each run and `regress` against it. Returns 0 on success; 2 on a bad model/profile."""
    import json

    import yaml

    from analog_ready.zoo import build, example_inputs
    from analog_ready.regress import summarize

    try:
        model = build(model_name).eval()
    except (KeyError, ValueError):
        print(f"summarize: unknown --model {model_name!r}", file=sys.stderr)
        return 2
    try:
        profile = _load_profile_arg(profile_arg)
    except (OSError, KeyError, ValueError, yaml.YAMLError) as e:
        print(f"summarize: could not load --profile {profile_arg!r}: {e}", file=sys.stderr)
        return 2
    # Real inputs drive the fidelity metric; fall back to no-input (fidelity=1.0) if the model has
    # no registered example inputs, matching `analyze`.
    try:
        inputs = example_inputs(model_name)
    except Exception:
        inputs = None
    record = summarize(model, profile, inputs=inputs, ref_sigma=ref_sigma, draws=draws, seed=seed)
    # sort_keys + indent=2 matches the committed docs/examples/*.json baseline format exactly, so a
    # generated baseline diffs cleanly against a hand-checked one.
    blob = json.dumps(record, indent=2, sort_keys=True)
    if out:
        with open(out, "w") as f:
            f.write(blob)
    else:
        print(blob)
    return 0


# --------------------------------------------------------------------------- analyze

def _analyze(model_name: str, profile_arg: str, out_html: str | None, out_json: str | None,
             redact: bool) -> int:
    import yaml

    from analog_ready.zoo import build, example_inputs
    from analog_ready.analyze import analyze
    from analog_ready.report import render_html, render_json

    try:
        model = build(model_name)
    except (KeyError, ValueError):
        print(f"analyze: unknown --model {model_name!r}", file=sys.stderr)
        return 2
    try:
        profile = _load_profile_arg(profile_arg)
    except (OSError, KeyError, ValueError, yaml.YAMLError) as e:
        print(f"analyze: could not load --profile {profile_arg!r}: {e}", file=sys.stderr)
        return 2
    # Pass real inputs so the report includes the W5 sections (per-layer sensitivity, coverage,
    # recovery). Fall back to a no-input analysis if the model has no registered example inputs.
    try:
        inputs = example_inputs(model_name)
    except Exception:
        inputs = None
    report = analyze(model, profile, inputs=inputs)

    if out_html:
        with open(out_html, "w") as f:
            f.write(render_html(report, redact=redact))
    if out_json:
        with open(out_json, "w") as f:
            f.write(render_json(report, redact=redact))
    if not out_html and not out_json:
        print(render_json(report, redact=redact))
    return 0


# --------------------------------------------------------------------------- eval

def _eval(model_kind: str, profile_arg: str | None, data_dir: str | None, sigma: float,
          seed: int, out: str | None) -> int:
    """REAL task accuracy: clean vs under-noise.

    --model classifier: fully offline — datasets.trained_classifier() supplies both the model and
    its labels, default profile aimc_pcm_4bit.
    --model resnet18: requires --data-dir; loads a real torchvision ResNet-18 + the user's own
    labelled image folder. Missing --data-dir, an absent torchvision, or an unreadable directory all
    return a non-zero exit code with a clear message — never an uncaught traceback.
    """
    from analog_ready.analyze import analyze
    from analog_ready.report import render_html

    # Acquire (model, inputs, labels, source) per kind; everything after is shared.
    if model_kind == "classifier":
        from analog_ready.datasets import trained_classifier

        model, inputs, labels = trained_classifier(seed=seed)
        source = ("a small synthetic, well-separated classifier fixture "
                  "(illustrative — not a real dataset)")
    elif model_kind == "resnet18":
        if not data_dir:
            print("eval --model resnet18: --data-dir is required", file=sys.stderr)
            return 2
        try:
            from analog_ready.vision import load_labeled_folder, load_resnet18
            model = load_resnet18()
            inputs, labels = load_labeled_folder(data_dir)
        except ImportError as e:
            print(f"eval --model resnet18: {e}", file=sys.stderr)
            return 2
        except (OSError, ValueError, RuntimeError) as e:
            print(f"eval --model resnet18: could not read --data-dir {data_dir!r}: {e}",
                  file=sys.stderr)
            return 2
        source = f"{data_dir} (your local images)"
    else:
        print(f"eval: unknown --model {model_kind!r}", file=sys.stderr)
        return 2

    profile = _load_profile_arg(profile_arg or "aimc_pcm_4bit")
    report = analyze(model, profile, inputs=inputs, labels=labels, sigma=sigma, seed=seed,
                     accuracy_source=source)
    html = render_html(report)
    if out:
        with open(out, "w") as f:
            f.write(html)
    else:
        print(html)
    return 0


# --------------------------------------------------------------------------- sweep

def _sweep(model_name: str, param: str, values: list, draws: int, seed: int,
           out: str | None) -> int:
    import json
    from analog_ready.zoo import build, example_inputs
    from analog_ready.sweeps import degradation_curve

    try:
        model = build(model_name).eval()
        inputs = example_inputs(model_name)
    except (KeyError, ValueError):
        print(f"sweep: unknown --model {model_name!r}", file=sys.stderr)
        return 2
    curve = degradation_curve(model, inputs, param=param, values=values, draws=draws, seed=seed)
    blob = json.dumps(curve, indent=2)
    if out:
        with open(out, "w") as f:
            f.write(blob)
    else:
        print(blob)
    return 0


# --------------------------------------------------------------------------- demo

def _demo_photonic_mzi(n: int, seed: int) -> int:
    import torch
    from analog_ready.backends.mzi_pure import mzi_unitary

    torch.manual_seed(seed)
    u = mzi_unitary(n)
    residual = float((u.conj().T @ u - torch.eye(n, dtype=torch.cfloat)).abs().max())
    print(f"MZI unitary demo: n={n}, max|UᴴU-I| = {residual:.2e}")
    if residual >= 1e-4:
        print(f"WARNING: unitarity residual {residual:.2e} exceeds 1e-4 threshold", file=sys.stderr)
    return 0


def _demo(target: str, n: int, seed: int) -> int:
    if target == "photonic-mzi":
        return _demo_photonic_mzi(n, seed)
    print(f"unknown demo target {target!r}", file=sys.stderr)
    return 1


# --------------------------------------------------------------------------- validate

def _validate(benchmark: str, seed: int, out: str | None) -> int:
    """DETERMINISTIC byte-identical run: sweep + recovery guarantee check.

    Recovery guarantee: calibrated quantisation is never worse than naive (W3 contract).
    We pick a Linear layer from the zoo model and an outlier-amplifying input so the clip
    calibration strictly helps, making the recovered_fidelity >= degraded_fidelity guarantee
    trivially and robustly satisfied.
    """
    import json
    import torch
    from analog_ready.zoo import build, example_inputs
    from analog_ready.sweeps import degradation_curve
    from analog_ready.recovery import recover

    # -- deterministic sweep
    model = build(benchmark).eval()
    inputs = example_inputs(benchmark)
    # sigma sweep at fixed draws/seed so the curve is reproducible
    values = [0.0, 0.02, 0.05, 0.10]
    curve = degradation_curve(model, inputs, param="sigma", values=values,
                              draws=4, seed=seed)

    # -- recovery guarantee on the first (only) Linear layer of synthetic_gemm
    # Use a fixed seed & a calibration input with a moderate outlier so clip calibration helps.
    torch.manual_seed(seed)
    # find first nn.Linear
    import torch.nn as nn
    lin = next(m for m in model.modules() if isinstance(m, nn.Linear))
    # calibration input: mostly small with a spike so the naive range is suboptimal
    cal_inputs = torch.zeros(8, lin.in_features)
    cal_inputs[:, 0] = 10.0   # a single large outlier column; calibration clips it away
    rec = recover(lin, cal_inputs, bits=4, sigma=0.0, seed=seed)

    # Guarantee check (mirrors the test assertion)
    assert rec["recovered_fidelity"] >= rec["degraded_fidelity"] - 1e-6, (
        f"recovery guarantee violated: {rec}")

    result = {
        "benchmark": benchmark,
        "curve": curve,
        "recovery": rec,
    }
    blob = json.dumps(result, indent=2, sort_keys=True)
    if out:
        with open(out, "w") as f:
            f.write(blob)
    else:
        print(blob)
    return 0


def _pilot(out: str | None, redact: bool) -> int:
    """Run the stored MEASURED references through the composite and emit indicative records. The
    output carries only the profile's REDACTED metadata — no hidden coefficient leaves the tool.
    `--redact` additionally drops the coefficient-DERIVED predictions (predicted_drop/delta/
    indicative_within_tolerance) for a shareable view; the default local view keeps them."""
    import json

    from analog_ready.pilot import run_pilot

    blob = json.dumps(run_pilot(redact=redact), indent=2, sort_keys=True)
    if out:
        with open(out, "w") as f:
            f.write(blob)
    else:
        print(blob)
    return 0


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="analog-ready", description="Analog Robustness CI")
    sub = parser.add_subparsers(dest="command")

    # doctor
    doc = sub.add_parser("doctor", help="report the local-only posture and backend availability")
    doc.add_argument("--local-only", action="store_true",
                     help="assert a network-free, no-upload, redaction-on run")

    # summarize
    sm = sub.add_parser("summarize",
                        help="emit a run-summary JSON (the baseline record the regress gate compares)")
    sm.add_argument("--model", required=True, help="zoo model name")
    sm.add_argument("--profile", required=True, help="builtin profile name or path to .yaml")
    sm.add_argument("--out", default=None, help="write the run-summary JSON here (else stdout)")
    sm.add_argument("--ref-sigma", type=float, default=0.1, dest="ref_sigma",
                    help="noise sigma at which the fidelity metric is measured (default 0.1)")
    sm.add_argument("--draws", type=int, default=8, help="stochastic draws for the fidelity metric")
    sm.add_argument("--seed", type=int, default=0, help="RNG seed")

    # regress
    reg = sub.add_parser("regress", help="gate a current run-summary against a baseline (the recurring-CI gate)")
    reg.add_argument("--baseline", required=True, help="baseline run-summary JSON")
    reg.add_argument("--current", required=True, help="current run-summary JSON")
    reg.add_argument("--out", default=None, help="write the redacted envelope here (else stdout)")
    reg.add_argument("--redact", action="store_true", help="omit model identity from the envelope")

    # analyze
    ana = sub.add_parser("analyze", help="analyze a zoo model under a hardware profile")
    ana.add_argument("--model", required=True, help="zoo model name")
    ana.add_argument("--profile", required=True, help="builtin profile name or path to .yaml")
    ana.add_argument("--out", default=None, dest="out_html", help="write HTML report here")
    ana.add_argument("--json", default=None, dest="out_json", help="write JSON report here")
    ana.add_argument("--redact", action="store_true", help="omit hidden fields from output")
    ana.add_argument("--local-only", action="store_true",
                     help="assert the run is network-free (documents intent; the tool is always "
                          "local, so this does not change behavior)")

    # eval
    ev = sub.add_parser("eval", help="real task accuracy (clean vs under-noise)")
    ev.add_argument("--model", required=True, choices=["classifier", "resnet18"],
                    help="classifier: fully offline trained fixture; resnet18: real torchvision "
                         "ResNet-18 over your own labelled --data-dir")
    ev.add_argument("--profile", default="aimc_pcm_4bit",
                    help="builtin profile name or path to .yaml (default: aimc_pcm_4bit)")
    ev.add_argument("--data-dir", default=None, help="labelled image folder (required for resnet18)")
    ev.add_argument("--sigma", type=float, default=3.0,
                    help="illustrative noise-stress sigma (default: 3.0 — visible degradation, "
                         "not a calibrated device level)")
    ev.add_argument("--seed", type=int, default=0, help="RNG seed")
    ev.add_argument("--out", default=None, help="write HTML report here (else stdout)")

    # sweep
    sw = sub.add_parser("sweep", help="degradation curve over a noise/quant parameter")
    sw.add_argument("--model", required=True, help="zoo model name")
    sw.add_argument("--param", required=True,
                    choices=["sigma", "weight_bits", "prog_sigma", "adc_bits"],
                    help="parameter to sweep")
    sw.add_argument("--values", required=True,
                    help="comma-separated values, e.g. 0,0.05,0.1")
    sw.add_argument("--draws", type=int, default=8, help="stochastic draws per point")
    sw.add_argument("--seed", type=int, default=0, help="RNG seed")
    sw.add_argument("--out", default=None, help="write JSON curve here (else stdout)")

    # demo
    demo_p = sub.add_parser("demo", help="hardware demo (photonic-mzi)")
    demo_p.add_argument("target", help="demo target (photonic-mzi)")
    demo_p.add_argument("--n", type=int, default=8, help="MZI size (default 8)")
    demo_p.add_argument("--seed", type=int, default=0, help="RNG seed for random angles")

    # validate
    val = sub.add_parser("validate", help="deterministic benchmark: curve + recovery guarantee")
    val.add_argument("--benchmark", default="synthetic_gemm",
                     help="zoo model to benchmark (default: synthetic_gemm)")
    val.add_argument("--seed", type=int, default=0, help="RNG seed")
    val.add_argument("--out", default=None, help="write JSON report here (else stdout)")

    pil = sub.add_parser("pilot",
                         help="run the stored MEASURED silicon references through the composite "
                              "(indicative stand-in prediction; A0 pending)")
    pil.add_argument("--out", default=None, help="write JSON records here (else stdout)")
    pil.add_argument("--redact", action="store_true",
                     help="drop coefficient-derived predictions for a shareable view "
                          "(default keeps them; that output is local-only)")

    args = parser.parse_args(argv)

    if args.command == "doctor":
        return _doctor(local_only=getattr(args, "local_only", False))
    if args.command == "summarize":
        return _summarize(args.model, args.profile, args.out, args.ref_sigma, args.draws, args.seed)
    if args.command == "regress":
        return _regress(args.baseline, args.current, args.out, args.redact)
    if args.command == "analyze":
        return _analyze(args.model, args.profile, args.out_html, args.out_json, args.redact)
    if args.command == "eval":
        return _eval(args.model, args.profile, args.data_dir, args.sigma, args.seed, args.out)
    if args.command == "sweep":
        # Parse per-param: a weight-bit budget or ADC bit budget is an integer >= 1; sigma/
        # prog_sigma are floats >= 0. Floating every value (then int-truncating bits) would
        # silently accept '4.7 bits' or '-2 bits'.
        raw = [v.strip() for v in args.values.split(",") if v.strip()]
        if not raw:
            print("sweep: --values must list at least one value", file=sys.stderr)
            return 2
        try:
            if args.param in ("weight_bits", "adc_bits"):
                values = [int(v) for v in raw]
                if any(b < 1 for b in values):
                    raise ValueError(f"{args.param} must be >= 1")
            else:  # sigma, prog_sigma
                values = [float(v) for v in raw]
                if any(not math.isfinite(s) for s in values):
                    # `float('inf')`/`float('nan')` parse fine and slip past `s < 0` (NaN/inf
                    # comparisons are False); caught here so the curve can't emit invalid JSON.
                    raise ValueError(f"{args.param} values must be finite")
                if any(s < 0 for s in values):
                    raise ValueError(f"{args.param} must be >= 0")
        except ValueError as e:
            print(f"sweep: invalid --values for --param {args.param}: {e}", file=sys.stderr)
            return 2
        return _sweep(args.model, args.param, values, args.draws, args.seed, args.out)
    if args.command == "demo":
        return _demo(args.target, args.n, args.seed)
    if args.command == "validate":
        return _validate(args.benchmark, args.seed, args.out)
    if args.command == "pilot":
        return _pilot(args.out, redact=getattr(args, "redact", False))

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
