# analog-ready — Analog Robustness CI

[![CI](https://github.com/demirelo/analog-ready/actions/workflows/ci.yml/badge.svg)](https://github.com/demirelo/analog-ready/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> *Does your model survive your accelerator's noise / quantization / drift / converter budget,
> which layers break, where does analog stop beating digital, and what recovers accuracy?
> Runs behind your firewall; share only a redacted envelope.*

**AIMC-first** (analog in-memory compute: PCM / ReRAM / SRAM crossbars), with a coherent
**photonic Clements-MZI** mesh as the flagship public demo. Pure-PyTorch defaults — no GPU, no network,
no third-party accelerator libraries required to run the core.

---

## Why

Analog accelerators promise large energy wins on matrix math, but they inject noise, clamp precision
to a handful of effective bits (ENOB), and pay a steep ADC/DAC + DRAM tax at every analog/digital
boundary. Whether a *given* network actually wins is a per-model, per-profile question. Device
simulators answer it from the chip's point of view; `analog-ready` answers it from the **model's** —
and produces a shareable, rerunnable readiness report.

**The spine is the break-even envelope**, not noise injection. Everyone can inject noise. The
differentiated answer is: *for THIS model, under THIS converter / reuse / precision budget, where
does analog stop making economic sense?*

## Install

```bash
pip install -e .            # core: torch, numpy, pyyaml only
# optional backends (never on the critical path):
pip install -e '.[aimc]'    # aihwkit
pip install -e '.[mzi]'     # torchonn
```

## The CLI — one tool, nine verbs

```bash
# 1. Prove the local-only posture (auditable trust contract)
analog-ready doctor --local-only

# 2. Per-op cost + score + break-even → a self-contained HTML/JSON report (the sales artifact)
analog-ready analyze --model gpt2_block --profile aimc_pcm_4bit \
    --out report.html --json report.json --redact --local-only

# 3. Degradation curve over a swept parameter (seeded, reproducible)
analog-ready sweep --model mlp --param sigma --values 0,0.02,0.05,0.1 --out curve.json
analog-ready sweep --model mlp --param weight_bits --values 8,6,4,3 --out wbits.json
# faithful analog weight-programming noise (proportional programming error), literature defaults:
analog-ready sweep --model mlp --param prog_sigma --values 0,0.05,0.1,0.2 --out prog.json
# ADC readout quantization (effective bits), per-sample dynamic range, deterministic:
analog-ready sweep --model mlp --param adc_bits --values 8,6,4,3 --out adc.json

# 4. The photonic demo — pure-PyTorch Clements mesh, zero third-party deps
analog-ready demo photonic-mzi --n 8

# 5. Emit a run-summary JSON — the record the regress gate compares (commit it once as a baseline)
analog-ready summarize --model mlp --profile aimc_pcm_4bit --out baseline.json

# 6. The recurring-CI primitive: re-emit the summary and gate it against the baseline; fail closed
analog-ready summarize --model mlp --profile aimc_pcm_4bit --out now.json
analog-ready regress --baseline baseline.json --current now.json --redact --out envelope.json

# 7. One-command, byte-deterministic validation curve + a recovery recipe
analog-ready validate --benchmark synthetic_gemm --out validation.json

# 8. Real task accuracy on a labelled set — clean vs under an illustrative noise stress
analog-ready eval --model classifier --profile aimc_pcm_4bit --seed 0

# 9. Run the stored MEASURED-silicon references through the composite (indicative; A0 baseline pending)
analog-ready pilot --out pilot.json
```

`--profile` accepts a builtin name (`aimc_pcm_4bit`, `aimc_reram_6enob`, `aimc_sram_8bit`,
`photonic_clements_mzi`) **or** a path to your own profile YAML.

## How it works

```
1 · Instrument     2 · Cost model      3 · Break-even        4 · Report
module-replace  →  compute + ADC/DAC →  envelope (X,Y,Z)  →  self-contained HTML/JSON
walk; Noisy         + DRAM per op,       the verdict &        + redacted envelope to
Linear/Conv2d       each with source     where it flips        share + "Limits of
+ attn-proj         & uncertainty        ★ the spine           this estimate"
```

Instrumentation is **module replacement, not graph capture**: a recursive walk over
`named_children()` swaps `nn.Linear`, `nn.Conv2d`, and attention projections (`c_attn`/`c_proj`,
`q_proj`/`k_proj`/`v_proj`, …) for noisy equivalents — the aihwkit `convert_to_analog` pattern.
Architecture-agnostic and fully auditable (a coverage report says what was replaced vs skipped).

### The break-even envelope (the three gates)

Analog is favorable for an op **only if all three clear**:

| Gate | Constraint | Meaning |
|------|-----------|---------|
| **X · converter** | `converter_pj_per_mac < digital_mac_energy_pj` | converters must undercut the digital MAC they replace |
| **Y · reuse** | `weight_reuse > weight_program_energy / digital_mac` | reuse must amortise the analog weight write |
| **Z · precision** | `enob_required < enob_avail`, `enob_required = input_bits + ½·log₂K` | the required ENOB must fit the analog dynamic range |

The report shows each op's verdict, its **dominant limiter**, and a sensitivity sweep where the
verdict flips — the bottom-line *"where analog stops winning."* With a model + inputs it also
surfaces real **task accuracy** under noise, **programming-noise robustness** at the profile's
operating point, and a **retention curve** — output fidelity after 1 hour / 1 day / 1 year of
conductance drift (*"does it still hit accuracy after a day?"*), driven by the profile's own drift
coefficients.

## Trust model — local-only, redacted

A vendor's profile is crown-jewel IP. Every profile field declares how it may leave the tool:
`exact_ok` (verbatim) · `bucket` (coarsened to an order-of-magnitude band) · `hidden` (never
emitted, even via `repr`). The shareable report carries qualitative verdicts; a `hidden` coefficient
never appears. `analog-ready doctor --local-only` makes the no-network / no-telemetry / no-export /
redaction-on posture auditable.

## Recurring CI — rerunnable by design

This isn't a one-off report; it's a gate a hardware team reruns on every model /
profile / converter change. `regress` summarises a run, compares it to a baseline, **fails closed**,
and emits a redacted envelope safe to share outside the firewall. A worked example ships in
[`docs/examples/`](docs/examples/):

```bash
# the accelerator profile drifted noisier between runs:
analog-ready regress --baseline docs/examples/baseline.json \
    --current docs/examples/current.json --redact --out envelope.json
echo $?        # -> 1  (regression caught)
```

The resulting [`envelope.json`](docs/examples/envelope.json) reports
`"fidelity dropped 0.131 (> tol 0.05)"` and ships the **redacted profile** — `enob_avail` coarsened
to `"1..10"`, the hidden `mem_energy_pj_per_byte` omitted entirely. Nothing a vendor would object to
leaving their firewall.

### Drop-in GitHub Action

The CI is literal: `analog-ready` ships a composite action ([`action.yml`](action.yml)) that
summarizes a run and gates it against a **committed baseline**, failing the job on a real
regression. Generate the baseline once (`analog-ready summarize … --out ci/baseline.json`) and
commit it, then:

```yaml
# .github/workflows/analog-ready.yml
name: analog-ready gate
on: [push, pull_request]
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4          # your repo: the committed baseline (+ optional profile)
      - uses: demirelo/analog-ready@v0.1.0
        with:
          model: mlp
          profile: aimc_pcm_4bit           # a builtin name, or a path to your profile YAML
          baseline: ci/baseline.json
          redact: "true"                    # the written envelope is safe to publish
```

Prefer a reusable workflow? Call
[`.github/workflows/regress.yml`](.github/workflows/regress.yml) directly:

```yaml
jobs:
  gate:
    uses: demirelo/analog-ready/.github/workflows/regress.yml@v0.1.0
    with: { model: mlp, profile: aimc_pcm_4bit, baseline: ci/baseline.json }
```

Exit codes are fail-closed: `0` no regression, `1` a regression (favorability/fidelity drop or a
break-even flip), `2` an unreadable baseline — each fails the job so a real regression can't slip
through silently.

## Interactive explainer

Open [`docs/explainer.html`](docs/explainer.html) in any browser — a standalone (offline, no network)
page with a live **break-even explorer** that runs the exact three-gate logic from
`analog_ready/breakeven.py`. Two committed example reports sit alongside it:
[`sample_report.html`](docs/sample_report.html) (full internal view) and
[`sample_report_redacted.html`](docs/sample_report_redacted.html) (the shareable view — qualitative
verdicts only, no energy figure from which a hidden coefficient could be recovered).

## Development

```bash
make test-core                 # core suite, zero optional deps (must always pass)
make test-aimc-aihwkit         # optional aihwkit integration target
make test-mzi-torchonn         # optional torchonn integration target
python3 -m pytest -q           # everything
ruff check analog_ready/
```

Built **test-first** against an immutable acceptance oracle, one increment at a time:
skeleton → cost-model spine → sweeps/recovery/regress → product surface → the simulation realism
ladder (task accuracy, programming noise, conductance drift, ADC readout, end-of-life composition,
fixed-range ADC saturation) → the measured-reference `pilot` → three adversarial review, hardening
and correctness-audit rounds. Every increment's acceptance tests remain in `tests/` unedited.

## Honest naming ladder (no overclaiming)

1. **v0 — Analog Robustness CI** *(this release)*: simulated noise, no hardware.
2. **v1 — Profile-Conditioned QAT Runtime**: vendor-characterised profiles drive the recovery recipe.
3. **v2 — Calibration / QAT Runtime**: only once measured-device data backs the word "calibration".

Coefficients are **literature defaults** (LightCode arXiv:2509.16443; AIMC amortisation
arXiv:2405.14978), **not measured silicon**. Claims are directional and relative, with uncertainty
bands — never "analog is N× better." See every report's *"Limits of this estimate"* section.
What stands between this release and the word *"validated"* is stated precisely in the
[**measured-validation roadmap**](docs/ROADMAP.md).

## Custom profiles, license, citing

- **Your own device profile**: start from
  [`docs/vendor_profile_template.yaml`](docs/vendor_profile_template.yaml) — every field documents
  its unit, provenance schema, and redaction level — then `--profile path/to/yours.yaml`.
- **License**: [Apache-2.0](LICENSE). **Citing**: see [`CITATION.cff`](CITATION.cff)
  (the tool plus the four load-bearing references).
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md) — note the immutable-oracle test
  convention before touching `tests/`.
