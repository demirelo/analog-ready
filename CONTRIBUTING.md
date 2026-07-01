# Contributing to analog-ready

Thanks for your interest. This project values **honesty about what is modeled** above all else —
a contribution that makes the tool more capable but less honest will be declined.

## What is supported

- **Core simulation ladder + builtin profiles** (`aimc_pcm_4bit`, `aimc_sram_8bit`,
  `aimc_reram_6enob`, `photonic_clements_mzi`): fully supported. Bug reports with a minimal
  reproduction are very welcome.
- **Custom / measured device profiles**: the schema is documented in
  [`docs/vendor_profile_template.yaml`](docs/vendor_profile_template.yaml) and loads via
  `--profile path/to/yours.yaml`. Integrating *measured-silicon* coefficients (calibration,
  validation against your hardware) is beyond what issues can cover — open an issue describing
  your device class and we'll discuss the right path.
- **Optional backends** (aihwkit, torchonn, brevitas, torchao): best-effort. The core must never
  break because an optional dependency broke (`make test-core` runs with zero optional deps).

## Development setup

```bash
pip install -e . && pip install ruff pytest pytest-cov
make test-core            # full suite + the same 85% coverage gate CI enforces
ruff check analog_ready tests
```

CI runs the suite on Python 3.9 / 3.11 / 3.12. Note the pinned `numpy<2` (torch 2.2 ABI);
`analog-ready doctor` will warn if your environment violates it.

## The immutable-oracle convention (please read before touching `tests/`)

Files named `tests/test_w<N>_*.py` are **immutable acceptance oracles**: once merged, they are
never edited — not for lint, not for style, not to make a change pass. If your change breaks one,
the change is wrong (or you have found a genuinely wrong oracle: open an issue making that case
rather than editing it; lint exceptions go in `pyproject.toml` per-file-ignores). New behavior
gets a **new** oracle file.

## Pull requests

1. Write the failing test first, in a new test file.
2. Keep physics/math claims cited: a docstring that states a model's behavior must match what the
   code does, and a coefficient must name its source. Disclosures of what is *not* modeled belong
   at the point of use, not in a footnote.
3. `make test-core` and `ruff check analog_ready tests` must pass.
4. PRs that change any redaction-relevant path must state how they verified no `hidden`
   coefficient (or value derived from one) can reach a shareable artifact.
