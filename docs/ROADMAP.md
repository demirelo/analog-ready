# Measured-Validation Roadmap

`analog-ready` v0.1 is **simulation with literature-default coefficients** — every report says so
at the point of use. This page states, precisely, what stands between v0.1 and the word
*"validated"*, so nobody has to infer it from scattered caveats.

## Where v0.1 honestly stands

- **Physics models**: programming noise (proportional + read floor), conductance drift
  `G(t) = G0·(t/t0)^−ν` with per-device ν spread, ADC quantization (best-case auto-ranged **and**
  per-tensor fixed-range with saturation), and their end-of-life composition. Functional forms
  match the published literature (aihwkit's PCM model family; Joshi 2020; Le Gallo 2023); the
  **coefficients are literature defaults, not measured silicon**.
- **Measured references**: two published operating points are stored and compared against —
  IBM HERMES (arXiv:2212.02872, software→analog mapping drop) and Joshi 2020 (arXiv:1906.03138,
  1-day drift retention). The `pilot` command runs them through the composite **on an offline
  stand-in model**, so its output is labeled `INDICATIVE`, never a validated pass.
- **Known modeling simplifications** (each disclosed where it applies): signed-weight drift
  rather than a G⁺/G⁻ differential pair (understates drift of near-zero weights); per-sample
  rather than per-output-column ADC granularity; a static read floor folded into the weight
  rather than per-inference readout noise; single-pass DRAM lower bound.

## A0 — the credibility gate (next milestone)

Reproduce the **exact reference architectures on their exact datasets**, so the pilot compares
like with like and drops `INDICATIVE`:

1. **ResNet-9 / CIFAR-10** software baseline at HERMES's reported 93.67%, then the composite's
   predicted mapping drop vs the measured 92.81% (TDP).
2. **ResNet-32 / CIFAR-10** at Joshi's post-mapping 93.7%, then the predicted 1-day retention vs
   the reported >93.5% floor — with the caveat that the paper's figure is compensation-assisted
   (GDC + AdaBS) while the tool models raw drift, so the honest comparison bounds, not matches.

Pass criterion: predictions land within the stored per-reference tolerance bands, or the gap is
characterised and published in the report rather than hidden. Either outcome ships.

## After A0

- **v1 — Profile-Conditioned QAT Runtime**: vendor-characterised profiles drive the recovery
  recipe (noise-aware fine-tuning conditioned on the profile's own coefficients).
- **v2 — Calibration / QAT Runtime**: the word *"calibration"* appears only once measured-device
  data (vendor-provided, redaction-protected) backs it.
- **Model realism ladder**: grow the zoo beyond instrumentation-scale stand-ins (full ResNets,
  a small full transformer) so per-layer sensitivity claims rest on realistic architectures.
- **Physics refinements**: differential-pair drift, per-output-column ADC, per-inference read
  noise, partial-sum DRAM re-streaming.

No dates are promised here; sequencing is the commitment. If a milestone ships with a caveat,
the caveat ships in the report itself — that is the project's standing rule.
