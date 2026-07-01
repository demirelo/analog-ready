"""W4 report — the sales artifact. Two renderers over an AnalysisReport:

  render_html(report, *, redact=False) -> str
      A SELF-CONTAINED HTML page (all CSS inline in <style>; no remote src/link/@import).
      Carries the three cost-model spines (compute / conversion / DRAM), the break-even verdict,
      sensitivity-flip summary, and an honest "Limits of this estimate" section.

  render_json(report, *, redact=False) -> str
      Machine-readable twin. redact=False: full to_dict(); redact=True: redacted() view.

Both honour redaction. The shareable view never emits a `hidden` profile coefficient, the raw
`coefficients` block, OR any value DERIVED from a hidden coefficient: the per-op energy figures
(`dram_pj` especially, = dram_bytes x the hidden mem-energy coefficient) are a back-channel from
which the hidden value can be divided straight back out, so redact=True drops the entire energy
table and the energy roll-up and renders only the qualitative verdicts that `report.redacted()`
is allowed to share.
"""
from __future__ import annotations

import html as _html
import json
import math


# --------------------------------------------------------------------------- HTML

_CSS = """
body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;color:#1a1a2e;
     background:#f8f9fa;padding:0 1rem}
h1{color:#0d1b2a;border-bottom:3px solid #4a90d9;padding-bottom:.4rem}
h2{color:#1b4f72;margin-top:2rem}
table{width:100%;border-collapse:collapse;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.1)}
th{background:#1b4f72;color:#fff;padding:.5rem .75rem;text-align:left}
td{padding:.45rem .75rem;border-bottom:1px solid #dee2e6}
tr:last-child td{border-bottom:none}
tr:nth-child(even) td{background:#f1f5f9}
.good{color:#155724;font-weight:600}
.bad{color:#721c24;font-weight:600}
.flip-yes{background:#d4edda;padding:.2rem .5rem;border-radius:3px}
.flip-no{background:#f8d7da;padding:.2rem .5rem;border-radius:3px}
.limits{background:#fff3cd;border-left:4px solid #ffc107;padding:1rem 1.2rem;
        border-radius:0 4px 4px 0;margin-top:1rem}
.limits h2{margin-top:0;color:#856404}
.limits ul{margin:.5rem 0 0 1.2rem;padding:0}
.limits li{margin:.3rem 0}
.redacted-note{background:#e7f1ff;border-left:4px solid #4a90d9;padding:.6rem 1rem;
        border-radius:0 4px 4px 0;margin:1rem 0;font-size:.9rem;color:#1b4f72}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;
              margin-bottom:1.5rem}
.summary-card{background:#fff;border:1px solid #dee2e6;border-radius:6px;padding:.8rem 1rem;
              text-align:center}
.summary-card .val{font-size:1.4rem;font-weight:700;color:#1b4f72}
.summary-card .lbl{font-size:.8rem;color:#666;margin-top:.2rem}
"""


def _fmt(v: float) -> str:
    """Format an energy in picojoules with the CORRECT SI prefix (k=1e3, M=1e6, G=1e9). 1e3 pJ is a
    nanojoule, 1e6 pJ a microjoule, 1e9 pJ a millijoule — naming them kpJ/MpJ/GpJ would be a
    1000x-mislabelled magnitude, so we step up to the natural unit at each decade."""
    if not math.isfinite(v):
        return "—"  # em dash — never print "inf pJ"/"nan pJ" in a buyer-facing report
    if v >= 1e9:
        return f"{v/1e9:.2f} mJ"
    if v >= 1e6:
        return f"{v/1e6:.2f} µJ"
    if v >= 1e3:
        return f"{v/1e3:.2f} nJ"
    return f"{v:.2f} pJ"


def _esc(s) -> str:
    return _html.escape(str(s))


def _full_rows(ops: list) -> str:
    """Internal (redact=False) rows: the full cost-model spine, one row per op."""
    rows = []
    for op in ops:
        cost = op.get("cost", {})
        be = op.get("break_even", {})
        sc = op.get("score", {})
        name = _esc(op.get("name", "?"))
        compute = _fmt(cost.get("compute_pj", 0))
        conversion = _fmt(cost.get("conversion_pj", 0))
        dram = _fmt(cost.get("dram_pj", 0))
        total = _fmt(cost.get("total_pj", 0))
        fav = be.get("favorable", False)
        fav_str = ('<span class="good">✓ favorable</span>' if fav
                   else '<span class="bad">✗ unfavorable</span>')
        limiter = _esc(sc.get("dominant_limiter", ""))
        rows.append(f"<tr><td>{name}</td><td>{compute}</td><td>{conversion}</td>"
                    f"<td>{dram}</td><td>{total}</td><td>{fav_str}</td><td>{limiter}</td></tr>")
    return "\n".join(rows)


def _redacted_rows(ops: list) -> str:
    """Shareable (redact=True) rows: qualitative verdicts ONLY — no energy figures, because the
    per-op energies are derived from a hidden cost coefficient. `ops` here is report.redacted()'s
    flat op view: {name, favorability, favorable, dominant_limiter}."""
    rows = []
    for op in ops:
        name = _esc(op.get("name", "?"))
        favval = _esc(op.get("favorability", ""))
        fav = op.get("favorable", False)
        fav_str = ('<span class="good">✓ favorable</span>' if fav
                   else '<span class="bad">✗ unfavorable</span>')
        limiter = _esc(op.get("dominant_limiter", ""))
        rows.append(f"<tr><td>{name}</td><td>{favval}</td>"
                    f"<td>{fav_str}</td><td>{limiter}</td></tr>")
    return "\n".join(rows)


def _fmt_pct(x: float) -> str:
    """A fidelity drop/gain as a legible percent, so a real 3e-5 doesn't floor to 0.0000."""
    x = max(0.0, float(x)) * 100.0
    if x == 0.0:
        return "0%"
    if x < 0.01:
        return f"{x:.1e}%"
    return f"{x:.2f}%"


def _coverage_section(coverage: dict) -> str:
    """W5 · F2 — instrumentation coverage (non-redacted view; module counts are not
    coefficient-derived)."""
    if not coverage:
        return ""
    modules_total = coverage.get("modules_total", 0)
    linear_replaced = coverage.get("linear_replaced", 0)
    conv_replaced = coverage.get("conv_replaced", 0)
    match_rate = coverage.get("attention_projection_match_rate", 1.0)
    confidence = _esc(coverage.get("coverage_confidence") or "n/a")
    unhandled = coverage.get("unhandled_linear_like", [])
    unhandled_txt = (f"{len(unhandled)} unhandled linear-like module(s) disclosed."
                     if unhandled else "No unhandled linear-like modules.")
    return f"""
<h2>Coverage — what instrumentation actually replaced</h2>
<p>Of <b>{modules_total}</b> eligible modules, <b>{linear_replaced}</b> Linear and
<b>{conv_replaced}</b> Conv2d were replaced. Attention-projection match rate:
<b>{match_rate:.2f}</b>. Coverage confidence: <b>{confidence}</b>. {unhandled_txt}</p>
"""


def _pct_or_dash(x) -> str:
    return f"{x * 100:.2f}%" if math.isfinite(x) else "—"


def _accuracy_section(accuracy: dict) -> str:
    """W6 · F1 — real task accuracy: clean vs under-noise, as percentages (non-redacted view;
    accuracy is measured on the user's own labels, not derived from a hidden cost coefficient).
    Self-labelling: the data source is named, and the drop is shown SIGNED (a noise draw that does
    not reduce accuracy must not read as 'Drop: 0%')."""
    if not accuracy:
        return ""
    clean = accuracy.get("clean", 0.0)
    under_noise = accuracy.get("under_noise", 0.0)
    drop = accuracy.get("drop", 0.0)
    sigma = accuracy.get("sigma", 0.0)
    source = _esc(accuracy.get("source", "the supplied labels"))
    if not math.isfinite(drop):
        drop_txt = "—"
    elif drop < 0:
        drop_txt = f"{drop * 100:.2f}% (no net drop — noise within draw variance)"
    else:
        drop_txt = f"{drop * 100:.2f}%"
    return f"""
<h2>Accuracy — task accuracy, clean vs under noise</h2>
<p>Top-1 accuracy on {source}: <b>{_pct_or_dash(clean)}</b> clean,
<b>{_pct_or_dash(under_noise)}</b> under noise (sigma={_esc(sigma)}). Drop: <b>{drop_txt}</b>.</p>
<p><em>Illustrative: the injected noise is a single additive-Gaussian placeholder on layer outputs
(a proxy, not a calibrated device model), and a large sigma is an extreme stress, not a realistic
operating point.</em></p>
"""


def _validation_section() -> str:
    """W6 · F4 — surface the measured-hardware validation references so a buyer actually sees them.
    These are published operating points the simulator's predictions can be CHECKED AGAINST by
    running the matching workload — not a claim that this run has been validated."""
    from analog_ready.validation import REFERENCES
    if not REFERENCES:
        return ""
    def _acc(r):
        # A "lower_bound" reference (e.g. Joshi's ">93.5% over one day") is a floor, not an exact
        # point — render it as ">=" so a reader doesn't mistake the bound for a measured value.
        prefix = "&ge;" if r.get("relation") == "lower_bound" else ""
        return f"{prefix}{r.get('reference_value', 0) * 100:.2f}%"
    rows = "\n".join(
        f"<tr><td>{_esc(r.get('name', '?'))}</td>"
        f"<td>{_acc(r)}</td>"
        f"<td>{_esc(r.get('kind', '?'))}</td>"
        f"<td>{_esc(r.get('citation', '?'))}</td></tr>"
        for r in REFERENCES
    )
    return f"""
<h2>Validation — published silicon to check against</h2>
<p>Measured / published accelerator operating points this tool's predictions can be checked against
by running the matching workload. These are <b>references</b>, not a claim that this run has been
validated against hardware.</p>
<table>
<thead><tr><th>Reference</th><th>Reported accuracy</th><th>Kind</th><th>Citation</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""


def _program_noise_section(fidelity) -> str:
    """W7 · F1 — output fidelity under the profile's characterised weight-programming noise. This is
    where the profile's HIDDEN prog/read-noise coefficients drive a buyer-visible number; only the
    derived fidelity appears (non-redacted view), never the coefficients themselves."""
    if fidelity is None:
        return ""
    shown = f"{fidelity:.4f}" if math.isfinite(fidelity) else "—"
    return f"""
<h2>Programming-noise robustness — at this profile's operating point</h2>
<p>Output fidelity under the profile's characterised weight-programming noise (proportional
programming noise + a read floor, both hidden device parameters): <b>{shown}</b>
(1.0 = unaffected). <em>Simulated literature-default model, not measured silicon; this is the
programming-noise effect alone; when the profile declares conductance drift, the End-of-life section
below composes it with drift and ADC.</em></p>
"""


def _fmt_horizon(t: float) -> str:
    """A retention time-horizon (seconds, relative to the profile's t0) as a legible label."""
    if not math.isfinite(t):
        return "—"
    if t >= 3.0e7:
        return f"{t / 3.15e7:.2g} yr"
    if t >= 2.0e6:
        return f"{t / 2.6e6:.2g} mo"
    if t >= 8.0e4:
        return f"{t / 86400.0:.2g} day"
    if t >= 3.0e3:
        return f"{t / 3600.0:.2g} hr"
    return f"{t:.2g} s"


def _retention_section(retention: list) -> str:
    """W8 · F4 — output fidelity vs elapsed time under the profile's own conductance-drift model.
    This is where the profile's drift exponent (bucket-shareable) and its device-to-device spread
    (hidden) drive a buyer-visible retention curve; only the derived fidelity numbers appear here
    (non-redacted view), never the raw drift parameters themselves."""
    if not retention:
        return ""

    def _fid(p):
        f = p.get("fidelity", 0.0)
        return f"{f:.4f}" if math.isfinite(f) else "—"

    rows = "\n".join(
        f"<tr><td>{_esc(_fmt_horizon(p.get('t', 0)))}</td><td>{_fid(p)}</td></tr>"
        for p in retention
    )
    return f"""
<h2>Retention — accuracy over time</h2>
<p>Output fidelity vs the noiseless baseline, at increasing elapsed time since programming, under
this profile's own analog conductance-drift model (see <code>analog_ready/drift.py</code>). The
first row (t0) is the programming reference — fidelity 1.0 by definition.
<em>Simulated literature-default power-law drift, not measured silicon. To first order a uniform
drift rate is a global scale that periodic recalibration largely corrects; the degradation shown is
dominated by device-to-device variability in the drift rate, which recalibration cannot undo.
Recalibration itself is not modelled here.</em></p>
<table>
<thead><tr><th>Elapsed time</th><th>Fidelity</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""


def _adc_readout_section(adc_fidelity) -> str:
    """W9 · F3/F4 — output fidelity under the profile's own ADC readout precision (enob_avail
    effective bits). This is where the profile's ENOB — already priced by the cost-model's
    precision gate — drives a buyer-visible signal-fidelity number; non-redacted view only, a
    derived local artifact never in the shareable view."""
    if adc_fidelity is None:
        return ""
    shown = f"{adc_fidelity:.4f}" if math.isfinite(adc_fidelity) else "—"
    return f"""
<h2>ADC readout precision</h2>
<p>Output fidelity with every eligible layer's output read back through a simulated ADC at this
profile's own ENOB (effective number of bits): <b>{shown}</b> (1.0 = unaffected). <em>Uses the
profile's literature-default ENOB, not measured silicon; this is the ADC effect alone (composed with
the others in the End-of-life section when the profile declares drift). This is an <b>optimistic</b>
number: the model auto-ranges to each sample's own peak
(an idealised ADC reference), so a real fixed-range ADC is usually worse — often materially so for
outlier / high-dynamic-range activations (though not a hard bound; see the realistic fixed-range
number below).</em></p>
"""


def _adc_fixed_section(fixed_fidelity, saturation) -> str:
    """W12 — the REALISTIC fixed-range ADC (per-tensor full-scale, values beyond it saturate),
    reported next to the best-case auto number above so a buyer sees the bracket. Non-redacted view
    only; derived local artifacts, never shareable."""
    if fixed_fidelity is None:
        return ""
    fid = f"{fixed_fidelity:.4f}" if math.isfinite(fixed_fidelity) else "—"
    sat = f"{saturation * 100:.2f}%" if (saturation is not None and math.isfinite(saturation)) else "—"
    return f"""
<h2>ADC readout — per-tensor fixed-range (clipped)</h2>
<p>Output fidelity under a <b>fixed</b>-range ADC (one full-scale per output tensor, covering the
profile's <code>adc_full_scale</code> fraction of the signal peak — values beyond it saturate):
<b>{fid}</b> (1.0 = unaffected), with <b>{sat}</b> of the signal saturating. <em>A more realistic
counterpart to the best-case auto-ranged number above — usually below it, but not a hard bound: a
tighter per-tensor range can occasionally beat a wide per-sample one on multi-axis (e.g.
attention/sequence) outputs. This is a <b>per-tensor clipped-ADC stress</b>, not an absolute
calibration: the full-scale is still derived from the signal's own peak (a fraction of it), not a
measured device reference, and it clips per-tensor rather than per-output-column.</em></p>
"""


def _end_of_life_section(end_of_life: list) -> str:
    """W10 capstone — combined-effect output fidelity vs time: drift + programming noise + ADC
    readout composed from one profile. Non-redacted view only; a derived local artifact never in the
    shareable view. Honest that it inherits every component caveat, including the best-case ADC."""
    if not end_of_life:
        return ""

    def _fid(p):
        f = p.get("fidelity", 0.0)
        return f"{f:.4f}" if math.isfinite(f) else "—"

    rows = "\n".join(
        f"<tr><td>{_esc(_fmt_horizon(p.get('t', 0)))}</td><td>{_fid(p)}</td></tr>"
        for p in end_of_life
    )
    return f"""
<h2>End-of-life — all effects combined</h2>
<p>Output fidelity vs the clean baseline with conductance drift, weight-programming noise, and ADC
readout precision <b>composed together</b> from this profile — the honest "what does the model
actually produce on this chip after this long" number, not any single effect in isolation.
<em>Read it directionally, not as a rigorous bound: it inherits every component caveat
(literature-default, not-measured noise / drift / ENOB, and the best-case auto-ranged ADC) and is a
noisy mean over seeded draws, so a real deployment is likely worse.</em></p>
<table>
<thead><tr><th>Elapsed time</th><th>Combined fidelity</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""


def _sensitivity_layers_section(layers: list) -> str:
    """W5 · F1 — ranked per-layer fragility (most-fragile first)."""
    if not layers:
        return ""
    rows = "\n".join(
        f"<tr><td>{_esc(layer.get('name', '?'))}</td>"
        f"<td>{layer.get('fidelity', 0):.4f}</td>"
        f"<td>{_fmt_pct(layer.get('fidelity_drop', 0))}</td></tr>"
        for layer in layers
    )
    return f"""
<h2>Sensitivity — most-fragile layers</h2>
<p>Each row injects noise on ONE layer's output only — <b>scaled to that layer's output
magnitude</b>, so the ranking reflects sensitivity rather than output scale — holding every other
layer exact, and measures whole-model output fidelity vs the noiseless baseline. Most fragile
first.</p>
<table>
<thead><tr><th>Layer</th><th>Fidelity</th><th>Fidelity drop</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""


def _recovery_section(recovery: list) -> str:
    """W5 · F3 — recovery recipe per fragile Linear layer (degraded -> recovered fidelity)."""
    if not recovery:
        return ""

    def _row(rec):
        deg = rec.get("degraded_fidelity", 0.0)
        rec_f = rec.get("recovered_fidelity", 0.0)
        gain = rec_f - deg
        gain_txt = _fmt_pct(gain) if gain > 1e-6 else "—  (already optimal)"
        return (f"<tr><td>{_esc(rec.get('name', '?'))}</td><td>{_esc(rec.get('bits', '?'))}</td>"
                f"<td>{deg:.4f}</td><td>{rec_f:.4f}</td><td>{gain_txt}</td></tr>")

    rows = "\n".join(_row(r) for r in recovery)
    return f"""
<h2>Recovery — what recovers accuracy</h2>
<p>Per-layer calibrated quantisation recipe for the most fragile <b>Linear</b> layers (Conv2d
fragility is reported in Sensitivity above but is not yet recoverable). Recovered fidelity is never
worse than the naive degraded baseline.</p>
<table>
<thead><tr><th>Layer</th><th>Bits</th><th>Degraded fidelity</th><th>Recovered fidelity</th>
<th>Gain</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""


def render_html(report, *, redact: bool = False) -> str:  # noqa: ANN001
    """Self-contained HTML page for the given AnalysisReport. redact=True produces the SHAREABLE
    view: qualitative verdicts only, with no energy figure from which a hidden coefficient could be
    recovered."""
    n_ops = len(report.ops)
    n_fav = sum(1 for o in report.ops if o.get("break_even", {}).get("favorable", False))
    flip = report.has_verdict_flip()
    flip_cls = "flip-yes" if flip else "flip-no"
    flip_txt = "Yes" if flip else "No"

    # ---- summary cards (the energy roll-up is itself coefficient-derived -> withheld when redacting)
    cards = [
        f'<div class="summary-card"><div class="val">{n_fav}/{n_ops}</div>'
        f'<div class="lbl">Ops favorable (break-even)</div></div>',
    ]
    if not redact:
        cards.append(
            f'<div class="summary-card"><div class="val">{_fmt(report.total_pj)}</div>'
            f'<div class="lbl">Total estimated energy</div></div>')
    cards.append(
        f'<div class="summary-card"><div class="val"><span class="{flip_cls}">{flip_txt}</span>'
        f'</div><div class="lbl">Sensitivity / verdict flip</div></div>')
    cards_html = "".join(cards)

    # ---- op table: full spine internally, qualitative-only when shareable
    if redact:
        header = ("<th>Op</th><th>Favorability</th><th>Break-even</th>"
                  "<th>Dominant limiter</th>")
        op_rows = _redacted_rows(report.redacted().get("ops", []))
        spine_intro = ('This is the <strong>shareable (redacted)</strong> view: per-op energy '
                       'figures are withheld because they are derived from a private cost '
                       'coefficient. Qualitative break-even verdicts only.')
        redacted_banner = ('<div class="redacted-note">Redacted envelope — no raw or '
                           'coefficient-derived values leave the tool.</div>')
    else:
        header = ("<th>Op</th><th>Compute</th><th>Conversion</th><th>DRAM</th><th>Total</th>"
                  "<th>Break-even</th><th>Dominant limiter</th>")
        op_rows = _full_rows(report.ops)
        spine_intro = ('Energy components: <b>compute</b> (analog MACs) + <b>conversion</b> '
                       '(ADC/DAC) + <b>DRAM</b> movement. Break-even verdict per op.')
        redacted_banner = ""

    # ---- sensitivity section (ENOB is an integer sweep; render it as an integer)
    sweep = report.sweep
    if sweep:
        flip_at = next((p["value"] for i, p in enumerate(sweep)
                        if i > 0 and p["n_favorable"] != sweep[i - 1]["n_favorable"]), None)
        if flip_at is not None:
            shown = f"{int(flip_at)}" if float(flip_at).is_integer() else f"{flip_at}"
            sens_text = (f'The verdict <b>flips</b> at {_esc(shown)} ENOB. '
                         f'Points surveyed: {len(sweep)}.')
        else:
            sens_text = f'No flip detected across the ENOB sweep ({len(sweep)} points).'
    else:
        sens_text = "No sensitivity sweep available."

    profile_name = _esc(report.profile_name)
    model_name = _esc(report.model)

    # ---- W5 sections: coverage / per-layer sensitivity / recovery — internal (non-redacted) view
    # only. These carry layer names/counts/fidelities, never profile coefficients, but we keep the
    # shareable surface unchanged from W4 (qualitative verdicts only) by gating on `redact`.
    extra_sections = ""
    if not redact:
        extra_sections = (
            _accuracy_section(getattr(report, "accuracy", None))
            + _program_noise_section(getattr(report, "program_noise_fidelity", None))
            + _retention_section(getattr(report, "retention", None))
            + _adc_readout_section(getattr(report, "adc_fidelity", None))
            + _adc_fixed_section(getattr(report, "adc_fixed_fidelity", None),
                                 getattr(report, "adc_saturation", None))
            + _end_of_life_section(getattr(report, "end_of_life", None))
            + _coverage_section(getattr(report, "coverage", None))
            + _sensitivity_layers_section(getattr(report, "layers", None))
            + _recovery_section(getattr(report, "recovery", None))
            + _validation_section()
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Analog-Ready Report — {model_name} / {profile_name}</title>
<style>
{_CSS}
</style>
</head>
<body>
<h1>Analog-Ready Report</h1>
<p><strong>Model:</strong> {model_name} &nbsp;|&nbsp;
   <strong>Profile:</strong> {profile_name}</p>
{redacted_banner}
<div class="summary-grid">
{cards_html}
</div>

<h2>Cost-Model Spine — per-op breakdown</h2>
<p>{spine_intro}</p>
<table>
<thead>
<tr>
{header}
</tr>
</thead>
<tbody>
{op_rows}
</tbody>
</table>

<h2>Sensitivity — where the verdict flips</h2>
<p>{sens_text}</p>
{extra_sections}
<div class="limits">
<h2>Limits of this estimate</h2>
<ul>
<li>All energy estimates use <em>literature defaults</em>, not measured silicon. Uncertainty: ±50%
    on energy per MAC/conversion, ±2x on DRAM movement.</li>
<li>DRAM is a <em>single-pass lower bound</em> (weights once + activations). Partial-sum
    re-streaming and tile weight re-loads are excluded (v0.1 refinement).</li>
<li>Converter sharing is pessimistic (no inter-tile DAC sharing). An optimistic shared-DAC design
    would lower conversion energy.</li>
<li>Latency estimates are placeholders and do not enter any verdict.</li>
<li>ADC readout fidelity auto-ranges per sample (an optimistic, idealised ADC reference); a real
    fixed-range ADC is usually worse, especially for outlier-heavy activations — though not a hard
    bound. It also quantises per-sample rather than per output column.</li>
<li>Weight-programming noise (prog/read sigma) and conductance drift over time both use
    <em>literature defaults</em>, not measured silicon; the programming-noise model is proportional
    noise + a static read floor, and neither models 1/f noise, drift-compensation/recalibration, or
    cycling. Drift is applied to the <em>signed weight</em>, not a G+/G&minus; differential
    conductance pair — this understates drift of near-zero weights (whose real pair would drift
    them away from zero). Each effect is reported per-effect above; when the profile declares
    conductance drift, the End-of-life section composes them.</li>
<li>Results apply only to the analysed workload shape; out-of-distribution inputs may shift
    the break-even point.</li>
</ul>
</div>

</body>
</html>"""


# --------------------------------------------------------------------------- JSON

def render_json(report, *, redact: bool = False) -> str:  # noqa: ANN001
    """JSON string for the AnalysisReport. redact=True: calls report.redacted() (drops hidden
    fields + coefficients); redact=False: calls report.to_dict() (full internal view)."""
    d = report.redacted() if redact else report.to_dict()
    return json.dumps(d, indent=2, sort_keys=True)
