"""W5 · F4 — credibility. A CI workflow runs the gate on every push (ruff + pytest + a coverage
floor), and a property/fuzz test hardens the leak-free promise beyond the fixed cases: over many
randomly-shaped profiles, no `hidden` field's raw value may appear in any redacted surface."""
from __future__ import annotations

import glob
import os
import random


def _workflow_blob() -> str:
    paths = glob.glob(".github/workflows/*.yml") + glob.glob(".github/workflows/*.yaml")
    return "\n".join(open(p).read() for p in paths)


def test_ci_workflow_exists_and_runs_the_gate():
    import yaml
    paths = glob.glob(".github/workflows/*.yml") + glob.glob(".github/workflows/*.yaml")
    assert paths, "no CI workflow under .github/workflows/"
    for p in paths:
        yaml.safe_load(open(p))                         # must be valid YAML
    blob = _workflow_blob()
    assert "pytest" in blob
    assert "ruff" in blob


def test_coverage_floor_is_configured():
    blob = _workflow_blob().lower()
    pyproject = open("pyproject.toml").read().lower() if os.path.exists("pyproject.toml") else ""
    assert "cov" in blob, "CI does not collect coverage"
    assert ("fail-under" in blob) or ("fail_under" in pyproject) or ("fail-under" in pyproject), \
        "no coverage floor (fail-under) configured"


def _random_profile(rng):
    """A profile with random fields; `hidden` fields carry a distinctive 6-digit sentinel value."""
    from analog_ready.core.profile import HardwareProfile, ProfileField, REDACTIONS
    fields, sentinels = {}, []
    for i in range(rng.randint(1, 8)):
        red = rng.choice(REDACTIONS)
        if red == "hidden":
            val = float(f"{rng.randint(100000, 999999)}.{rng.randint(0, 999)}")
            sentinels.append(str(val))
        else:
            val = round(rng.uniform(-9, 9) * 10 ** rng.randint(-3, 3), 5)
        fields[f"k{i}"] = ProfileField(value=val, redaction=red)
    return HardwareProfile(name="vendor", fields=fields), sentinels


def test_redaction_fuzz_primitive_never_leaks_hidden():
    """The redaction primitive: redacted() + every repr surface, over 300 random profiles."""
    rng = random.Random(0)
    for _ in range(300):
        prof, sentinels = _random_profile(rng)
        surface = (str(prof.redacted()) + " " + repr(prof) + " "
                   + " ".join(repr(f) for f in prof.fields.values()))
        for s in sentinels:
            assert s not in surface, f"hidden value {s} leaked"


def test_redaction_fuzz_full_report_never_leaks_hidden():
    """The full render path: inject a hidden sentinel into a real profile, render the redacted
    HTML+JSON report, and assert neither the value nor the hidden key leaks."""
    from analog_ready.core.profile import ProfileField
    from analog_ready.profiles import load_profile
    from analog_ready.analyze import analyze
    from analog_ready.report import render_html, render_json
    from analog_ready.zoo import build, example_inputs
    rng = random.Random(1)
    m, x = build("mlp"), example_inputs("mlp")
    for _ in range(12):
        p = load_profile("aimc_pcm_4bit")
        secret = float(f"{rng.randint(100000, 999999)}.{rng.randint(0, 999)}")
        p.fields["secret_extra"] = ProfileField(value=secret, redaction="hidden")
        rep = analyze(m, p, inputs=x)
        for blob in (render_html(rep, redact=True), render_json(rep, redact=True)):
            assert str(secret) not in blob
            assert "secret_extra" not in blob
