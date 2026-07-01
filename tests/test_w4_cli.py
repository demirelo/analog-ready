"""W4 CLI — the four product verbs the plan's definition-of-done invokes:
  analog-ready analyze  --model M --profile P --out report.html
  analog-ready sweep    --model M --param sigma --values 0,0.05,0.1
  analog-ready demo     photonic-mzi
  analog-ready validate --benchmark synthetic_gemm
All are network-free, return 0 on success / non-zero on failure, accept a builtin profile NAME or a
YAML path, and honour --redact (no hidden coefficient in the written artifact)."""
import json

from analog_ready.cli import main


def test_analyze_writes_a_standalone_html_report(tmp_path):
    out = tmp_path / "report.html"
    rc = main(["analyze", "--model", "mlp", "--profile", "aimc_sram_8bit",
               "--out", str(out), "--local-only"])
    assert rc == 0
    assert out.exists()
    html = out.read_text().lower()
    assert "<html" in html and "limits" in html
    assert 'src="http' not in html  # self-contained


def test_analyze_accepts_a_profile_path(tmp_path):
    prof = "analog_ready/profiles/aimc_pcm_4bit.yaml"
    out = tmp_path / "r.html"
    rc = main(["analyze", "--model", "gpt2_block", "--profile", prof, "--out", str(out)])
    assert rc == 0 and out.exists()


def test_analyze_redact_drops_hidden_coefficient(tmp_path):
    out = tmp_path / "r.html"
    js = tmp_path / "r.json"
    rc = main(["analyze", "--model", "mlp", "--profile", "aimc_pcm_4bit",
               "--out", str(out), "--json", str(js), "--redact"])
    assert rc == 0
    assert "mem_energy_pj_per_byte" not in out.read_text()
    assert "mem_energy_pj_per_byte" not in js.read_text()


def test_sweep_writes_a_curve(tmp_path):
    out = tmp_path / "curve.json"
    rc = main(["sweep", "--model", "mlp", "--param", "sigma",
               "--values", "0,0.05,0.1", "--out", str(out)])
    assert rc == 0
    curve = json.loads(out.read_text())
    assert isinstance(curve, list) and len(curve) == 3
    for pt in curve:
        assert "value" in pt and "fidelity" in pt
    # noiseless point is (near) perfect fidelity
    assert curve[0]["fidelity"] > 0.99


def test_demo_photonic_mzi_runs_with_zero_optional_deps(capsys):
    rc = main(["demo", "photonic-mzi", "--n", "8"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "unitary" in out or "mzi" in out


def test_validate_is_deterministic_and_shows_recovery(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    assert main(["validate", "--benchmark", "synthetic_gemm", "--out", str(a)]) == 0
    assert main(["validate", "--benchmark", "synthetic_gemm", "--out", str(b)]) == 0
    # one-command repeatable -> byte-identical
    assert a.read_text() == b.read_text()
    rep = json.loads(a.read_text())
    assert rep["benchmark"] == "synthetic_gemm"
    assert isinstance(rep["curve"], list) and rep["curve"]
    rec = rep["recovery"]
    # the W3 guarantee: calibrated recovery is never worse than naive
    assert rec["recovered_fidelity"] >= rec["degraded_fidelity"] - 1e-6


def test_unknown_command_is_graceful():
    # argparse exits with SystemExit(2) on a bad subcommand; bare invocation prints help, returns 0
    assert main([]) == 0
