"""`analog-ready doctor --local-only` exits 0, reports the local-only posture (network/telemetry/
export disabled, redaction enabled, backend availability), and touches NO network."""
import socket


def test_doctor_local_only_exits_zero_and_reports_posture(capsys):
    from analog_ready.cli import main

    rc = main(["doctor", "--local-only"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    for token in ("network", "disabled", "telemetry", "redaction"):
        assert token in out, f"doctor output missing {token!r}"


def test_doctor_local_only_opens_no_socket(monkeypatch):
    from analog_ready.cli import main

    def _boom(*a, **k):
        raise AssertionError("doctor --local-only must not open a network socket")

    monkeypatch.setattr(socket, "socket", _boom)
    rc = main(["doctor", "--local-only"])
    assert rc == 0


def test_doctor_reports_backend_availability(capsys):
    from analog_ready.cli import main

    main(["doctor", "--local-only"])
    out = capsys.readouterr().out.lower()
    assert "mzi_pure" in out and "aimc_simple" in out
