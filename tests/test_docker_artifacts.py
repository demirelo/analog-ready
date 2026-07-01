"""Two isolated Docker images + day-1 dependency import kill-tests. The libs cannot co-resolve
(aihwkit needs torch>=2.9.1; brevitas is tested only <=2.8), so they live in separate images. The
kill-test scripts must be valid Python."""
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKER = ROOT / "docker"


def test_image_a_is_mzi_quant_torch28():
    txt = (DOCKER / "imageA.Dockerfile").read_text().lower()
    assert "2.8" in txt, "Image A should pin the torch ~2.8 line (brevitas-tested)"
    assert "brevitas" in txt
    assert "torchonn" in txt
    # the torchonn pyutils name-trap must be handled
    assert "torchonn-pyutils" in txt and "--no-build-isolation" in txt


def test_image_b_is_aimc_torch29():
    txt = (DOCKER / "imageB.Dockerfile").read_text().lower()
    assert "2.9" in txt, "Image B should require torch>=2.9.1 for aihwkit"
    assert "aihwkit" in txt


def test_killtest_scripts_exist_and_compile():
    for name in ("killtest_imageA.py", "killtest_imageB.py"):
        p = DOCKER / name
        assert p.exists(), f"missing {name}"
        py_compile.compile(str(p), doraise=True)


def test_killtest_a_imports_mzi_quant():
    txt = (DOCKER / "killtest_imageA.py").read_text()
    assert "torchonn" in txt and "brevitas" in txt


def test_killtest_b_imports_aimc():
    txt = (DOCKER / "killtest_imageB.py").read_text()
    assert "aihwkit" in txt and "InferenceRPUConfig" in txt
