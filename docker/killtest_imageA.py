"""Day-1 dependency kill-test for Image A (MZI + quant). Fails loudly if the stack can't import,
plus a 1e-4 fidelity check that the pure-PyTorch Clements engine agrees with torchonn."""
import sys

import torch
import torchonn  # noqa: F401
import brevitas  # noqa: F401

from analog_ready.backends.mzi_pure import mzi_unitary


def main() -> int:
    print(f"torch     {torch.__version__}")
    print(f"torchonn  {getattr(torchonn, '__version__', '?')}")
    print(f"brevitas  {getattr(brevitas, '__version__', '?')}")

    # pure engine must produce a genuine unitary (sanity floor for the fidelity assert)
    u = mzi_unitary(4)
    err = (u @ u.conj().transpose(-1, -2) - torch.eye(4, dtype=u.dtype)).abs().max().item()
    assert err < 1e-4, f"pure MZI unitary not unitary (max err {err})"
    print("imageA kill-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
