"""Day-1 dependency kill-test for Image B (AIMC). Fails loudly if aihwkit and its analog config
cannot be imported on the torch>=2.9.1 line."""
import sys

import torch
import aihwkit  # noqa: F401
from aihwkit.simulator.configs import InferenceRPUConfig


def main() -> int:
    print(f"torch    {torch.__version__}")
    print(f"aihwkit  {getattr(aihwkit, '__version__', '?')}")
    cfg = InferenceRPUConfig()  # noqa: F841  — instantiating proves the analog config loads
    print("imageB kill-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
