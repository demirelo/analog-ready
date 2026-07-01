# Optional-integration markers — these tests only run when the matching optional dep is installed
# (see Makefile test-* targets). Registered here so the core gate doesn't warn on unknown marks.
_MARKERS = ("mzi_torchonn", "aimc_aihwkit", "quant_brevitas", "quant_torchao")


def pytest_configure(config):
    for m in _MARKERS:
        config.addinivalue_line("markers", f"{m}: requires the {m} optional dependency")
