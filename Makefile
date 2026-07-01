# Core must pass with ZERO optional deps installed. Optional integrations are tested separately
# so an optional package breaking never breaks the core product.
.PHONY: test-core test-aimc-aihwkit test-mzi-torchonn test-quant-brevitas test-quant-torchao test

test-core:
	python3 -m pytest --cov=analog_ready --cov-fail-under=85 --tb=short -rf

test-aimc-aihwkit:
	python3 -m pytest -q -m aimc_aihwkit -rf

test-mzi-torchonn:
	python3 -m pytest -q -m mzi_torchonn -rf

test-quant-brevitas:
	python3 -m pytest -q -m quant_brevitas -rf

test-quant-torchao:
	python3 -m pytest -q -m quant_torchao -rf

test: test-core
