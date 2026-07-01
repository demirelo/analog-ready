# Image A — MZI + quant lane. Pinned to the torch ~2.8 line because Brevitas is only tested on
# torch<=2.8; aihwkit (torch>=2.9.1) lives in a SEPARATE image (they cannot co-resolve).
FROM python:3.11-slim

RUN pip install --no-cache-dir "torch~=2.8.0" "brevitas"

# torchonn pulls a stale tensorflow-gpu transitive dep and a misnamed 'pyutils' — install it
# without deps and provide the real pyutils via the correctly-named package, avoiding the trap
# where `pip install torchonn` pulls the unrelated PyPI 'pyutils'.
RUN pip install --no-cache-dir --no-deps torchonn \
 && pip install --no-cache-dir torchonn-pyutils --no-build-isolation

COPY docker/killtest_imageA.py /killtest_imageA.py
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -e ".[quant-brevitas]" || pip install --no-cache-dir -e .

# Day-1 dependency kill-test: fail the build if the MZI+quant stack does not import.
RUN python /killtest_imageA.py
