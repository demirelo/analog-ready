# Image B — AIMC lane. aihwkit 1.1.0 requires torch>=2.9.1 (CPU wheel by default), which is
# incompatible with Brevitas's tested ceiling — hence a separate image from Image A.
FROM python:3.11-slim

RUN pip install --no-cache-dir "torch>=2.9.1"
RUN pip install --no-cache-dir aihwkit

COPY docker/killtest_imageB.py /killtest_imageB.py
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -e ".[aimc]" || pip install --no-cache-dir -e .

# Day-1 dependency kill-test: fail the build if the AIMC stack does not import.
RUN python /killtest_imageB.py
