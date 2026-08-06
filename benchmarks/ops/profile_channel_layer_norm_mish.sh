#!/usr/bin/env bash
set -euo pipefail

# Application entry point for `msprof op --application=...`.
: "${PYTHON:?set PYTHON to the Python executable}"
exec "$PYTHON" "$(dirname "$0")/bench_channel_layer_norm_mish.py" \
    --batch "${BATCH:-2}" \
    --channels "${CHANNELS:-512}" \
    --time "${TIME:-50}" \
    --warmup "${WARMUP:-10}" \
    --repeats "${REPEATS:-20}"
