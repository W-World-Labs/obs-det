#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-artifacts/omnidet_960.onnx}"
ENGINE_PATH="${2:-artifacts/omnidet_960_fp16.engine}"

trtexec \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --fp16 \
  --shapes=images:1x3x960x960 \
  --memPoolSize=workspace:4096 \
  --verbose
