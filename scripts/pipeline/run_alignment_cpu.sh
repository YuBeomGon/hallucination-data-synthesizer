#!/usr/bin/env bash
set -euo pipefail

python -m src.pipeline.step_01_align \
  --config configs/default_config.yaml \
  --split train \
  --limit 20
