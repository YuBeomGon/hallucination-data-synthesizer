#!/usr/bin/env bash
set -euo pipefail

SPLIT=${1:-train}
RAW_SAMPLES="data/zeroth/raw_samples_${SPLIT}.jsonl"
OUTPUT="data/labels/${SPLIT}/raw_alignment.jsonl"

if [[ ! -f "$RAW_SAMPLES" ]]; then
  echo "Raw samples file not found: $RAW_SAMPLES" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

python -m src.pipeline.step_01_align \
  --config configs/default_config.yaml \
  --split "$SPLIT" \
  --raw-samples "$RAW_SAMPLES" \
  --out "$OUTPUT" \
  "${@:2}"
