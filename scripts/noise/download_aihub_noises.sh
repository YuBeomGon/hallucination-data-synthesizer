#!/usr/bin/env bash
# Download and extract AI Hub noise datasets into assets/noises.
# Defaults to the 도시 소리 데이터 (dataset 585) resource unless overridden.

if ! command -v aihubshell >/dev/null 2>&1; then
    echo "aihubshell CLI is required but not found." >&2
    exit 1
fi

if ! command -v 7z >/dev/null 2>&1; then
    echo "7z is required to extract archives but not found." >&2
    exit 1
fi

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "Please run this script with bash (e.g., 'bash scripts/download_aihub_noises.sh')." >&2
    exit 1
fi

set -euo pipefail

# Default dataset/resource pairs. Override by providing arguments:
#   scripts/download_aihub_noises.sh DATASET_KEY RESOURCE_ID [...]
DEFAULT_DATASET_KEY="585"
DEFAULT_RESOURCE_IDS=(
    "4C228107-8608-482B-AC25-E2E91F17E122"
)

TARGET_DIR=${TARGET_DIR:-"assets/noises"}

if [[ $# -ge 2 ]]; then
    DATASET_KEY=$1
    shift
    RESOURCE_IDS=("$@")
else
    DATASET_KEY=$DEFAULT_DATASET_KEY
    RESOURCE_IDS=("${DEFAULT_RESOURCE_IDS[@]}")
fi

mkdir -p "$TARGET_DIR"
pushd "$TARGET_DIR" >/dev/null

for resource_id in "${RESOURCE_IDS[@]}"; do
    echo "[dataset:$DATASET_KEY resource:$resource_id] Downloading with aihubshell..."
    aihubshell -mode d -datasetkey "$DATASET_KEY" -resourcekey "$resource_id"
done

echo "Scanning for ZIP archives to extract..."
while IFS= read -r -d '' zip_file; do
    dir=$(dirname "$zip_file")
    base=$(basename "$zip_file")
    echo "[$zip_file] Extracting..."
    if (cd "$dir" && 7z x -aos "$base"); then
        echo "[$zip_file] Extraction complete. Removing archive..."
        (cd "$dir" && rm -f "$base" "${base%.zip}.z"*)
    else
        echo "[$zip_file] Extraction failed. Archive retained." >&2
    fi
done < <(find . -type f -name '*.zip' -print0)

popd >/dev/null
