#!/usr/bin/env bash
# Run all baseline experiments and produce the results table.
set -euo pipefail

CONFIGS=(
  configs/text_only_ravdess.yaml
  configs/audio_only_ravdess.yaml
  configs/multimodal_ravdess.yaml
  configs/text_only_meld.yaml
  configs/audio_only_meld.yaml
  configs/multimodal_meld.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Running: $cfg"
  echo "============================================================"
  python -m src.train --config "$cfg"
done

echo "All experiments complete. Results in outputs/."
