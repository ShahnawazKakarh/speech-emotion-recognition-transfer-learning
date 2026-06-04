#!/usr/bin/env bash
# Download and unpack the RAVDESS speech-only subset (Audio_Speech_Actors_01-24.zip)
# Source: Zenodo (https://zenodo.org/record/1188976)
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/ravdess}"
ZIP_URL="https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
ZIP_PATH="/tmp/ravdess_speech.zip"

echo "[ravdess] Target dir: $DATA_DIR"
mkdir -p "$DATA_DIR"

if [ ! -f "$ZIP_PATH" ]; then
  echo "[ravdess] Downloading ~200MB from Zenodo..."
  curl -L --fail -o "$ZIP_PATH" "$ZIP_URL"
else
  echo "[ravdess] Zip already cached at $ZIP_PATH"
fi

echo "[ravdess] Extracting..."
unzip -q -o "$ZIP_PATH" -d "$DATA_DIR"

echo "[ravdess] Done. $(find "$DATA_DIR" -name '*.wav' | wc -l) wav files at $DATA_DIR"
