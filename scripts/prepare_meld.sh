#!/usr/bin/env bash
# Download MELD and extract audio (.wav) from the .mp4 videos using ffmpeg.
# Source: https://github.com/declare-lab/MELD
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/meld}"
TAR_URL="https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
TAR_PATH="/tmp/MELD.Raw.tar.gz"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[meld] ERROR: ffmpeg is required. Install with: brew install ffmpeg (mac) or apt install ffmpeg (linux)"
  exit 1
fi

echo "[meld] Target dir: $DATA_DIR"
mkdir -p "$DATA_DIR"

if [ ! -f "$TAR_PATH" ]; then
  echo "[meld] Downloading ~10GB (this will take a while)..."
  curl -L --fail -o "$TAR_PATH" "$TAR_URL"
else
  echo "[meld] Tarball already cached at $TAR_PATH"
fi

echo "[meld] Extracting..."
tar -xzf "$TAR_PATH" -C "$DATA_DIR" --strip-components=1

# Copy CSVs to root for our loader
for split in train dev test; do
  csv_src=$(find "$DATA_DIR" -name "${split}_sent_emo.csv" | head -n1 || true)
  if [ -n "$csv_src" ]; then
    cp "$csv_src" "$DATA_DIR/${split}_sent_emo.csv"
  fi
done

# Extract audio from videos
for split in train dev test; do
  video_dir=$(find "$DATA_DIR" -type d -iname "${split}*" | grep -i video | head -n1 || true)
  audio_dir="$DATA_DIR/${split}_audio"
  mkdir -p "$audio_dir"
  if [ -n "$video_dir" ]; then
    echo "[meld] Extracting audio for $split from $video_dir → $audio_dir"
    find "$video_dir" -name "*.mp4" | while read -r mp4; do
      base=$(basename "$mp4" .mp4)
      out="$audio_dir/${base}.wav"
      if [ ! -f "$out" ]; then
        ffmpeg -hide_banner -loglevel error -y -i "$mp4" -ac 1 -ar 16000 "$out"
      fi
    done
  fi
done

echo "[meld] Done. Audio files: $(find "$DATA_DIR" -name '*.wav' | wc -l)"
