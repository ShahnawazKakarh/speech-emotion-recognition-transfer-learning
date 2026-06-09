#!/usr/bin/env bash
# Download MELD and extract audio (.wav) from the .mp4 videos using ffmpeg.
# Primary mirror:  https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
# Fallback flow:   resume on failure, multiple retries, separate verification.
#
# If the umich.edu mirror is consistently slow or timing out, see the
# "ALTERNATIVE: HuggingFace mirror" comment block at the end of this file.
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/meld}"
TAR_URL="${MELD_URL:-https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz}"
TAR_PATH="${TAR_PATH:-/tmp/MELD.Raw.tar.gz}"
MAX_DOWNLOAD_ATTEMPTS=5

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[meld] ERROR: ffmpeg is required. Install with: brew install ffmpeg (mac) or apt install ffmpeg (linux)"
  exit 1
fi

echo "[meld] Target dir:   $DATA_DIR"
echo "[meld] Tarball path: $TAR_PATH"
mkdir -p "$DATA_DIR"

# --- Robust download with resume ------------------------------------------
# Strategy:
#   1. ALWAYS try curl with -C - (resume). It handles missing/partial/complete.
#   2. Retry on transient failures (the umich.edu mirror times out routinely).
#   3. ONLY verify integrity after curl returns success — never delete a
#      partial download just because gzip -t failed mid-stream.

if [ -f "$TAR_PATH" ]; then
  bytes=$(wc -c < "$TAR_PATH" | tr -d ' ')
  echo "[meld] Existing file: $bytes bytes ($(echo "scale=2; $bytes / 1073741824" | bc) GB). Will resume."
fi

download_ok=false
for attempt in $(seq 1 $MAX_DOWNLOAD_ATTEMPTS); do
  echo "[meld] Download attempt $attempt/$MAX_DOWNLOAD_ATTEMPTS …"
  # -C -                     resume if local file exists
  # --connect-timeout 30     don't hang on dead connections
  # --retry / --retry-delay  curl-level retry on HTTP 5xx etc.
  # --speed-time/limit       abort if speed stays under 50 KB/s for 60s (forces retry loop)
  if curl -L --fail \
       --connect-timeout 30 \
       --retry 3 --retry-delay 10 \
       --speed-time 60 --speed-limit 51200 \
       -C - -o "$TAR_PATH" "$TAR_URL"; then
    download_ok=true
    break
  fi
  echo "[meld] Attempt $attempt failed. Waiting 30s before retry…"
  sleep 30
done

if [ "$download_ok" != true ]; then
  echo ""
  echo "[meld] ============================================================"
  echo "[meld] FAILED after $MAX_DOWNLOAD_ATTEMPTS attempts."
  echo "[meld] The umich.edu mirror is often slow/flaky for the MELD tarball."
  echo "[meld]"
  echo "[meld] What you can do:"
  echo "[meld]  1. Re-run this script — partial download is preserved at $TAR_PATH"
  echo "[meld]     and will resume from where it left off."
  echo "[meld]  2. Run during off-peak hours (umich.edu network)."
  echo "[meld]  3. Use a download manager that handles flaky connections:"
  echo "[meld]       brew install aria2"
  echo "[meld]       aria2c -x 8 -s 8 -c -d /tmp -o MELD.Raw.tar.gz $TAR_URL"
  echo "[meld]     then re-run this script — it'll skip download and just extract."
  echo "[meld]  4. Use the HuggingFace dataset mirror (faster CDN):"
  echo "[meld]       pip install huggingface_hub"
  echo "[meld]       huggingface-cli download declare-lab/MELD-Multimodal-Dataset \\"
  echo "[meld]                                 --repo-type dataset --local-dir data/meld"
  echo "[meld]     (requires adapting paths; see scripts/prepare_meld_hf.py for"
  echo "[meld]      an alternative HF-based loader if needed.)"
  echo "[meld] ============================================================"
  exit 1
fi

# --- Verify only after download completes ---------------------------------
echo "[meld] Download complete. Verifying tarball integrity …"
if ! gzip -t "$TAR_PATH" 2>/dev/null; then
  echo "[meld] ERROR: Downloaded file failed gzip integrity check."
  echo "[meld]        File at $TAR_PATH may be corrupted by a server-side issue."
  echo "[meld]        Remove it manually and re-run:"
  echo "[meld]          rm $TAR_PATH && bash scripts/prepare_meld.sh"
  exit 1
fi
echo "[meld] Tarball verified."

# --- Extract --------------------------------------------------------------
echo "[meld] Extracting outer tarball (this also takes a while — ~10GB of video) …"
tar -xzf "$TAR_PATH" -C "$DATA_DIR" --strip-components=1

# MELD ships as a tarball-of-tarballs: after the outer extract we still have
# train.tar.gz / dev.tar.gz / test.tar.gz sitting in $DATA_DIR. Extract those.
for split in train dev test; do
  inner="$DATA_DIR/${split}.tar.gz"
  if [ -f "$inner" ]; then
    echo "[meld] Extracting inner archive: ${split}.tar.gz …"
    tar -xzf "$inner" -C "$DATA_DIR"
  fi
done

# Copy CSVs to root for our loader (skip silently if src == dst)
for split in train dev test; do
  csv_src=$(find "$DATA_DIR" -name "${split}_sent_emo.csv" | head -n1 || true)
  dst="$DATA_DIR/${split}_sent_emo.csv"
  if [ -n "$csv_src" ] && [ "$csv_src" != "$dst" ]; then
    cp "$csv_src" "$dst"
  fi
done

# --- Extract audio from videos --------------------------------------------
# MELD's video directories have inconsistent names. Map split -> dir.
# Using a function (not assoc-array) because macOS ships bash 3.2 without `declare -A`.
meld_video_dir_for() {
  case "$1" in
    train) echo "train_splits" ;;
    dev)   echo "dev_splits_complete" ;;
    test)  echo "output_repeated_splits_test" ;;
    *)     echo "" ;;
  esac
}

for split in train dev test; do
  video_dir_name=$(meld_video_dir_for "$split")
  video_dir="$DATA_DIR/$video_dir_name"
  audio_dir="$DATA_DIR/${split}_audio"
  mkdir -p "$audio_dir"

  if [ ! -d "$video_dir" ]; then
    echo "[meld] WARNING: expected video dir not found: $video_dir (skipping $split)"
    continue
  fi

  total=$(find "$video_dir" -name '*.mp4' | wc -l | tr -d ' ')
  echo "[meld] Extracting audio for $split: $total .mp4 → $audio_dir"
  count=0
  find "$video_dir" -name "*.mp4" | while read -r mp4; do
    base=$(basename "$mp4" .mp4)
    out="$audio_dir/${base}.wav"
    if [ ! -f "$out" ]; then
      ffmpeg -hide_banner -loglevel error -y -i "$mp4" -ac 1 -ar 16000 "$out" || \
        echo "[meld]   (skipped corrupted file: $mp4)"
    fi
    count=$((count + 1))
    if [ $((count % 500)) -eq 0 ]; then
      echo "[meld]   $split: $count / $total"
    fi
  done
done

echo "[meld] Done. Audio files: $(find "$DATA_DIR" -name '*.wav' | wc -l)"
