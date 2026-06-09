# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/) and [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added — MELD baselines (text / audio / multimodal) 🎹
- **Three new MELD-7 baselines** trained and evaluated end-to-end:
  - Text-only RoBERTa-base + 2-utterance context — **WF1 0.609** (top of literature range 0.55–0.62)
  - Audio-only WavLM-base — WF1 0.357 (literature range 0.30–0.45; class-collapse on surprise/fear/disgust as expected on MELD)
  - Multimodal RoBERTa + WavLM cross-attention — WF1 0.590
- **Headline finding**: on MELD, text-only beats multimodal by 1.9 pp WF1 — the *opposite* of RAVDESS where multimodal won by 6.9 pp. This isn't a regression: it's the modality-complementarity phenomenon that gives a strong two-dataset story (see `results/results.md` § "Cross-experiment observations").
- New per-class F1 tables and a full "Detailed results — MELD" section in `results/results.md`.
- Long-form writeups under `docs/writeups/`: SEO blog post, IEEE-style short paper, LinkedIn announcement.

### Added — MELD data pipeline
- `scripts/prepare_meld.sh`: robust download (resume, gzip integrity check, retry loop, speed-floor abort), inner-tarball extraction, hardcoded video-dir mapping (bash 3.2 / macOS compatible), per-500-file ffmpeg progress, graceful skip of corrupted .mp4s.
- `scripts/check_meld_audio.py`: standalone audit script that scans all three splits and reports missing / empty .wav files vs the CSV labels.
- `src/data/meld.py`: loader now filters rows whose .wav is missing or 0-byte at `__init__` time and prints a clean count (`dropped N rows with missing or empty audio`).

### Fixed — MELD config tuning
- `configs/audio_only_meld.yaml`: lowered LR from 1e-4 (the WavLM *pre-training* rate, prone to divergence on small fine-tuning runs) to 2e-5. Reuses the lesson learned from the RAVDESS audio-only divergence saga.
- `configs/multimodal_meld.yaml`: MPS-friendly memory profile — `batch_size: 2` with `accumulate_grad_batches: 8` (effective batch 16 unchanged), `max_audio_seconds` 8.0 → 5.0, `max_text_tokens` 128 → 96. Original config OOM'd at epoch 0, batch 154/1249 on a 24GB unified-memory M-series Mac. New config completes 10 epochs cleanly.

### Added — HuggingFace Spaces live demo 🎉
- **Live deployment**: [huggingface.co/spaces/Shahnawazkakarh/speech-emotion-recognition](https://huggingface.co/spaces/Shahnawazkakarh/speech-emotion-recognition) is now public, running `superb/wav2vec2-base-superb-er` on a free CPU tier.
- New `🤗 Open in Spaces` badge in the main README.

### Fixed — HuggingFace Spaces deployment
- Pinned `huggingface_hub<1.0` in `space/requirements.txt` so that `HfFolder` (removed in `huggingface_hub` v1.0 but still imported by older Gradio) remains available.
- Explicitly declared `torch>=2.0,<3.0` and `torchaudio` in `space/requirements.txt` after the HF Spaces base image stopped pre-installing them for this Gradio version.
- Replaced `gr.Label(num_top_classes=5)` with `gr.Markdown` rendering manual probability bars to sidestep `gradio_client.utils.get_type` schema introspection.
- Added a defensive monkeypatch in `space/app.py` that wraps `gradio_client.utils.get_type` and `_json_schema_to_python_type` to handle bool schemas gracefully — fully eliminates the well-known `TypeError: argument of type 'bool' is not iterable` issue from `/api/info` regardless of Gradio version.

### Added — HuggingFace Spaces scaffold
- New `space/` directory with a ready-to-deploy Gradio app for HuggingFace Spaces (`space/app.py`), Space frontmatter README (`space/README.md`), Space `requirements.txt`, and a step-by-step deploy guide (`space/README_DEPLOY.md`).
- Default app runs `superb/wav2vec2-base-superb-er` for instant inference; doc explains how to swap in a custom-trained checkpoint via HF Model Hub.

### Added — RAVDESS speaker-independent baselines (the publishable numbers)
- **Multimodal cross-attention SI** (test = actors 21-24): Test WF1 **0.728**, UF1 **0.731**, Accuracy **0.729**. Beats audio-only SI by **+6.9 pp WF1** (larger margin than on the random split — multimodal generalizes to unseen speakers better than audio alone).
- **Audio-only wav2vec2-base SI**: Test WF1 **0.659**, UF1 **0.631**, Accuracy **0.667**.
- **Text-only RoBERTa-base SI**: Test WF1 **0.031** — still at chance, as expected. Model collapsed to predicting `calm` for every input.
- **Key finding**: multimodal F1 on the *neutral* class jumped from 0.21 (audio-only) to 0.78 (+57 pp) on unseen speakers — the text branch acts as a strong disambiguator even though it's at chance overall.
- New `split_strategy: speaker_independent` option in `SERDataModule` with configurable actor lists. Three new configs: `*_ravdess_si.yaml` for text/audio/multimodal.
- New test `test_speaker_independent_split_disjoint` verifies no actor leaks across train/val/test.

### Added — README polish
- Updated LinkedIn URL to `linkedin.com/in/skakarh`.
- Added `skakarh.com/products` link in the SK footer.

### Added — RAVDESS random-split baselines (earlier)
- **Multimodal cross-attention (RoBERTa + wav2vec2)**: Test WF1 **0.858**, UF1 **0.851**, Accuracy **0.858**. Beats audio-only by +6.2 pp WF1.
- **Audio-only wav2vec2-base**: Test WF1 **0.796**, UF1 **0.784**, Accuracy **0.795**.
- **Text-only RoBERTa-base** (deliberate ablation): Test WF1 **0.053**, near chance — confirms text-only fails on RAVDESS's 2-fixed-sentence setup.

### Fixed
- MPS "Unaligned blit request" bug when loading RoBERTa checkpoints via `load_from_checkpoint`. Now load to CPU first with `map_location="cpu"`, then let Lightning move to MPS during `.fit()` / `.test()`. Affects `src/evaluate.py` and `demo/gradio_app.py`.
- Demo device mismatch: input audio tensor was on CPU while model was on MPS. Now explicitly moves audio to model's device.
- Multimodal RAVDESS config: applied lessons learned from audio-only run (LR=2e-5, freeze 8/12 audio encoder layers, class-weighted CE loss).

## [0.1.1] — 2026-06-04

### Added
- `CONTRIBUTING.md`, `CITATION.cff`, `CHANGELOG.md`, GitHub issue/PR templates, CI status badge in README.

### Fixed
- `train.py`: skip `ckpt_path="best"` in `--fast-dev-run` mode (Lightning disables checkpointing in that mode); added `--patience` and `--skip-test` flags.
- `audio_only_ravdess.yaml`: lower LR (1e-4 → 2e-5), freeze 8/12 encoder layers, enable class weights — initial config diverged on 1.1k samples.
- `.gitignore`: anchor `/data/`, `/outputs/`, etc. to the repo root so they don't shadow `src/data/`, `src/utils/`, etc.
- CI: relax `ruff` (ignore `N812` for Lightning's `as L` convention and `B905` zip-strict).

## [0.1.0] — 2026-06-04

### Added
- Initial project scaffold: configs, data loaders (RAVDESS, MELD, IEMOCAP stub), text + audio encoders, fusion modules (concat / gated / cross-attention), Lightning training and evaluation entry points, Gradio demo, smoke tests, CI workflow.
