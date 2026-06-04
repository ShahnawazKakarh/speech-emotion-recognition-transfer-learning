# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/) and [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added — RAVDESS baselines complete
- **Multimodal cross-attention (RoBERTa + wav2vec2)**: Test WF1 **0.858**, UF1 **0.851**, Accuracy **0.858**. Beats audio-only by +6.2 pp WF1; gains concentrate on previously-weakest classes (neutral +10.2, sad +11.0, disgust +11.6).
- **Audio-only wav2vec2-base**: Test WF1 **0.796**, UF1 **0.784**, Accuracy **0.795**.
- **Text-only RoBERTa-base** (deliberate ablation): Test WF1 **0.053**, near chance — confirms text-only fails on RAVDESS's 2-fixed-sentence setup, demonstrating *why* multimodal SER matters.

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
