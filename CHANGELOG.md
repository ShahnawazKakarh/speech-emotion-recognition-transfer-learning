# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/) and [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **First real baseline result**: audio-only wav2vec2-base on RAVDESS — Test WF1 **0.7958**, UF1 **0.7843**, Accuracy **0.7951**. Per-class breakdown and training notes in `results/results.md`.

## [0.1.1] — 2026-06-04

### Added
- `CONTRIBUTING.md`, `CITATION.cff`, `CHANGELOG.md`, GitHub issue/PR templates, CI status badge in README.

### Fixed
- `train.py`: skip `ckpt_path="best"` in `--fast-dev-run` mode (Lightning disables checkpointing in that mode).
- `audio_only_ravdess.yaml`: lower LR (1e-4 → 2e-5), freeze 8/12 encoder layers, enable class weights — initial config diverged on 1.1k samples.
- `.gitignore`: anchor `/data/`, `/outputs/`, etc. to the repo root so they don't shadow `src/data/`, `src/utils/`, etc.
- CI: relax `ruff` (ignore `N812` for Lightning's `as L` convention and `B905` zip-strict).

## [0.1.0] — 2026-06-04

### Added
- Initial project scaffold: configs, data loaders (RAVDESS, MELD, IEMOCAP stub), text + audio encoders, fusion modules (concat / gated / cross-attention), Lightning training and evaluation entry points, Gradio demo, smoke tests, CI workflow.
