# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/) and [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
