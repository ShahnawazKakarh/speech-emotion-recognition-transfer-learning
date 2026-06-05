# 🎙️ Speech Emotion Recognition via Transfer Learning

> Comparing **text-only (RoBERTa)**, **audio-only (wav2vec2 / WavLM / HuBERT)**, and **multimodal fusion** transfer learning approaches for Speech Emotion Recognition on **RAVDESS**, **MELD**, and **IEMOCAP**.

[![CI](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/actions/workflows/ci.yml)
[![🤗 Open in Spaces](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/Shahnawazkakarh/speech-emotion-recognition)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.x-792EE5.svg?logo=lightning)](https://lightning.ai/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-FF7C00.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🧭 Overview

Speech Emotion Recognition (SER) lives at the intersection of **paralinguistic signal** (pitch, prosody, energy) and **linguistic content** (what was actually said). Labeled emotion data is scarce — IEMOCAP has ~12 hours, RAVDESS has 1,440 clips — so **transfer learning from self-supervised speech models and large language models** is the dominant paradigm.

This repository implements and benchmarks three families of approaches on three standard datasets, so you can directly compare:

| Approach | Encoder | What it captures | Strength |
|---|---|---|---|
| **Text-only** | `RoBERTa` on Whisper transcripts | Lexical / semantic emotion cues | Cheap, leverages NLP ecosystem |
| **Audio-only** | `wav2vec2` / `WavLM` / `HuBERT` | Prosody, voice quality, paralinguistic cues | Captures *how* something is said |
| **Multimodal** | Audio + Text with cross-attention fusion | Both signals jointly | State-of-the-art on conversational SER |

> **Why this matters for NLP:** the "NLP-only" view of SER (ASR → BERT) systematically loses sarcasm, intonation, and arousal cues. This repo demonstrates *quantitatively* where text-only fails and how multimodal fusion recovers — and even *exceeds* — audio-only performance, both with random splits and with speaker-independent (publishable) splits.

---

## 📈 Results

### RAVDESS — speaker-independent split (publishable, honest numbers)

Test = actors 21–24 (4 actors, 240 samples). Val = actors 19, 20. Train = actors 1–18. **These are the numbers to report.**

| Approach | Encoder | WF1 | UF1 | Accuracy |
|---|---|---|---|---|
| **Multimodal (cross-attn)** | RoBERTa + wav2vec2 | **0.728** | **0.731** | **0.729** |
| Audio-only | wav2vec2-base | 0.659 | 0.631 | 0.667 |
| Text-only (ablation) | RoBERTa-base | 0.031 | 0.029 | 0.133 |

**Headline findings**:

- **Multimodal beats audio-only by +6.9 pp WF1** on the speaker-independent split, *larger* than the +6.2 pp gain on the random split. Cross-attention fusion generalizes to unseen speakers better than audio alone.
- **Multimodal rescues the *neutral* class** spectacularly: audio-only collapses to F1=0.21 (recall 0.125 — barely identifies neutral utterances from new actors), while multimodal reaches F1=0.78 (+57 pp).
- **Text-only remains at chance** even with the proper split — confirming the deliberate ablation: pure-NLP SER fails on acted, fixed-sentence datasets regardless of split strategy.

### RAVDESS — random split (for reference, inflated by speaker leakage)

| Approach | Encoder | WF1 | UF1 | Accuracy |
|---|---|---|---|---|
| Multimodal (cross-attn) | RoBERTa + wav2vec2 | 0.858 | 0.851 | 0.858 |
| Audio-only | wav2vec2-base | 0.796 | 0.784 | 0.795 |
| Text-only (ablation) | RoBERTa-base | 0.053 | 0.053 | 0.132 |

> The ~13 pp drop from random → speaker-independent is the **speaker-leakage premium** — a useful reference for interpreting SER papers that report random-split numbers.

Full per-class breakdown, confusion matrices, and the LR-divergence postmortem in [`results/results.md`](results/results.md).

### MELD (7-class, official splits)

| Approach | Encoder | WF1 | UF1 |
|---|---|---|---|
| Text-only (context=2) | RoBERTa-base | – | – |
| Audio-only | WavLM-base | – | – |
| Multimodal (cross-attn) | RoBERTa + WavLM | – | – |

### IEMOCAP (4-class)

⏳ Pending USC SAIL license. Loader stub at `src/data/iemocap.py`. **Unofficial HF/Kaggle copies are not used — license violation risk.**

---

## 📊 Datasets

| Dataset | Size | Emotions | Free? | Notes |
|---|---|---|---|---|
| **RAVDESS** | 1,440 clips, 24 actors | 8 (neutral, calm, happy, sad, angry, fearful, disgust, surprised) | ✅ [Zenodo](https://zenodo.org/record/1188976) | Acted; only 2 fixed sentences → text-only baseline is **deliberately weak** (great ablation) |
| **MELD** | 13,000+ utterances from Friends | 7 (anger, disgust, fear, joy, neutral, sadness, surprise) | ✅ [GitHub](https://github.com/declare-lab/MELD) | Conversational, multi-party, severe class imbalance |
| **IEMOCAP** | ~12 hours, 5 sessions, 10 actors | 4-class subset (happy, sad, angry, neutral) | 🔒 [License request](https://sail.usc.edu/iemocap/) (free, gated) | The standard SER benchmark; ~1–2 week turnaround |

Run `scripts/download_ravdess.sh` and `scripts/prepare_meld.sh` to fetch the public datasets. The IEMOCAP loader (`src/data/iemocap.py`) is a drop-in once you obtain official access.

---

## 🏗️ Architecture

```
                        ┌───────────────────────┐
                        │       Audio (.wav)    │
                        └───────────┬───────────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
              │                                            │
              ▼                                            ▼
   ┌──────────────────┐                        ┌────────────────────┐
   │  Whisper (ASR)   │                        │   wav2vec2 / WavLM │
   │   → transcript   │                        │   → audio embed    │
   └────────┬─────────┘                        └─────────┬──────────┘
            ▼                                            │
   ┌──────────────────┐                                  │
   │     RoBERTa      │                                  │
   │   → text embed   │                                  │
   └────────┬─────────┘                                  │
            │                                            │
            └───────────────┐          ┌─────────────────┘
                            ▼          ▼
                   ┌──────────────────────────┐
                   │  Cross-Attention Fusion  │
                   └────────────┬─────────────┘
                                ▼
                         ┌────────────┐
                         │ Classifier │
                         └────────────┘
```

Implementations live in:
- `src/models/text_encoder.py` — RoBERTa / DeBERTa wrappers
- `src/models/audio_encoder.py` — wav2vec2 / WavLM / HuBERT wrappers
- `src/models/fusion.py` — concat, gated, and cross-attention fusion
- `src/asr/whisper_transcribe.py` — offline transcription

---

## 🚀 Quickstart

### Install

```bash
git clone https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning.git
cd speech-emotion-recognition-transfer-learning

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,demo]"
```

### Try the demo immediately (no training needed)

```bash
python demo/gradio_app.py --pretrained
# → opens http://localhost:7860 — uses a public wav2vec2 SER model
```

> 🌐 **Live web demo**: a deployable HuggingFace Spaces scaffold lives at [`space/`](space/). See [`space/README_DEPLOY.md`](space/README_DEPLOY.md) for the 15-minute deploy guide.

### Download data and train

```bash
bash scripts/download_ravdess.sh   # ~200MB
bash scripts/prepare_meld.sh       # ~10GB (videos + audio, requires ffmpeg)

# Smoke test the pipeline (1 train + 1 val + 1 test batch, ~60s)
python -m src.train --config configs/audio_only_ravdess_si.yaml --fast-dev-run

# Speaker-independent runs (the publishable numbers)
python -m src.train --config configs/text_only_ravdess_si.yaml
python -m src.train --config configs/audio_only_ravdess_si.yaml
python -m src.train --config configs/multimodal_ravdess_si.yaml

# Random-split runs (for reference, easier to compare with older lit)
python -m src.train --config configs/multimodal_ravdess.yaml
```

### Evaluate a trained checkpoint

```bash
python -m src.evaluate \
  --checkpoint outputs/multimodal_ravdess_si/best-*.ckpt \
  --config configs/multimodal_ravdess_si.yaml
# → writes metrics.json + confusion_matrix.png to outputs/.../eval/
```

### Demo your trained checkpoint

```bash
python demo/gradio_app.py \
  --checkpoint outputs/multimodal_ravdess_si/best-*.ckpt \
  --config configs/multimodal_ravdess_si.yaml
```

---

## 📁 Repository structure

```
speech-emotion-recognition-transfer-learning/
├── configs/                  # YAML per experiment (text/audio/multimodal × ravdess/meld × random/SI)
├── src/
│   ├── data/                 # ravdess, meld, iemocap loaders + Lightning DataModule with SI splits
│   ├── models/               # text, audio, fusion encoders + LightningModule
│   ├── asr/                  # Whisper transcription
│   ├── utils/                # metrics, seeding
│   ├── train.py              # PyTorch Lightning trainer
│   └── evaluate.py           # WF1, UAR, per-class F1, confusion matrices
├── scripts/                  # data download + experiment runners
├── results/                  # benchmark tables + plots
├── demo/gradio_app.py        # interactive demo
├── tests/                    # smoke tests + data parsing + speaker-independent split
├── notebooks/                # exploration / analysis (stubs)
└── .github/workflows/ci.yml  # lint + tests on every push
```

---

## 🔬 Research notes

A few findings highlighted for reviewers / fellow researchers, all from the [`results/results.md`](results/results.md) RAVDESS write-up:

- **Multimodal advantage *grows* on speaker-independent split.** Random: multimodal +6.2 pp over audio-only. SI: multimodal **+6.9 pp**. The fusion is more useful, not less, when the test set has unseen speakers.
- **Multimodal rescues the neutral class on unseen speakers.** Audio-only F1=0.21 (recall 0.125) → multimodal F1=0.78. The text branch, despite being at chance overall, provides enough disambiguation signal to recover neutral utterances from new voices.
- **Speaker leakage inflates random-split numbers by ~13 pp.** Useful reference when comparing to literature: a 0.85 WF1 random-split number is roughly equivalent to a 0.72 WF1 on a proper speaker-independent split.
- **Multimodal helps even when one modality is uninformative.** Text-only is at chance (0.03 SI WF1) because RAVDESS has only 2 fixed sentences. Yet adding the text branch via cross-attention still lifts the audio-only baseline meaningfully.
- **Hyperparameter sensitivity in SSL fine-tuning is severe.** An early run with LR=1e-4 (the wav2vec2 *pre-training* LR) diverged — model unlearned representations. LR=2e-5 with 8/12 layers frozen reached the published numbers above. **5× LR difference → 3× performance gap.**

---

## 🛣️ Roadmap

- [x] Repo scaffolding + configs
- [x] Data loaders (RAVDESS, MELD), IEMOCAP stub
- [x] Text + audio encoders + concat / gated / cross-attention fusion
- [x] PyTorch Lightning training + evaluation
- [x] Gradio demo (pretrained + custom checkpoint modes)
- [x] CI: lint + smoke tests on Python 3.10 / 3.11
- [x] **RAVDESS random-split baselines** (text-only, audio-only, multimodal — multimodal WF1 0.858)
- [x] **RAVDESS speaker-independent baselines** (the publishable numbers — multimodal WF1 0.728)
- [x] **HuggingFace Spaces scaffold** ([`space/`](space/) — follow [`space/README_DEPLOY.md`](space/README_DEPLOY.md) to deploy)
- [ ] MELD baselines (text-only with context, audio-only, multimodal)
- [ ] IEMOCAP loader implementation (pending USC SAIL license)
- [ ] HuggingFace Spaces deployment of the demo — *push the [`space/`](space/) scaffold to your HF Space*
- [ ] Blog post on [skakarh.com](https://www.skakarh.com/blog/) with results writeup
- [ ] Cross-lingual transfer experiment (XLM-R + multilingual wav2vec2) — future research phase

---

## 📚 Citations & References

Key papers this work builds on:

- Baevski et al., *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*, NeurIPS 2020
- Chen et al., *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*, IEEE JSTSP 2022
- Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, 2019
- Poria et al., *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations*, ACL 2019
- Busso et al., *IEMOCAP: Interactive Emotional Dyadic Motion Capture Database*, LREC 2008
- Livingstone & Russo, *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*, PLOS ONE 2018

If you use this work, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub will generate the BibTeX for you from the sidebar).

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, code style, and the dataset/model extension guide. For substantial changes, open an issue first so we can align.

---

## 📄 License

MIT © [Shahnawaz Khan](https://github.com/ShahnawazKakarh)

---

## 🌐 More from SK

| | |
|---|---|
| 🌐 **Website** | [www.skakarh.com](https://www.skakarh.com) |
| 🛍️ **Products** | [www.skakarh.com/products](https://www.skakarh.com/products) |
| ✍️ **Blog** | [skakarh.com/blog](https://www.skakarh.com/blog/) |
| 🛠️ **Services** | [skakarh.com/services](https://www.skakarh.com/services/) |
| 💼 **LinkedIn** | [linkedin.com/in/skakarh](https://www.linkedin.com/in/skakarh) |
| 📦 **More projects** | [github.com/ShahnawazKakarh](https://github.com/ShahnawazKakarh) |
