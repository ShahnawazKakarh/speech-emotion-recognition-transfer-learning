# 🎙️ Speech Emotion Recognition via Transfer Learning

> Comparing **text-only (RoBERTa)**, **audio-only (wav2vec2 / WavLM / HuBERT)**, and **multimodal fusion** transfer learning approaches for Speech Emotion Recognition on **RAVDESS**, **MELD**, and **IEMOCAP**.

[![CI](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/actions/workflows/ci.yml)
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

> **Why this matters for NLP:** the "NLP-only" view of SER (ASR → BERT) systematically loses sarcasm, intonation, and arousal cues. This repo demonstrates *quantitatively* where text-only fails and how multimodal fusion recovers — and even *exceeds* — audio-only performance.

---

## 📈 Results

### RAVDESS (8-class, random 70/10/20 split)

| Approach | Encoder | WF1 | UF1 | Accuracy |
|---|---|---|---|---|
| **Multimodal (cross-attn)** | RoBERTa + wav2vec2 | **0.858** | **0.851** | **0.858** |
| Audio-only | wav2vec2-base | 0.796 | 0.784 | 0.795 |
| Text-only (ablation) | RoBERTa-base | 0.053 | 0.053 | 0.132 |

**Key finding**: multimodal cross-attention beats audio-only by **+6.2 pp WF1**, with the gains concentrated on the previously-weakest classes — *neutral* (+10.2 pp), *sad* (+11.0 pp), *disgust* (+11.6 pp). The text branch is at chance on its own (0.053), yet still contributes useful regularization signal when fused with audio. Full per-class breakdown and the LR-divergence postmortem in [`results/results.md`](results/results.md).

> ⚠️ Current split is random (same actor in train/test) — a speaker-independent split is in the roadmap and will give the honest published-style numbers (expect ~5–10 pp lower).

### MELD (7-class, official splits)

| Approach | Encoder | WF1 | UF1 |
|---|---|---|---|
| Text-only (context=2) | RoBERTa-base | – | – |
| Audio-only | WavLM-base | – | – |
| Multimodal (cross-attn) | RoBERTa + WavLM | – | – |

### IEMOCAP (4-class)

⏳ Pending dataset access — loader stub at `src/data/iemocap.py`.

---

## 📊 Datasets

| Dataset | Size | Emotions | Free? | Notes |
|---|---|---|---|---|
| **RAVDESS** | 1,440 clips, 24 actors | 8 (neutral, calm, happy, sad, angry, fearful, disgust, surprised) | ✅ [Zenodo](https://zenodo.org/record/1188976) | Acted; only 2 fixed sentences → text-only baseline is **deliberately weak** (great ablation) |
| **MELD** | 13,000+ utterances from Friends | 7 (anger, disgust, fear, joy, neutral, sadness, surprise) | ✅ [GitHub](https://github.com/declare-lab/MELD) | Conversational, multi-party, severe class imbalance |
| **IEMOCAP** | ~12 hours, 5 sessions, 10 actors | 4-class subset (happy, sad, angry, neutral) | 🔒 [License request](https://sail.usc.edu/iemocap/) (free, gated) | The standard SER benchmark; ~1–2 week turnaround |

Run `scripts/download_ravdess.sh` and `scripts/prepare_meld.sh` to fetch the public datasets. The IEMOCAP loader (`src/data/iemocap.py`) is a drop-in once you obtain access.

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

### Download data and train

```bash
bash scripts/download_ravdess.sh   # ~200MB
bash scripts/prepare_meld.sh       # ~10GB (videos + audio, requires ffmpeg)

# Smoke test the pipeline (1 train + 1 val + 1 test batch, ~60s)
python -m src.train --config configs/audio_only_ravdess.yaml --fast-dev-run

# Real run
python -m src.train --config configs/audio_only_ravdess.yaml

# Multimodal cross-attention (the headline configuration)
python -m src.train --config configs/multimodal_ravdess.yaml
```

### Evaluate a trained checkpoint

```bash
python -m src.evaluate \
  --checkpoint outputs/multimodal_ravdess/best-*.ckpt \
  --config configs/multimodal_ravdess.yaml
# → writes metrics.json + confusion_matrix.png to outputs/.../eval/
```

### Demo your trained checkpoint

```bash
python demo/gradio_app.py \
  --checkpoint outputs/multimodal_ravdess/best-*.ckpt \
  --config configs/multimodal_ravdess.yaml
```

---

## 📁 Repository structure

```
speech-emotion-recognition-transfer-learning/
├── configs/                  # YAML per experiment (text/audio/multimodal × ravdess/meld)
├── src/
│   ├── data/                 # ravdess, meld, iemocap loaders + Lightning DataModule
│   ├── models/               # text, audio, fusion encoders + LightningModule
│   ├── asr/                  # Whisper transcription
│   ├── utils/                # metrics, seeding
│   ├── train.py              # PyTorch Lightning trainer
│   └── evaluate.py           # WF1, UAR, per-class F1, confusion matrices
├── scripts/                  # data download + experiment runners
├── results/                  # benchmark tables + plots
├── demo/gradio_app.py        # interactive demo
├── tests/                    # smoke tests + data parsing
├── notebooks/                # exploration / analysis (stubs)
└── .github/workflows/ci.yml  # lint + tests on every push
```

---

## 🔬 Research notes

A few findings highlighted for reviewers / fellow researchers, all from the [`results/results.md`](results/results.md) RAVDESS write-up:

- **Multimodal helps even when one modality is uninformative.** Text-only is at chance (0.053 WF1) because RAVDESS has only 2 fixed sentences. Yet adding the text branch via cross-attention lifts the audio-only baseline from 0.796 to 0.858 WF1. The text branch likely acts as a soft regularizer or weak per-input prior rather than a semantic signal contributor.
- **Multimodal gains concentrate on previously-weakest classes.** Neutral, sad, and disgust each gain +10 pp or more; surprised gains only +1.5 pp. This is the signature of a useful intervention, not random noise.
- **Hyperparameter sensitivity in SSL fine-tuning is severe.** An early run with LR=1e-4 (the wav2vec2 *pre-training* LR) diverged — model unlearned representations, EarlyStopping triggered at epoch 4, test WF1 0.27. LR=2e-5 with 8/12 layers frozen reached 0.796. **5× LR difference → 3× performance gap.**
- **MELD's class imbalance is going to dominate when we get there** (~48 % neutral utterances). UF1 will be the more honest metric than WF1, and the confusion matrices will tell the real story.

---

## 🛣️ Roadmap

- [x] Repo scaffolding + configs
- [x] Data loaders (RAVDESS, MELD), IEMOCAP stub
- [x] Text + audio encoders + concat / gated / cross-attention fusion
- [x] PyTorch Lightning training + evaluation
- [x] Gradio demo (pretrained + custom checkpoint modes)
- [x] CI: lint + smoke tests on Python 3.10 / 3.11
- [x] **RAVDESS audio-only baseline — WF1 0.796**
- [x] **RAVDESS text-only ablation — WF1 0.053 (at chance, as expected)**
- [x] **RAVDESS multimodal cross-attention — WF1 0.858 (+6.2 pp over audio-only)**
- [ ] Speaker-independent split for honest RAVDESS numbers
- [ ] MELD baselines (text-only with context, audio-only, multimodal)
- [ ] IEMOCAP loader implementation (pending license)
- [ ] HuggingFace Spaces deployment of the demo
- [ ] Blog post on [skakarh.com](https://www.skakarh.com/blog/) with results writeup
- [ ] Cross-lingual transfer experiment (XLM-R + multilingual wav2vec2)

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
| ✍️ **Blog** | [skakarh.com/blog](https://www.skakarh.com/blog/) |
| 🛠️ **Services** | [skakarh.com/services](https://www.skakarh.com/services/) |
| 💼 **LinkedIn** | [linkedin.com/in/shahnawazkakarh](https://www.linkedin.com/in/shahnawazkakarh) |
| 📦 **More projects** | [github.com/ShahnawazKakarh](https://github.com/ShahnawazKakarh) |
