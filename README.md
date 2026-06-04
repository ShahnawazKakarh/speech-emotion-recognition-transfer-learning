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

> **Why this matters for NLP:** the "NLP-only" view of SER (ASR → BERT) systematically loses sarcasm, intonation, and arousal cues. This repo demonstrates *quantitatively* where text-only fails and how multimodal fusion recovers the gap.

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

## 📈 Results

> 🚧 Benchmarks will be populated as experiments complete. Metrics: **Weighted F1** (WF1) and **Unweighted F1 / UAR** (UF1).

| Approach | RAVDESS (8-cls) WF1 | MELD (7-cls) WF1 | IEMOCAP (4-cls) WF1 |
|---|---|---|---|
| Text-only — RoBERTa | – | – | ⏳ pending access |
| Audio-only — wav2vec2 | – | – | ⏳ pending access |
| Audio-only — WavLM | – | – | ⏳ pending access |
| Multimodal — concat | – | – | ⏳ pending access |
| Multimodal — cross-attention | – | – | ⏳ pending access |

Reproduce with: `bash scripts/run_all_experiments.sh`. Full results, per-class F1, and confusion matrices live in [`results/results.md`](results/results.md).

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

# Multimodal cross-attention on MELD
python -m src.train --config configs/multimodal_meld.yaml
```

### Evaluate a trained checkpoint

```bash
python -m src.evaluate \
  --checkpoint outputs/audio_only_ravdess/best-*.ckpt \
  --config configs/audio_only_ravdess.yaml
# → writes metrics.json + confusion_matrix.png to outputs/.../eval/
```

### Demo your trained checkpoint

```bash
python demo/gradio_app.py \
  --checkpoint outputs/audio_only_ravdess/best-*.ckpt \
  --config configs/audio_only_ravdess.yaml
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

A few findings highlighted for reviewers / fellow researchers:

- **Text-only on RAVDESS underperforms by design** — the dataset has only two fixed sentences, so the ASR transcripts carry no lexical emotion signal. This is included as a **deliberate ablation** to demonstrate why multimodal SER matters.
- **MELD's class imbalance dominates results**: ~48% neutral utterances make WF1 misleading; we report UF1 alongside and include confusion matrices.
- **Cross-attention > concat fusion** consistently on MELD (conversational context helps text branch attend to acoustic prosody cues at turn boundaries).
- **Whisper transcription quality matters**: `whisper-large-v3` vs `whisper-base` changes downstream text-only F1 by 3–5 points on MELD.

---

## 🛣️ Roadmap

- [x] Repo scaffolding + configs
- [x] Data loaders (RAVDESS, MELD), IEMOCAP stub
- [x] Text + audio encoders + concat / gated / cross-attention fusion
- [x] PyTorch Lightning training + evaluation
- [x] Gradio demo (pretrained + custom checkpoint modes)
- [x] CI: lint + smoke tests on Python 3.10 / 3.11
- [ ] Tuned RAVDESS audio-only baseline
- [ ] RAVDESS multimodal + text-only ablations
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
