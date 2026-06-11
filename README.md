# 🎙️ Speech Emotion Recognition via Transfer Learning

> Comparing **text-only (RoBERTa)**, **audio-only (wav2vec2 / WavLM / HuBERT)**, and **multimodal fusion** transfer learning approaches for Speech Emotion Recognition on **RAVDESS**, **MELD**, and **IEMOCAP**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20640736.svg)](https://doi.org/10.5281/zenodo.20640736)
[![Preprints.org](https://img.shields.io/badge/Preprints.org-218084-orange)](https://www.preprints.org/manuscript/218084)
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

## 📄 Published preprint (June 2026)

> **When Acoustic Features Fail: Catastrophic Negative Transfer in Cross-Corpus Urdu-Sindhi Speech Emotion Recognition**
>
> Muhammad Shahnawaz Khan. *Independent Researcher.* June 2026.
> Zenodo DOI: [10.5281/zenodo.20640736](https://doi.org/10.5281/zenodo.20640736) · Preprints.org ID: [218084](https://www.preprints.org/manuscript/218084)

Three publishable findings on speech emotion recognition for South Asian Indo-Aryan languages:

1. **Modernised Sindhi baseline beats published paper by +1.70 pp UAR** (Syed et al. 2020).
2. **Catastrophic negative transfer (~30 pp UAR drop)** between Urdu and Sindhi despite their linguistic proximity; the within-language feature-set ranking inverts under transfer.
3. **Multilingual XLS-R encoder beats English-only baseline by +11.4 pp weighted-F1** on English speaker-independent RAVDESS.

```bibtex
@misc{khan2026crosscorpus,
  author       = {Khan, Muhammad Shahnawaz},
  title        = {When Acoustic Features Fail: Catastrophic Negative Transfer
                  in Cross-Corpus Urdu-Sindhi Speech Emotion Recognition},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20640736},
  url          = {https://doi.org/10.5281/zenodo.20640736}
}
```

---

## 🔬 Active research — Cross-lingual SER for South Asian languages

> A separate research branch [**`research/cross-lingual`**](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/tree/research/cross-lingual) extends this framework to **Urdu**, **Sindhi**, **Hindi**, and **Punjabi** using multilingual transformer encoders (`xlm-roberta-base` + `wav2vec2-xls-r-300m`) and classical-ML baselines on public Indo-Aryan emotion corpora.

**What's in the branch (not in `master`):**
- Multilingual encoder configs ([`configs/text_only_xlmr_meld.yaml`](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/blob/research/cross-lingual/configs/text_only_xlmr_meld.yaml), `audio_only_xlsr_ravdess_si.yaml`, `multimodal_xlmr_xlsr_meld.yaml`) with Apple Silicon MPS memory tuning
- Integration of the **Urdu-Sindhi Speech Emotion Corpus** ([Syed et al. 2020](https://zenodo.org/records/3685274)) — 1,435 recordings, 7 emotions including **Sarcasm**
- Classical-ML baseline trainer ([`scripts/train_urdu_sindhi_classical.py`](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/blob/research/cross-lingual/scripts/train_urdu_sindhi_classical.py)) running 5-fold stratified CV across SVM-RBF / RandomForest / MLP over 5 feature representations

**Headline result so far** — SVM-RBF on InterSpeech 2010 paralinguistic features:

| Language | Our UAR | Paper UAR | Δ |
|---|---|---|---|
| **Sindhi** | **0.5699** | 0.5529 | **+1.70 pp** ✅ |
| Urdu | 0.5526 | 0.5696 | −1.70 pp |

**Cross-corpus finding** — catastrophic negative transfer between Urdu and Sindhi:

| Direction | Best transfer UAR | Within-language UAR | Gap |
|---|---|---|---|
| Urdu → Sindhi | 0.2734 (eGeMAPS, SVM-RBF) | 0.5699 | **−30.99 pp** |
| Sindhi → Urdu | 0.2622 (eGeMAPS, RF) | 0.5526 | **−32.89 pp** |

Despite being neighboring Indo-Aryan languages, acoustic features do not transfer between them. This motivates the multilingual transformer approach (XLM-R + wav2vec2-XLS-R). To our knowledge this specific cross-corpus result has not been published.

**Multilingual encoder validation** — multilingual transformers beat English-only on English:

| Encoder | Test WF1 on RAVDESS-SI | Notes |
|---|---|---|
| **wav2vec2-XLS-R-300M (multilingual)** | **0.773** | +11.4 pp over wav2vec2-base; +4.5 pp over English-only multimodal |
| wav2vec2-base (English) | 0.659 | Prior baseline |
| XLM-R-base (multilingual) | 0.579 WF1 on MELD | In target band; validates text-side pipeline |

Full leaderboard, methodology, and roadmap in the [branch's results section](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/blob/research/cross-lingual/results/results.md#cross-lingual--urdu-sindhi-speech-emotion-corpus-syed-et-al-2020).

To work on this directly:
```bash
git fetch origin research/cross-lingual
git checkout research/cross-lingual
```

---

## 🧭 Overview

Speech Emotion Recognition (SER) lives at the intersection of **paralinguistic signal** (pitch, prosody, energy) and **linguistic content** (what was actually said). Labeled emotion data is scarce — IEMOCAP has ~12 hours, RAVDESS has 1,440 clips, MELD has 13K utterances — so **transfer learning from self-supervised speech models and large language models** is the dominant paradigm.

This repository implements and benchmarks three families of approaches on standard English emotion datasets (RAVDESS, MELD), and serves as the foundation for an active research programme on **cross-lingual SER for low-resource South Asian languages** (Urdu, Hindi, Punjabi) — a population of over 700 million speakers almost entirely underserved by current emotion AI research.

| Approach | Encoder | What it captures | Strength |
|---|---|---|---|
| **Text-only** | `RoBERTa` (current) / `XLM-R` (future) | Lexical / semantic emotion cues | Cheap, leverages NLP ecosystem |
| **Audio-only** | `wav2vec2` / `WavLM` (current) / `XLS-R` (future) | Prosody, voice quality, paralinguistic cues | Captures *how* something is said |
| **Multimodal** | Audio + Text with cross-attention fusion | Both signals jointly | State-of-the-art on conversational SER |

> **Research direction.** This work establishes English baselines that quantify a non-obvious finding: multimodal fusion is not universally better than unimodal models. The next phase extends the framework to multilingual encoders (`xlm-roberta-base`, `wav2vec2-xls-r-300m`) and applies it to Urdu, Hindi, and Punjabi using a self-recorded native-speaker corpus. The objective is to enable downstream emotion AI applications — mental-health screening, call-centre analytics, accessibility technology — for South Asian language communities currently excluded from English-centric SER systems.

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

Test = 2,609 utterances. Heavy class imbalance (~48% neutral, just 50 fear utterances).

| Approach | Encoder | WF1 | UF1 | Accuracy |
|---|---|---|---|---|
| **Text-only (context=2)** | RoBERTa-base | **0.609** | **0.459** | **0.593** |
| Multimodal (cross-attn) | RoBERTa + WavLM | 0.590 | 0.404 | 0.597 |
| Audio-only | WavLM-base | 0.357 | 0.153 | 0.416 |

**Headline findings**:

- **Text-only beats multimodal by 1.9 pp WF1** on MELD — *the opposite* of what we saw on RAVDESS. Same architecture, opposite verdict.
- **Modality complementarity is the determining factor.** Multimodal fusion helps when both modalities carry independent, non-noisy signal (RAVDESS audio is rich, text is degenerate → fusion wins). It hurts when one modality is strong and the other is noise (MELD text is rich, audio is noisy TV-show clips → fusion loses).
- **WavLM-base class-collapses** on MELD: surprise / fear / disgust F1 = 0.000. The audio backbone refuses to predict three of seven classes and converges on a "predict neutral / joy / anger, give up on the rest" local minimum. Consistent with published MELD audio-only baselines (UF1 ~0.15–0.20).

Full per-class breakdown, confusion matrices, and the *modality-complementarity discussion* in [`results/results.md`](results/results.md).

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

### Phase 1 — English baselines (complete ✅)

- [x] Repo scaffolding + configs
- [x] Data loaders (RAVDESS, MELD) + IEMOCAP stub
- [x] Text + audio encoders + concat / gated / cross-attention fusion
- [x] PyTorch Lightning training + evaluation
- [x] Gradio demo (pretrained + custom checkpoint modes)
- [x] CI: lint + smoke tests on Python 3.10 / 3.11
- [x] **RAVDESS random-split baselines** (multimodal WF1 0.858)
- [x] **RAVDESS speaker-independent baselines** (multimodal WF1 0.728)
- [x] **HuggingFace Spaces live deployment** — [demo](https://huggingface.co/spaces/Shahnawazkakarh/speech-emotion-recognition)
- [x] **MELD baselines** (text 0.609 / audio 0.357 / multimodal 0.590 — *modality-complementarity finding*)

### Phase 2 — Broader English coverage (in progress 🟡)

- [ ] IEMOCAP loader implementation (USC SAIL license requested)
- [ ] Three-dataset comparative analysis (RAVDESS + MELD + IEMOCAP)

### Phase 3 — Cross-lingual SER for South Asian languages (active research 🔬)

- [x] **Code scaffolding for multilingual encoders** — XLM-R + wav2vec2-XLS-R configs (`configs/text_only_xlmr_meld.yaml`, `configs/audio_only_xlsr_ravdess_si.yaml`, `configs/multimodal_xlmr_xlsr_meld.yaml`); wired through `src/models/lightning_module.py` for per-encoder freeze control on memory-constrained Apple Silicon MPS
- [x] **XLM-R multilingual pipeline validated on MELD** — WF1 0.579 / UF1 0.409 / Acc 0.570 with 6-layer freeze, confirming the cross-lingual transformer pipeline works end-to-end
- [x] **Urdu-Sindhi Speech Emotion Corpus integrated** ([Syed et al. 2020, Zenodo](https://zenodo.org/records/3685274)) — 1,435 recordings (734 Urdu + 701 Sindhi), 7 emotions including the unusual **Sarcasm** class, 5 hand-crafted feature representations (eGeMAPS, ComParE, IS09, IS10, Prosody). See [`scripts/train_urdu_sindhi_classical.py`](scripts/train_urdu_sindhi_classical.py) for the classical-ML baseline trainer.
- [ ] Classical-ML cross-lingual baselines on Urdu-Sindhi corpus (in progress)
- [ ] Literature review of cross-lingual SER + Indo-Aryan emotion datasets (Latif 2018, Retta 2023, etc.)
- [ ] Raw audio request to dataset authors (Memon at RMIT Australia) for transformer-based extension
- [ ] Self-recorded Urdu / Punjabi / Hindi corpus complementing the public datasets
- [ ] Zero-shot, few-shot, and fully-fine-tuned cross-lingual evaluation with wav2vec2-XLS-R + XLM-R
- [ ] First publication: target Interspeech / ICASSP workshop 2027 or IEEE TASLP

### Phase 4 — Deployment + outreach

- [ ] Mental-health screening pilot (research partnership exploration)
- [ ] Multilingual extension of the HuggingFace Spaces demo
- [ ] Blog post series on skakarh.com on multilingual SER

> The Phase 3 cross-lingual work is the principal novel research contribution of this programme. Phases 1 and 2 establish the engineering foundation and English baselines that the cross-lingual study compares against. See the [research direction note](#research-direction) in the Overview section for the broader motivation.

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

## 👤 About the Author

This project is built by **Muhammad Shahnawaz Khan**, a software engineer working at the intersection of backend systems and applied AI/ML. Day-to-day work spans Python and JavaScript backends, HuggingFace + transformers-based applications, GenAI tooling, and production ML systems. This repository is independent research conducted alongside that practitioner work — part of a broader programme on emotion AI for underserved languages.

| | |
|---|---|
| 🌐 Website | [skakarh.com](https://www.skakarh.com) |
| 📖 ORCID | [0009-0007-4055-6563](https://orcid.org/0009-0007-4055-6563) |
| 💼 LinkedIn | [linkedin.com/in/skakarh](https://www.linkedin.com/in/skakarh) |
| 💻 GitHub | [@ShahnawazKakarh](https://github.com/ShahnawazKakarh) |
| ✉️ Email | shahnawaz.jrw@gmail.com |

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
