# 🎙️ Speech Emotion Recognition via Transfer Learning

> Comparing **text-only (RoBERTa)**, **audio-only (wav2vec2 / WavLM / HuBERT)**, and **multimodal fusion** transfer learning approaches for Speech Emotion Recognition on **RAVDESS**, **MELD**, and **IEMOCAP**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20709692.svg)](https://doi.org/10.5281/zenodo.20709692)
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

> **Cross-Lingual Speech Emotion Recognition for Indo-Aryan Languages: Acoustic Feature Collapse, Within-Language Show Overfitting, and Asymmetric Transformer Transfer**
>
> Muhammad Shahnawaz Khan. *Independent Researcher.* June 2026.
> Zenodo Concept DOI: [10.5281/zenodo.20709692](https://doi.org/10.5281/zenodo.20709692) — resolves to latest version.
> v2.0.0 version DOI: [10.5281/zenodo.20709693](https://doi.org/10.5281/zenodo.20709693).

The v2 preprint reports four findings on speech emotion recognition for South Asian Indo-Aryan languages — Urdu, Sindhi, and Punjabi — that together characterise persistent failure modes across both classical and transformer-based methodologies:

1. **Modernised Sindhi baseline beats published paper by +1.70 pp UAR** (SVM-RBF on IS10 features, UAR 0.5699).
2. **Catastrophic negative transfer (~30 pp UAR drop)** between Urdu and Sindhi despite their linguistic proximity; the within-language feature-set ranking inverts under transfer.
3. **Multilingual XLS-R encoder beats English-only baseline by +11.4 pp weighted-F1** on English speaker-independent RAVDESS.
4. **First multi-seed transformer-based cross-lingual SER study between Punjabi RASA and URDU-Latif**, identifying three structural failure modes with bootstrap confidence intervals over three seeds:
   - *Punjabi within-language is leakage-saturated:* WF1 0.994 [0.988, 0.998] only under the publisher-fixed split that admits speaker overlap.
   - *Urdu within-language under show-independent evaluation collapses below the chance baseline:* pooled UAR 0.103 [0.075, 0.131] (n=88); the corpus is too small (~290 utterances) to train a transformer that generalises across recording domains.
   - *Urdu → Punjabi zero-shot transfer degenerates to single-class prediction at chance level:* pooled UAR 0.248 [0.225, 0.272] (n=2,886), robust across all three seeds. Only Pun → Urdu clears chance (pooled UAR 0.467, n=88), but with per-show variance too large for strong quantitative claims.

Findings 1–3 establish the methodological hierarchy classical-ML → monolingual transformer → multilingual transformer. Finding 4 then shows that even the strongest of these (multilingual XLS-R) fails on the public Indo-Aryan corpora, with failure modes attributable to the corpora rather than the architecture (the same architecture reaches UAR 0.773 on English RAVDESS-SI).

The v1 short paper (3 findings on classical Urdu-Sindhi; [Zenodo DOI 10.5281/zenodo.20640736](https://doi.org/10.5281/zenodo.20640736), June 11 2026) is superseded by the v2 above but remains citable as the v0.3.1 release of this repository.

```bibtex
@misc{khan2026crossLingualIndoAryan,
  author       = {Khan, Muhammad Shahnawaz},
  title        = {Cross-Lingual Speech Emotion Recognition for Indo-Aryan Languages:
                  Acoustic Feature Collapse, Within-Language Show Overfitting,
                  and Asymmetric Transformer Transfer},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {2.0.0},
  doi          = {10.5281/zenodo.20709692},
  url          = {https://doi.org/10.5281/zenodo.20709692}
}
```

---

## 📦 Versioned releases & how to cite

This project is published as versioned Zenodo releases, each with its own DOI.
From v2.0.0 onwards the deposits are auto-minted by Zenodo on every GitHub
release; the **Concept DOI** resolves to the latest version, and each
**version DOI** pins to an exact artifact. The v2 paper additionally
progresses toward a peer-reviewed venue submission.

| Version | Scope | Status |
|---|---|---|
| **v0.3.1** | v1 short paper — 3 findings on classical Urdu-Sindhi cross-corpus + RAVDESS-SI XLS-R (superseded by v2.x but still citable) | 🟢 Released — [Zenodo DOI 10.5281/zenodo.20640736](https://doi.org/10.5281/zenodo.20640736) |
| **v2.0.0** | v2 expanded paper — 4 findings, adds first multi-seed transformer cross-lingual SER audit between Punjabi RASA and URDU-Latif with bootstrap CIs; identifies three structural failure modes in low-resource Indo-Aryan SER | 🟢 Released — Concept DOI [10.5281/zenodo.20709692](https://doi.org/10.5281/zenodo.20709692) · v2.0.0 DOI [10.5281/zenodo.20709693](https://doi.org/10.5281/zenodo.20709693) |
| **v2.1.0** | v2 paper visual + structural upgrade — cover restyled to match cardiac paper format; **6 figures** added (methodology diagram, C2 feature-set inversion bar chart, C4 condition overview, per-seed UAR forest plot, per-class recall heatmap, pooled confusion matrices); new §2 Related Work section with 5 paragraphs; Observations 6 and 8 expanded (Neutral-avoidant collapse-class analysis, per-show sample-size confound); 10-item Future Work covering IEMOCAP / IndoAryan-SER-Bench / KMI SpeeD-IA / Pashto / Balochi; Computational Cost subsection; Data Availability + Acknowledgments + Author Contributions; Appendix A DOIs filled in. Same methodology and findings as v2.0.0. | 🟢 Released — Concept DOI [10.5281/zenodo.20709692](https://doi.org/10.5281/zenodo.20709692) (resolves latest) |
| **v2.2.0** | + Speaker-disjoint RASA re-evaluation (pending AI4Bharat metadata or manual annotation) | 🔵 Planned |
| **v3.0.0** | + Multimodal Punjabi (Whisper + XLM-R cross-attention) + Hindi (IITKGP-SEHSC) + Sindhi raw audio (MUET) + IEMOCAP-English reference + IndoAryan-SER-Bench v0.1 | 🔵 Target: ACM TALLIP / IEEE TASLP |

Step-by-step Zenodo release & ORCID flow: [`docs/zenodo_release.md`](docs/zenodo_release.md).
Machine-readable citation: [`CITATION.cff`](CITATION.cff).

If you use this benchmark, cite the v2 BibTeX in the [Published preprint](#-published-preprint-june-2026) section above (which uses the Concept DOI and therefore always resolves to the latest version), or pull the equivalent from `CITATION.cff` via GitHub's "Cite this repository" sidebar.

---

## 🔬 Active research — Cross-lingual SER for South Asian languages

> A separate research branch [**`research/cross-lingual`**](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/tree/research/cross-lingual) extends this framework to **Urdu**, **Sindhi**, **Punjabi**, and **Hindi** using multilingual transformer encoders (`xlm-roberta-base` + `wav2vec2-xls-r-300m`) and classical-ML baselines on public Indo-Aryan emotion corpora. The Phase 3 work is substantially complete; an expanded v2 paper is in preparation.

**Datasets integrated** (all in the research branch, none committed to the repo — see `data/` `.gitignore`):

| Corpus | Language | Samples | Emotions | Format | Status |
|---|---|---|---|---|---|
| Urdu-Sindhi (Syed et al. 2020) | Urdu + Sindhi | 1,435 | 7 (inc. Sarcasm) | features only | ✅ integrated |
| URDU-Dataset (Latif et al. 2018) | Urdu | 400 | 4 | raw WAV | ✅ integrated |
| Punjabi RASA (AI4Bharat / Kaggle 2024) | Punjabi | 9,634 | 4 | raw WAV 48 kHz | ✅ integrated |
| ShEMO (Mohamad Nezami et al. 2019) | Persian | 3,000 | 6 | raw WAV | ✅ downloaded, future use |
| IITKGP-SEHSC | Hindi | ~1,000 | 6 | raw WAV | ⏳ access requested |

### Finding 1 — Sindhi within-language baseline beats published paper

SVM-RBF on IS10 paralinguistic features, 5-fold stratified CV:

| Language | Our UAR | Paper UAR | Δ |
|---|---|---|---|
| **Sindhi** | **0.5699** | 0.5529 | **+1.70 pp** ✅ |
| Urdu | 0.5526 | 0.5696 | −1.70 pp |

### Finding 2 — Classical features collapse cross-corpus (Urdu ↔ Sindhi)

| Direction | Best transfer UAR | Within-language UAR | Gap |
|---|---|---|---|
| Urdu → Sindhi | 0.2734 (eGeMAPS, SVM-RBF) | 0.5699 | −30.99 pp |
| Sindhi → Urdu | 0.2622 (eGeMAPS, RF) | 0.5526 | −32.89 pp |

The within-language feature-set ranking inverts under transfer: low-dimensional eGeMAPS (88 features) transfers best, while ComParE (6,373) and IS10 (1,582) — the within-language leaders — transfer worst. To our knowledge this specific finding has not been published.

### Finding 3 — Multilingual XLS-R beats English-only on English

| Encoder | Test WF1 on RAVDESS-SI | Notes |
|---|---|---|
| **wav2vec2-XLS-R-300M (multilingual)** | **0.773** | +11.4 pp over wav2vec2-base; +4.5 pp over English-only multimodal |
| wav2vec2-base (English) | 0.659 | Prior baseline |
| XLM-R-base (multilingual) | 0.579 WF1 on MELD | Validates the text-side pipeline |

### Finding 4 — First transformer-based SER on Punjabi RASA

wav2vec2-XLS-R-300M fine-tuned on the Punjabi RASA emotional speech corpus (8,672 train / 962 test, 4 emotions, severe 7:1 imbalance) with class-weighted cross-entropy:

| Metric | Test |
|---|---|
| **Weighted F1** | **0.9969** |
| **Unweighted F1** | **0.9957** |
| Accuracy | 0.9969 |
| Best val/wf1 | 0.9897 (epoch 7 of 11) |

**Honest caveat**: the RASA release uses a random train/test split and does not expose speaker IDs (filename format `PAN_<gender>_<emotion>_<nnnnn>.wav`). Same-speaker overlap between train and test is likely and inflates these numbers; the score should be read as an upper bound. We recommend AI4Bharat publish speaker IDs to enable leave-one-speaker-out evaluation.

### Finding 5 — Asymmetric cross-lingual transformer transfer

Zero-shot transfer using the within-language checkpoints from findings #4 (Punjabi) and a separate within-language URDU-Latif baseline (test WF1 0.640 on 40 samples, 320 train):

| Direction | Within-lang WF1 | Cross-lang WF1 | Cross-lang UAR | Above chance? |
|---|---|---|---|---|
| Punjabi → Punjabi | 0.997 | — | — | (within-lang) |
| **Punjabi → Urdu** | — | **0.427** | **0.500** | ✅ 2.0× chance |
| Urdu → Urdu | 0.640 | — | — | (within-lang) |
| **Urdu → Punjabi** | — | **0.020** | **0.253** | ❌ ≈ chance (degenerate) |

Chance for 4-class = 0.25. The Punjabi → Urdu direction shows **structured arousal-encoded transfer**: angry (recall 0.80) and neutral (recall 0.90) transfer strongly, while happy (recall 0.20) and sad (recall 0.10) collapse — consistent with the multilingual encoder learning cross-lingual emotion features dominated by energy/arousal cues rather than valence cues. The reverse Urdu → Punjabi direction collapses to single-class prediction (model labels nearly every Punjabi sample as "sad"). The asymmetry tracks the 27× source-corpus-size ratio (8,672 vs 320 training samples), suggesting a data-volume threshold below which fine-tuning damages the encoder's cross-lingual emotion representations.

**Implication**: a naive "transformers solve cross-lingual SER" narrative is wrong. Even with a 128-language pre-trained encoder, cross-lingual transfer in low-resource Indo-Aryan SER is partial, asymmetric, arousal-biased, and gated by source-corpus size.

### Finding 6 — Multimodal XLM-R + XLS-R cross-attention validated on MELD

Full multilingual multimodal pipeline (text via XLM-R, audio via XLS-R-300M, cross-attention fusion) on the standard MELD English benchmark, confirming the multilingual encoder stack performs competitively with monolingual baselines before applying it to non-English corpora:

| Metric | Value |
|---|---|
| Test WF1 | 0.598 |
| Test accuracy | 0.612 |
| Test UF1 | 0.405 |

Full leaderboard, per-class breakdowns, confusion matrices, and methodological notes in [`results/results.md`](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/blob/research/cross-lingual/results/results.md) on the research branch.

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

### Phase 3 — Cross-lingual SER for South Asian languages (substantially complete 🔬)

- [x] **Code scaffolding for multilingual encoders** — XLM-R + wav2vec2-XLS-R configs (`configs/text_only_xlmr_meld.yaml`, `configs/audio_only_xlsr_ravdess_si.yaml`, `configs/multimodal_xlmr_xlsr_meld.yaml`); wired through `src/models/lightning_module.py` for per-encoder freeze control on memory-constrained Apple Silicon MPS
- [x] **XLM-R multilingual pipeline validated on MELD** — WF1 0.579 / UF1 0.409 / Acc 0.570 with 6-layer freeze, confirming the cross-lingual transformer pipeline works end-to-end
- [x] **Urdu-Sindhi Speech Emotion Corpus integrated** ([Syed et al. 2020, Zenodo](https://zenodo.org/records/3685274)) — 1,435 recordings (734 Urdu + 701 Sindhi), 7 emotions including the unusual **Sarcasm** class, 5 hand-crafted feature representations (eGeMAPS, ComParE, IS09, IS10, Prosody)
- [x] **Classical-ML cross-lingual baselines on Urdu-Sindhi** ([`scripts/train_urdu_sindhi_classical.py`](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/blob/research/cross-lingual/scripts/train_urdu_sindhi_classical.py), [`scripts/cross_corpus_urdu_sindhi.py`](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/blob/research/cross-lingual/scripts/cross_corpus_urdu_sindhi.py)) — documented catastrophic ~30 pp UAR drop across all 30 (direction × feature-set × classifier) configurations
- [x] **Multimodal XLM-R + XLS-R cross-attention on MELD** — WF1 0.598 / UF1 0.405 / Acc 0.612, validates the full multilingual multimodal stack
- [x] **URDU-Dataset (Latif et al. 2018) integrated** — [`src/data/urdu_latif.py`](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/blob/research/cross-lingual/src/data/urdu_latif.py), with raw WAV audio cloned from [the authors' public GitHub](https://github.com/siddiquelatif/URDU-Dataset); supports random + show-independent splits
- [x] **Punjabi RASA (AI4Bharat / Kaggle) integrated** — [`src/data/punjabi_rasa.py`](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/blob/research/cross-lingual/src/data/punjabi_rasa.py), 9,634 samples, severe 7:1 class imbalance handled with class-weighted CE
- [x] **First transformer-based SER on Punjabi RASA** — wav2vec2-XLS-R-300M reaches WF1 0.997 / UF1 0.996 on the official test split (with same-speaker-leak caveat documented)
- [x] **Cross-lingual transformer transfer experiments** — [`scripts/cross_corpus_punjabi_urdu_latif.py`](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/blob/research/cross-lingual/scripts/cross_corpus_punjabi_urdu_latif.py); bidirectional zero-shot evaluation reveals asymmetric data-gated transfer (Punjabi→Urdu at 2× chance, Urdu→Punjabi at chance)
- [x] **First publication out** — v1 paper: [Zenodo DOI 10.5281/zenodo.20640736](https://doi.org/10.5281/zenodo.20640736), 11 June 2026 (3 findings on classical Urdu-Sindhi)
- [x] **v2 expanded paper out** — v2 paper with multi-seed transformer cross-lingual audit: [Zenodo Concept DOI 10.5281/zenodo.20709692](https://doi.org/10.5281/zenodo.20709692) (v2.0.0 DOI [10.5281/zenodo.20709693](https://doi.org/10.5281/zenodo.20709693)), 16 June 2026
- [ ] Raw audio request to Sajjad Ali Memon at MUET Pakistan (sent; awaiting reply) — for transformer-based Urdu-Sindhi experiments using the original 1,435-sample raw audio
- [ ] Raw audio request to Prof. K. Sreenivasa Rao at IIT Kharagpur (sent; awaiting reply) — for IITKGP-SEHSC Hindi corpus integration
- [ ] Speaker-independent re-evaluation of Punjabi RASA (pending AI4Bharat speaker-ID release or manual annotation)
- [ ] Multimodal cross-attention on Punjabi (currently audio-only; add Whisper-Punjabi transcripts + XLM-R text branch)
- [ ] Self-recorded Urdu / Punjabi / Hindi corpus complementing the public datasets
- [ ] **v2 expanded paper** — 6-finding journal version targeting ACM TALLIP / Interspeech 2027 workshop / IEEE TASLP

### Phase 4 — Deployment + outreach

- [ ] Mental-health screening pilot (research partnership exploration)
- [ ] Multilingual extension of the HuggingFace Spaces demo
- [ ] Blog post series on skakarh.com on multilingual SER

> The Phase 3 cross-lingual work is the principal novel research contribution of this programme. Phases 1 and 2 establish the engineering foundation and English baselines that the cross-lingual study compares against. See the [research direction note](#research-direction) in the Overview section for the broader motivation.

---

## 📚 Citations & References

Key papers this work builds on:

**Self-supervised speech & multilingual encoders:**
- Baevski et al., *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*, NeurIPS 2020
- Chen et al., *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*, IEEE JSTSP 2022
- Babu et al., *XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale*, Interspeech 2022
- Conneau et al., *Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)*, ACL 2020
- Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, 2019
- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)*, ICML 2023

**Hand-crafted paralinguistic feature sets:**
- Eyben et al., *The Geneva Minimalistic Acoustic Parameter Set (GeMAPS / eGeMAPS) for Voice Research and Affective Computing*, IEEE TAC 2016
- Schuller et al., *The INTERSPEECH 2010 Paralinguistic Challenge*, Interspeech 2010
- Schuller et al., *The INTERSPEECH 2009 Emotion Challenge*, Interspeech 2009
- Weninger et al., *On the Acoustics of Emotion in Audio: What Speech, Music, and Sound have in Common*, Frontiers in Psychology 2013 (ComParE feature set)

**English emotion datasets:**
- Livingstone & Russo, *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*, PLOS ONE 2018
- Poria et al., *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations*, ACL 2019
- Busso et al., *IEMOCAP: Interactive Emotional Dyadic Motion Capture Database*, LREC 2008

**South Asian / Indo-Aryan / low-resource SER:**
- Syed et al., *URDU & Sindhi Speech Emotion Corpus*, Zenodo 2020 — [DOI 10.5281/zenodo.3685274](https://doi.org/10.5281/zenodo.3685274)
- Latif et al., *Cross Lingual Speech Emotion Recognition: Urdu vs. Western Languages*, FIT 2018 — the URDU-Dataset
- Latif et al., *Survey of Deep Representation Learning for Speech Emotion Recognition*, IEEE TAC 2023
- Mohamad Nezami et al., *ShEMO — A Large-Scale Validated Database for Persian Speech Emotion Detection*, LREC 2019
- Retta et al., *Cross-lingual Speech Emotion Recognition: Amharic vs. English using CNN+MFCC*, arXiv 2307.10814, 2023
- AI4Bharat (IIT Madras), *Rasa: Building a Large-Scale, Multilingual Speech Emotion Corpus for Indian Languages*, 2024 — source of the Punjabi RASA Kaggle release
- Rao et al., *Database of Acted Speech Emotion in Hindi: IITKGP-SEHSC*, IIT Kharagpur

**Cross-lingual transfer methodology:**
- Sagi et al., *Cross-Lingual Transfer Learning for Speech Emotion Recognition*, various venues 2019–2024
- Pappagari et al., *X-vectors meet Emotions: A Study On Dependencies Between Emotion and Speaker Recognition*, ICASSP 2020 (motivates speaker-independent evaluation)

If you use this work, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub will generate the BibTeX for you from the sidebar). The canonical citation is the Zenodo DOI shown in the [Published preprint](#-published-preprint-june-2026) section above.

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
