---
title: "Multimodal Speech Emotion Recognition Isn't Always Better — Lessons From RAVDESS and MELD"
slug: multimodal-speech-emotion-recognition-ravdess-meld
date: 2026-06-06
author: Shahnawaz Khan
description: "Building text-only, audio-only, and multimodal speech emotion recognition models with RoBERTa, wav2vec2, and WavLM. On RAVDESS multimodal wins by 6.9 pp. On MELD text-only wins by 1.9 pp. The lesson is about modality complementarity, not architecture."
keywords:
  - speech emotion recognition
  - multimodal emotion recognition
  - RAVDESS
  - MELD
  - wav2vec2
  - WavLM
  - RoBERTa
  - cross-attention fusion
  - transfer learning
  - audio classification
  - emotion AI
  - PyTorch Lightning
  - HuggingFace
canonical: https://www.skakarh.com/blog/multimodal-speech-emotion-recognition-ravdess-meld/
og:
  type: article
  image: /assets/blog/ser-hero.png
  image_alt: Bar chart comparing WF1 across audio, text, and multimodal models on RAVDESS and MELD
twitter:
  card: summary_large_image
schema:
  type: BlogPosting
  speakable:
    - "h1"
    - ".lede"
    - ".key-finding"
faq:
  - q: Does multimodal speech emotion recognition always outperform unimodal models?
    a: No. On RAVDESS (clean lab speech, fixed sentences), our multimodal model beats audio-only by 6.9 percentage points weighted F1. On MELD (noisy multi-party TV conversation), text-only RoBERTa actually outperforms multimodal by 1.9 percentage points. Multimodal fusion helps when modalities carry complementary information; it can hurt when one modality is noisy and the other is strong.
  - q: What is the best speech emotion recognition model for clean audio?
    a: For RAVDESS-style clean lab speech, our best result is a RoBERTa + wav2vec2 cross-attention model — 0.728 weighted F1 on the official speaker-independent split. The key gain over audio-only is on the neutral class (+57 percentage points F1) where audio-only models tend to collapse on unseen speakers.
  - q: Why does text-only RoBERTa work so well on MELD?
    a: MELD utterances are real *Friends* TV-show dialogue with rich semantic content — words like "I can't believe this!" or "you're scaring me" carry clear emotional signal. RoBERTa with two utterances of conversational context reaches WF1 0.609, at the top of the published baseline range for text-only MELD models.
---

# Multimodal Speech Emotion Recognition Isn't Always Better — Lessons From RAVDESS and MELD

> **TL;DR**: I built three speech emotion recognition models — text-only, audio-only, and a multimodal cross-attention fusion — and trained them on two emotion datasets. On RAVDESS, multimodal wins by **+6.9 pp WF1**. On MELD, text-only beats multimodal by **−1.9 pp WF1**. The lesson isn't about architecture; it's about **modality complementarity**. Code on [GitHub](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning), live demo on [HuggingFace Spaces](https://huggingface.co/spaces/Shahnawazkakarh/speech-emotion-recognition).

<p class="lede">Speech Emotion Recognition (SER) is one of those problems where the "obvious" answer — "of course multimodal beats unimodal" — turns out to be wrong half the time. This post walks through what I built, what I measured, and what the numbers actually say.</p>

## What I built

A research codebase that trains and evaluates three SER architectures on two standard datasets, with a clean engineering story and reproducible configs:

- **Text-only**: `roberta-base` with a classification head on top of `[CLS]`. For MELD, I prepend two utterances of conversational context separated by `[SEP]` tokens to give the model dialogue grounding.
- **Audio-only**: `wav2vec2-base` (RAVDESS) or `microsoft/wavlm-base` (MELD), with mean-pooled hidden states feeding a small MLP classifier. The CNN feature extractor is frozen; the transformer is fine-tuned with a low learning rate.
- **Multimodal**: a cross-attention fusion module that lets text tokens attend to audio frames and vice versa, with the fused representation feeding the classifier head. ~221 M parameters total.

The whole thing runs on Apple Silicon MPS via PyTorch Lightning. Everything you'd want — configs, seeds, speaker-independent splits, class weighting, learning-rate sweeps — is in YAML rather than buried in Python.

[**GitHub repo →**](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning)

## The two datasets, briefly

| Dataset | Domain | Clips | Classes | What's hard about it |
|---|---|---|---|---|
| **RAVDESS** | Acted lab recordings | 1 440 | 8 (calm, happy, sad, angry, fearful, disgust, surprised, neutral) | Only **2 fixed sentences** repeated by 24 actors. Text carries zero signal. Easy to overfit to actor identity. |
| **MELD** | *Friends* TV-show dialogue | 13 708 | 7 (neutral, joy, sadness, anger, surprise, fear, disgust) | Multi-party, laugh tracks, background music, **48% neutral** class dominance. Audio is genuinely noisy. |

These two datasets are deliberately chosen to be **opposites along the modality-complementarity axis**. RAVDESS audio is rich; RAVDESS text is degenerate. MELD text is rich; MELD audio is messy. If multimodal fusion universally helped, both datasets would show gains. Spoiler: only one does.

## The numbers

### RAVDESS — speaker-independent split

Test set = actors 21-24 (240 samples). Train = actors 1-18. This is the published convention; numbers from random-split papers are inflated by ~13 pp due to **speaker leakage**.

| Model | Weighted F1 | Unweighted F1 | Accuracy |
|---|---|---|---|
| Text-only RoBERTa | 0.031 | 0.029 | 13.3% |
| Audio-only wav2vec2 | 0.659 | 0.631 | 66.7% |
| **Multimodal cross-attention** | **0.728** | **0.731** | **72.9%** |

<p class="key-finding"><strong>Multimodal beats audio-only by +6.9 percentage points weighted F1 on unseen speakers.</strong> Even more striking: the neutral class — which audio-only fails badly on (F1 0.21, recall 0.125) — gets a <strong>+57 pp F1 boost</strong> under multimodal (F1 0.78, recall 0.875).</p>

The mechanism: the text branch on its own collapses to predicting "calm" for every input (because the same 2 sentences appear in every emotion class), but that constant signal acts as a useful prior that lets the audio branch take more risks. The cross-attention layers can use the consistent text representation as an anchor while the audio branch handles the prosodic discrimination.

### MELD — official dev/test splits

Test set = 2 609 utterances. Class distribution is severely imbalanced (~48% neutral, 50 fear utterances total).

| Model | Weighted F1 | Unweighted F1 | Accuracy |
|---|---|---|---|
| Audio-only WavLM | 0.357 | 0.153 | 41.6% |
| Multimodal cross-attention | 0.590 | 0.404 | 59.7% |
| **Text-only RoBERTa (+ context)** | **0.609** | **0.459** | **59.3%** |

<p class="key-finding"><strong>Text-only beats multimodal by 1.9 percentage points weighted F1 on MELD.</strong> The audio branch is so weak (UF1 0.15, fully collapsing on three of seven classes) that the fusion layer can't help — it drags the strong text representation toward the noisier audio one.</p>

Per-class, multimodal vs text-only on MELD:

| Class | Text-only F1 | Multimodal F1 | Δ |
|---|---|---|---|
| neutral | 0.733 | 0.755 | +2.1 pp |
| joy | 0.587 | 0.582 | −0.5 pp |
| sadness | 0.353 | 0.293 | −5.9 pp |
| anger | **0.487** | 0.372 | **−11.5 pp** |
| surprise | 0.576 | 0.537 | −3.9 pp |
| fear | **0.205** | 0.059 | **−14.7 pp** |
| disgust | 0.273 | 0.229 | −4.3 pp |

The audio degradation is concentrated on the emotions that depend most on semantic content — fear, anger, sadness. Things people **say** to convey fear ("oh my god", "what's happening") carry more signal than the prosody of low-arousal fear; the audio branch can't see that, and dragging the text representation toward audio hurts.

## So when does multimodal help?

A simple heuristic falls out of these two datasets:

> **Multimodal fusion helps when both modalities carry independent, non-noisy signal.** It hurts when one modality is strong and the other is noise.

| Audio | Text | Multimodal verdict |
|---|---|---|
| Strong | Weak/random | **Multimodal wins** (RAVDESS) |
| Weak (noisy) | Strong | **Multimodal loses** (MELD) |
| Strong | Strong | Multimodal probably wins (literature: CMU-MOSEI, IEMOCAP audio+text) |
| Weak | Weak | Either could win, depends on which is less bad |

This isn't a new finding — [Zadeh et al. 2018](https://aclanthology.org/P18-1208/) and [Hazarika et al. 2020 (MISA)](https://arxiv.org/abs/2005.03545) document it for video sentiment analysis on CMU-MOSEI. What's interesting is **replicating it within a single SER codebase** with the same architecture, just swapping datasets.

## The engineering side, briefly

A few things I wish someone had told me before I started:

**Speaker-independent splits matter.** Random-split RAVDESS reports 0.86 WF1 multimodal. Speaker-independent splits report 0.73. The 13 pp gap is the **speaker-leakage premium** — the amount by which random-split numbers in the SER literature are systematically inflated. If you're benchmarking against papers, check whether their split was speaker-independent before claiming a beat or a loss.

**SSL fine-tuning is brutally sensitive to learning rate.** My first audio-only RAVDESS run used LR=1e-4 (the wav2vec2 *pre-training* rate). The model unlearned its representations within 4 epochs, EarlyStopping fired at WF1 0.27. Dropping to LR=2e-5 with the bottom 8 of 12 transformer layers frozen got me to 0.659. **5× LR difference → 3× performance difference.**

**MELD audio is harder than the papers suggest.** Audio-only WavLM-base never learned to predict surprise / fear / disgust at all — F1 = 0.000 for all three classes. The "best" checkpoint was at epoch 0 because the model immediately found a local minimum of "predict neutral / joy / anger, give up on the rest" and never escaped. This is consistent with published MELD audio-only baselines around UF1 0.15-0.20, but it's worth seeing in your own runs.

**MPS unified memory is the bottleneck for the multimodal model.** The original `batch_size=8, max_audio_seconds=8.0` OOM'd at epoch 0 on a 24 GB M-series Mac. Dropping to `batch_size=2, accumulate_grad_batches=8, max_audio_seconds=5.0` keeps the effective batch size at 16 but cuts per-step activation memory by ~4×. No quality loss.

**HuggingFace Spaces deployment has its own boss fight.** Four cascading bugs to ship the demo: `huggingface_hub<1.0` pin so `HfFolder` still imports, explicit `torch` declaration after the HF base image stopped pre-installing it, swapping `gr.Label` for `gr.Markdown` to avoid the `gradio_client` schema-introspection bug, and finally a monkeypatch on `gradio_client.utils.get_type` to handle bool schemas. The CHANGELOG documents all of it.

## Try it yourself

The live demo runs `superb/wav2vec2-base-superb-er` for instant inference. Upload a `.wav` clip or record from your mic:

[**🤗 Open the live demo on HuggingFace Spaces →**](https://huggingface.co/spaces/Shahnawazkakarh/speech-emotion-recognition)

[**Clone the GitHub repo and reproduce the numbers →**](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning)

If you want a deeper dive, the [full results document](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/blob/master/results/results.md) has per-class breakdowns, confusion matrices, and a much longer "Cross-experiment observations" section.

## What's next

A few directions I'd want to push this if I had more weeks:

1. **Cross-lingual transfer**. The current text encoder is English-only. Replacing RoBERTa with XLM-R and wav2vec2 with the multilingual variant would open up Urdu, Hindi, and other lower-resource languages I care about. I have a small target dataset planned (200-500 self-recorded Urdu/Punjabi clips from native speakers) for evaluation.
2. **Better MELD audio handling**. The class-collapse on surprise/fear/disgust suggests the model needs more inductive bias — speaker normalization, foreground/background separation, or simply a different audio backbone (Whisper encoder?).
3. **IEMOCAP**. The third standard SER dataset is licensed by USC SAIL. License request submitted; results pending.

If any of this is interesting and you're working in adjacent territory — sales-emotion analytics, language-learning feedback, accessibility, mental-health screening, multimodal robot perception — I'd love to compare notes. Email's on [skakarh.com](https://www.skakarh.com/).

---

**Built by [Shahnawaz Khan](https://www.skakarh.com)** · QA automation lead by day, ML/AI research on the side · [GitHub](https://github.com/ShahnawazKakarh) · [LinkedIn](https://www.linkedin.com/in/skakarh) · [Newsletter](https://www.skakarh.com/blog/)

**Citations** (if you use this work):
```bibtex
@misc{khan2026ser,
  title  = {Multimodal Speech Emotion Recognition with RoBERTa and wav2vec2: A Two-Dataset Study},
  author = {Khan, Shahnawaz},
  year   = {2026},
  url    = {https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning}
}
```
