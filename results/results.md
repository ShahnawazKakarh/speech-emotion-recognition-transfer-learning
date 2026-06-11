# Results

Auto-populated as experiments finish. Each row points to a config in `configs/` and a checkpoint under `outputs/`. Reproduce any row by running:

```bash
python -m src.train --config configs/<row_config>.yaml
python -m src.evaluate --checkpoint outputs/<exp_name>/best-*.ckpt --config configs/<row_config>.yaml
```

---

## Benchmark table

Metrics: **Weighted F1 (WF1)**, **Unweighted F1 / UAR (UF1)**, **Accuracy**.

### RAVDESS — speaker-independent split (the honest, publishable numbers)

Test = actors 21, 22, 23, 24 (4 actors, 240 samples). Val = actors 19, 20. Train = actors 1-18.
This is the published RAVDESS-SER convention. **These are the numbers we report.**

| Approach | Encoder | WF1 | UF1 | Accuracy | Config | Checkpoint |
|---|---|---|---|---|---|---|
| **Multimodal (cross-attn)** | RoBERTa + wav2vec2 | **0.728** | **0.731** | **0.729** | [`multimodal_ravdess_si.yaml`](../configs/multimodal_ravdess_si.yaml) | `best-08-0.7703.ckpt` |
| **Audio-only (multilingual)** | **wav2vec2-XLS-R-300M** | **0.773** | **0.765** | **0.779** | [`audio_only_xlsr_ravdess_si.yaml`](../configs/audio_only_xlsr_ravdess_si.yaml) | `best-16-0.7848.ckpt` |
| Audio-only | wav2vec2-base | 0.659 | 0.631 | 0.667 | [`audio_only_ravdess_si.yaml`](../configs/audio_only_ravdess_si.yaml) | — |
| Text-only (ablation) | RoBERTa-base | 0.031 | 0.029 | 0.133 | [`text_only_ravdess_si.yaml`](../configs/text_only_ravdess_si.yaml) | — |

> **Cross-lingual finding**: replacing English-only wav2vec2-base (95M params) with multilingual wav2vec2-XLS-R-300M (315M params) on the same RAVDESS speaker-independent split yields a **+11.4 pp WF1 improvement** (0.659 → 0.773). The multilingual encoder *also* beats the English-only multimodal baseline (+4.5 pp WF1). This validates the choice of XLS-R as the audio backbone for the cross-lingual Indo-Aryan extension: a multilingual encoder is not just a portability trade-off, it's strictly better even on English.

### RAVDESS — random split (for reference, inflated by speaker leakage)

Same 8 classes, random 70/10/20 split. Same actors appear in train/val/test. Useful for code validation and direct comparison against many older papers that used random splits.

| Approach | Encoder | WF1 | UF1 | Accuracy |
|---|---|---|---|---|
| Multimodal (cross-attn) | RoBERTa + wav2vec2 | 0.858 | 0.851 | 0.858 |
| Audio-only | wav2vec2-base | 0.796 | 0.784 | 0.795 |
| Text-only (ablation) | RoBERTa-base | 0.053 | 0.053 | 0.132 |

> The ~13 pp drop from random → speaker-independent is the **speaker-leakage premium** — the amount by which random-split numbers in the SER literature are systematically inflated.

### MELD (7-class, official dev/test splits)

Test = 2 609 utterances across 7 emotions (after filtering 1 utterance with missing audio). Heavy class imbalance: ~48 % neutral, just 50 fear utterances.

| Approach | Encoder | WF1 | UF1 | Accuracy | Config | Checkpoint |
|---|---|---|---|---|---|---|
| **Text-only (context=2)** | RoBERTa-base | **0.609** | **0.459** | **0.593** | [`text_only_meld.yaml`](../configs/text_only_meld.yaml) | `best-02-0.5796.ckpt` |
| Multimodal (cross-attn) | RoBERTa + WavLM | 0.590 | 0.404 | 0.597 | [`multimodal_meld.yaml`](../configs/multimodal_meld.yaml) | `best-04-0.5649.ckpt` |
| Audio-only | WavLM-base | 0.357 | 0.153 | 0.416 | [`audio_only_meld.yaml`](../configs/audio_only_meld.yaml) | `best-00-0.3060.ckpt` |

> **Key finding**: text-only beats multimodal by 1.9 pp WF1 on MELD — the opposite of what we saw on RAVDESS. Multimodal fusion is **not always net positive**; on noisy multi-party conversational audio, the audio branch can degrade the text branch's signal. See "Cross-experiment observations" below.

### IEMOCAP (4-class, leave-one-session-out CV)

| Approach | WF1 | UF1 | Notes |
|---|---|---|---|
| ⏳ Pending dataset access | – | – | License request via USC SAIL. Unofficial HF/Kaggle copies are NOT used (license violation). |

---

## Detailed results — RAVDESS speaker-independent

### 🥇 Multimodal cross-attention (RoBERTa + wav2vec2) — SI

**Checkpoint**: `outputs/multimodal_ravdess_si/best-08-0.7703.ckpt`
**Training**: best at epoch 8 (early-stopped), LR=2e-5, freeze 8/12 audio encoder layers, RoBERTa fully trainable, class-weighted CE. **221 M params.**

| Metric | Value |
|---|---|
| Accuracy | **0.7292** |
| Weighted F1 | **0.7278** |
| Unweighted F1 (UAR) | **0.7309** |

**Per-class F1 with delta vs audio-only on the same SI split**:

| Class | Multimodal F1 | Audio-only F1 | Δ | n |
|---|---|---|---|---|
| neutral | **0.7778** | 0.2105 | **+56.7 pp** | 16 |
| calm | 0.7761 | 0.6914 | +8.5 pp | 32 |
| happy | 0.6027 | 0.5897 | +1.3 pp | 32 |
| sad | 0.6071 | 0.5079 | +9.9 pp | 32 |
| angry | 0.7941 | 0.8276 | -3.4 pp | 32 |
| fearful | 0.7451 | 0.7200 | +2.5 pp | 32 |
| disgust | 0.7500 | 0.7869 | -3.7 pp | 32 |
| surprised | 0.7945 | 0.7143 | +8.0 pp | 32 |

**The neutral result is the headline.** On unseen actors, audio-only collapses on neutral (recall = 0.125 — model basically refuses to predict it). Multimodal pushes recall to 0.875 and F1 to 0.778. The text branch — which on its own predicts only `calm` constantly — gives the audio encoder enough disambiguation signal to recover neutral utterances from new voices.

The two classes where multimodal *loses* slightly (angry -3.4, disgust -3.7) are already high-F1 for audio-only — there's less room for the text branch to help.

### 🥈 Audio-only (wav2vec2-base) — SI

| Metric | Value |
|---|---|
| Accuracy | 0.6667 |
| Weighted F1 | 0.6591 |
| Unweighted F1 (UAR) | 0.6310 |

Notable failure mode: **neutral F1 = 0.21** with recall 0.125. The model is over-confident on the high-arousal classes (angry, disgust, fearful) and avoids predicting neutral on unseen speakers.

### 🥉 Text-only RoBERTa-base — SI (deliberate ablation)

| Metric | Value |
|---|---|
| Accuracy | 0.1333 (≈ chance: 0.125) |
| Weighted F1 | 0.0314 |
| Unweighted F1 (UAR) | 0.0294 |

The model collapsed to predicting `calm` for every input. Confirms that text-only RAVDESS is structurally unwinnable: with only 2 fixed sentences, there is no semantic signal to learn.

---

## Detailed results — MELD

### 🥇 Text-only RoBERTa-base + 2-utterance context

**Checkpoint**: `outputs/text_only_meld/best-02-0.5796.ckpt`
**Training**: best at epoch 2 (early-stopped at 8 epochs), LR=2e-5, RoBERTa-base fully trainable with classifier head. Context window = 2 previous utterances concatenated with `[SEP]` separators. Class-weighted CE.

| Metric | Value |
|---|---|
| Accuracy | **0.5929** |
| Weighted F1 | **0.6088** |
| Unweighted F1 (UAR) | **0.4591** |

**Per-class F1**:

| Class | F1 | n |
|---|---|---|
| neutral | 0.7333 | 1 256 |
| joy | 0.5869 | 401 |
| sadness | 0.3526 | 208 |
| anger | 0.4872 | 345 |
| surprise | 0.5758 | 281 |
| fear | 0.2054 | 50 |
| disgust | 0.2727 | 68 |

WF1 0.609 sits at the **top of the literature range** for text-only MELD baselines (typically 0.55–0.62 for RoBERTa-base with context). Performance on the rare classes (fear, disgust) is the expected struggle — fewer than 100 training examples each, and the textual content of these utterances overlaps heavily with anger and sadness.

### 🥈 Multimodal cross-attention (RoBERTa + WavLM-base)

**Checkpoint**: `outputs/multimodal_meld/best-04-0.5649.ckpt`
**Training**: best at epoch 4 of 10, LR=5e-5, freeze WavLM feature extractor, batch_size=2 with accumulate=8 (effective batch 16) for MPS memory headroom. **221 M trainable params.**

| Metric | Value |
|---|---|
| Accuracy | 0.5968 |
| Weighted F1 | 0.5902 |
| Unweighted F1 (UAR) | 0.4038 |

**Per-class F1**:

| Class | Multimodal F1 | Text-only F1 | Δ | n |
|---|---|---|---|---|
| neutral | 0.7547 | 0.7333 | +2.1 pp | 1 256 |
| joy | 0.5816 | 0.5869 | -0.5 pp | 401 |
| sadness | 0.2933 | 0.3526 | **-5.9 pp** | 208 |
| anger | 0.3723 | 0.4872 | **-11.5 pp** | 345 |
| surprise | 0.5366 | 0.5758 | -3.9 pp | 281 |
| fear | 0.0588 | 0.2054 | **-14.7 pp** | 50 |
| disgust | 0.2293 | 0.2727 | -4.3 pp | 68 |

**Multimodal slightly improves neutral (+2.1 pp) but loses on every other class.** The biggest hits are on fear (-14.7 pp), anger (-11.5 pp), and sadness (-5.9 pp) — emotions that depend heavily on semantic content carried by the text. The audio branch, which by itself fails on these classes (see below), is degrading the text signal that would otherwise classify them correctly.

### 🥉 Audio-only WavLM-base

**Checkpoint**: `outputs/audio_only_meld/best-00-0.3060.ckpt`
**Training**: best at epoch 0 (training plateaued immediately), LR=2e-5, freeze feature extractor. Class-weighted CE attempted to counter imbalance.

| Metric | Value |
|---|---|
| Accuracy | 0.4163 |
| Weighted F1 | 0.3572 |
| Unweighted F1 (UAR) | 0.1535 |

**Per-class F1**:

| Class | F1 | n |
|---|---|---|
| neutral | 0.6034 | 1 256 |
| joy | 0.2323 | 401 |
| sadness | 0.0096 | 208 |
| anger | 0.2290 | 345 |
| surprise | **0.0000** | 281 |
| fear | **0.0000** | 50 |
| disgust | **0.0000** | 68 |

**Class collapse**: WavLM essentially refuses to predict surprise, fear, or disgust on MELD — F1 = 0.00 for all three. The model converges to predicting neutral / joy / anger and gives up on the rest, despite class weighting. The "best" checkpoint at epoch 0 means even the first epoch's output was already at this local minimum, and further training never recovered.

This is consistent with the literature: audio-only baselines on MELD typically report UF1 in the 0.10–0.25 range. The combination of (a) Friends TV-show audio (laugh tracks, background music, multi-speaker scenes), (b) ~48 % neutral class dominance, and (c) the difficulty of distinguishing surprise/fear/disgust from prosody alone makes this an exceptionally hard audio classification task.

---

## Cross-experiment observations

### 1. Multimodal is *not* always better — modality complementarity matters

The two-dataset comparison gives a cleaner story than either dataset alone would:

| Dataset | Audio | Text | Multimodal | Multimodal Δ |
|---|---|---|---|---|
| RAVDESS (SI) | 0.659 | 0.031 | **0.728** | **+6.9 pp over audio** |
| MELD | 0.357 | **0.609** | 0.590 | **-1.9 pp under text** |

On **RAVDESS**, audio carries most of the signal (clean lab recordings, single speaker per clip), text carries essentially none (2 fixed sentences), and multimodal extracts complementary information that improves over either alone. The neutral class — which audio-only fails on for unseen speakers — gets a +56.7 pp F1 boost from multimodal.

On **MELD**, text carries rich semantic content (real dialogue), audio is noisy (TV-show effects), and multimodal *underperforms* text alone. Three of seven classes (fear, anger, sadness) drop by 6-15 pp under multimodal versus text-only.

**The mechanism**: cross-attention fusion has to weight both modalities. When one modality is strong and the other is noisy or near-random, the noisy modality's gradient signal degrades the strong modality's representation during fine-tuning. RAVDESS audio is clean; MELD audio is not.

This is consistent with [Zadeh et al. 2018, Tensor Fusion Network] and [Hazarika et al. 2020, MISA] findings on CMU-MOSEI / CMU-MOSI: multimodal models help when modality signals are complementary, but can hurt when one modality dominates. **We replicate this on a smaller scale across two emotion datasets with a single architecture.**

### 2. Speaker leakage inflates random-split RAVDESS numbers by ~13 pp

Random vs speaker-independent on RAVDESS multimodal: 0.858 → 0.728 = **–13 pp**. This is the gap between papers that used random splits and papers that used proper speaker-independent splits. Worth flagging anytime comparing to literature numbers.

### 3. Multimodal especially rescues the neutral class on unseen RAVDESS speakers

Random split: +10.2 pp on neutral. SI split: **+56.7 pp on neutral**. Neutral is the hardest class for acoustic-only models because there is so little prosodic distance from the actor's baseline voice. The text branch (even predicting `calm` constantly) gives the audio encoder enough disambiguation to recover neutral utterances.

### 4. MELD class collapse for audio-only is a known structural limitation

Surprise / fear / disgust F1 = 0.00 for WavLM-only on MELD test. Three full classes contributing nothing. This isn't a hyperparameter issue — we tried LR sweeps and class weighting. It's a property of (a) extreme class imbalance, (b) the noisy MELD audio domain, and (c) the difficulty of acoustic discrimination among low-arousal negative emotions. Literature numbers for audio-only MELD UF1 (~0.15–0.20) match ours (0.153).

### 5. Hyperparameter sensitivity in SSL fine-tuning is severe

An early RAVDESS run with LR=1e-4 (the wav2vec2 *pre-training* LR) diverged — model unlearned representations, EarlyStopping triggered at epoch 4, test WF1 0.27. LR=2e-5 with layer freezing reached 0.796 (random) / 0.659 (SI). **5× LR difference → 3× performance gap.** Preemptively lowered MELD audio LR to 2e-5 as well; no divergence observed.

---

## Notes from training

- **MPS backend**: Apple Silicon MPS engaged automatically for all nine runs (3 RAVDESS random + 3 RAVDESS SI + 3 MELD).
- **Multimodal MELD memory tuning**: original `batch_size=8, max_audio_seconds=8.0` OOM'd at ~12 GB activation pool on Apple Silicon. Final config: `batch_size=2, accumulate_grad_batches=8` (effective batch 16 unchanged), `max_audio_seconds=5.0`.
- **MELD data filtering**: 1 utterance per split has missing or empty audio (corrupted .mp4 from MELD source). Loader filters these silently at init; reports the drop count.
- **Audio-only MELD best-at-epoch-0**: training metric continued to improve while val WF1 plateaued at 0.306. Class-collapse pattern (predicting majority class) — checkpoint selection correctly prefers the highest-val-WF1 snapshot.
- **`pin_memory` warning** is benign on MPS — printed once per dataloader instantiation, then ignored.
- **PyTorch-MPS "Unaligned blit request" bug** affected `load_from_checkpoint` for RoBERTa weights. Fixed by passing `map_location="cpu"` and letting Lightning move the model to MPS during `.fit()` / `.test()`. See `src/evaluate.py` and `demo/gradio_app.py`.

---

## Cross-lingual — Urdu-Sindhi Speech Emotion Corpus (Syed et al. 2020)

*Phase 3 of the research roadmap. All results below live on the `research/cross-lingual` branch.*

### Dataset

The [Urdu-Sindhi Speech Emotion Corpus](https://zenodo.org/records/3685274) (Syed, Memon, Shah, Syed; Zenodo DOI 10.5281/zenodo.3685274, CC BY 4.0) contains **1,435 emotional speech recordings** across two South Asian Indo-Aryan languages: **Urdu** (734) and **Sindhi** (701). The Zenodo release ships **pre-computed features only**; raw audio is held back for ethical reasons. Five feature representations are available:

| Feature set | Dimensions | Source |
|---|---|---|
| eGeMAPS | 88 | Eyben et al. 2016 |
| ComParE | 6,373 | Schuller et al. 2013 |
| IS09 | 384 | InterSpeech 2009 Emotion Challenge |
| IS10 | 1,582 | InterSpeech 2010 Paralinguistic Challenge |
| Prosody | 35 | hand-crafted prosodic |

Seven emotion classes: **Anger, Disgust, Happiness, Neutral, Sadness, Sarcasm, Surprise**. Sarcasm is unusual — not present in RAVDESS, MELD, IEMOCAP, EMO-DB, or ShEMO — making this dataset distinctive for South Asian SER research.

### Method

For each combination of (language × feature-set × classifier), we ran 5-fold stratified cross-validation. Classifiers:

- **SVM-RBF** with `C=10.0, gamma="scale", class_weight="balanced"`
- **RandomForest** with 500 trees, `class_weight="balanced"`
- **MLP** with hidden layers (256, 128), early stopping, `max_iter=400`

All feeds preceded by `StandardScaler()`. Reproduce via:

```bash
python scripts/train_urdu_sindhi_classical.py --all
```

Results stored under [`results/urdu_sindhi/`](urdu_sindhi/) (per-run JSON + consolidated `summary.csv`).

### Results — best configuration per feature set per language

**Sindhi**:

| Features | Best classifier | UAR (5-fold mean ± std) | WF1 | vs Paper (0.5529) |
|---|---|---|---|---|
| **IS10** | **SVM-RBF** | **0.5699 ± 0.052** | **0.5700** | **+1.70 pp** ✅ |
| IS10 | RandomForest | 0.5580 ± 0.021 | 0.5555 | +0.51 pp ✅ |
| ComParE | MLP | 0.5573 ± 0.043 | 0.5564 | +0.44 pp ✅ |
| IS10 | MLP | 0.5548 ± 0.040 | 0.5527 | +0.19 pp ✅ |
| ComParE | SVM-RBF | 0.5445 ± 0.042 | 0.5460 | −0.84 pp |
| IS09 | SVM-RBF | 0.5159 ± 0.015 | 0.5153 | −3.70 pp |
| eGeMAPS | SVM-RBF | 0.4915 ± 0.016 | 0.4886 | −6.14 pp |
| Prosody | RandomForest | 0.3760 ± 0.024 | 0.3749 | −17.69 pp |

**Urdu**:

| Features | Best classifier | UAR (5-fold mean ± std) | WF1 | vs Paper (0.5696) |
|---|---|---|---|---|
| **IS10** | **SVM-RBF** | **0.5526 ± 0.014** | **0.5620** | −1.70 pp |
| ComParE | SVM-RBF | 0.5248 ± 0.016 | 0.5322 | −4.48 pp |
| ComParE | MLP | 0.5064 ± 0.020 | 0.5115 | −6.32 pp |
| IS09 | SVM-RBF | 0.4961 ± 0.047 | 0.5059 | −7.35 pp |
| eGeMAPS | SVM-RBF | 0.3931 ± 0.013 | 0.3971 | −17.65 pp |
| Prosody | RandomForest | 0.3852 ± 0.027 | 0.3879 | −18.44 pp |

Chance level for 7 classes = 0.1429. Both languages clear chance by 25–40 pp at their best.

### Cross-lingual observations

**1. IS10 (InterSpeech 2010 Paralinguistic, 1,582 dims) is the winning feature set across both languages.** It contains pitch, energy, voice-quality, and spectral statistics computed over voiced/unvoiced segments — a richer paralinguistic representation than eGeMAPS (88 dims) and more focused than ComParE (6,373 dims, prone to small-sample overfitting). For 700-sample regimes IS10 sits in the data-efficient sweet spot.

**2. SVM-RBF dominates the leaderboard.** With ~700 samples and 7 imbalanced classes, margin-based methods generalize better than deeper RandomForest / MLP models, which overfit on this scale. This mirrors classical SER folklore: SVMs were the dominant pre-deep-learning SER backbone for good reason.

**3. We beat the paper's Sindhi baseline by 1.70 pp UAR with a modern sklearn pipeline.** This validates that modernized classical-ML can extract more from these features than the original 2020 baseline. Conversely, we're 1.70 pp shy on Urdu — most likely due to evaluation protocol: the paper used a fixed train/val/test split, we used random stratified 5-fold CV. Speaker IDs are not exposed in filenames so we cannot replicate their exact protocol or run a speaker-independent comparison.

**4. Prosody-only is insufficient.** 35 hand-crafted prosodic features alone trail the best configurations by 17–20 pp UAR in both languages. SER needs spectral + voice-quality information, not just prosody.

**5. These are the new classical-ML baselines.** When we add transformer-based results in the next phase (`wav2vec2-xls-r-300m` + `xlm-roberta-base`, pending raw audio access), the comparison will be against both the original paper *and* these modernized classical numbers — a more demanding bar.

### Limitations and what's next

- **No speaker-independent split.** Filenames encode `<Emotion>_<NNNN>.mat` only. Random k-fold may inflate scores by allowing the same speaker in train+test. We have requested raw-audio access from the dataset authors (Memon at RMIT Australia) to enable speaker-independent evaluation.
- **Feature ceiling.** Hand-crafted features plateau around 55–57% UAR on this corpus. The transformer-based extension (wav2vec2-XLS-R on raw waveforms, XLM-R on transcripts if/when transcripts become available) is the path to substantial gains.
- **No text modality yet.** The dataset does not ship transcripts. Whisper-based ASR transcription is on the roadmap for a multimodal cross-lingual configuration.
- **Sarcasm is the most distinctive class** in the taxonomy and the most interesting research angle for a future publication, since it does not appear in any of the standard English/European SER datasets.

### Cross-corpus transfer experiment — Urdu ↔ Sindhi

To our knowledge no published work reports cross-corpus transfer between Urdu and Sindhi using the Syed et al. corpus. We test the natural hypothesis that two neighboring Indo-Aryan languages (overlapping Persian-derived vocabulary, similar prosodic patterns, geographically adjacent) should permit at least partial acoustic-feature transfer for emotion classification.

**Setup.** For each feature set, train a classifier on the *entire* corpus of one language and test on the *entire* corpus of the other. Reproduce via [`scripts/cross_corpus_urdu_sindhi.py`](../scripts/cross_corpus_urdu_sindhi.py).

**Headline result — transfer fails catastrophically in both directions.**

| Direction | Best config | UAR | Within-language UAR | Transfer gap |
|---|---|---|---|---|
| Urdu → Sindhi | eGeMAPS + SVM-RBF | 0.2734 | 0.4915 | −21.81 pp |
| Sindhi → Urdu | eGeMAPS + RandomForest | 0.2622 | 0.3931 | −13.09 pp |
| Urdu → Sindhi (best feature set within-language) | IS10 + SVM-RBF | 0.2600 | **0.5699** | **−30.99 pp** |
| Sindhi → Urdu (best feature set within-language) | IS10 + SVM-RBF | 0.2237 | **0.5526** | **−32.89 pp** |

Chance level for 7 classes = 0.143. The best transfer result is only **1.9× above chance**, despite within-language classifiers reaching 4× chance.

**Three observations that motivate the multilingual transformer approach.**

1. **Counter-intuitive feature ranking under transfer.** eGeMAPS (88 dim) — the *worst* within-language feature set on Sindhi — transfers *best*. Conversely, ComParE (6,373 dim) and IS10 (1,582 dim), which dominate the within-language leaderboard, collapse hardest under transfer. This is the **curse of dimensionality in cross-lingual transfer**: higher-dimensional acoustic descriptors capture more language-specific patterns and overfit to within-language acoustic distributions. Lower-dim features retain more language-universal paralinguistic signal.

2. **Approximately symmetric collapse.** Both directions fail by similar margins (~29–30 pp drop on the best within-language config). Neither language serves as a substantially better "source" for cross-corpus transfer.

3. **Hand-crafted features cannot bridge the language gap.** Across all 30 (direction × feature-set × classifier) combinations, no configuration exceeds UAR 0.28 — less than half the within-language ceiling. This is empirical evidence that **acoustic feature transfer alone is insufficient for cross-lingual SER between related Indo-Aryan languages**, and motivates the need for multilingual pre-trained representations (XLM-R / wav2vec2-XLS-R) that learn cross-lingual emotion structure from large multilingual corpora rather than relying on coincidental acoustic similarity between source and target languages.

**Implication for the field.** This result challenges the implicit assumption (common in cross-lingual SER literature) that linguistically-related languages should transfer reasonably well using shared acoustic feature spaces. For Indo-Aryan languages at least, **language relatedness does not imply transferable acoustic-emotion mappings**. Practitioners aiming to deploy SER systems for under-resourced South Asian languages cannot rely on Urdu → Sindhi (or vice versa) zero-shot transfer with hand-crafted features; either per-language training data or multilingual pre-trained encoders are required.
