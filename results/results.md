# Results

Auto-populated as experiments finish. Each row points to a config in `configs/` and a checkpoint under `outputs/`. Reproduce any row by running:

```bash
python -m src.train --config configs/<row_config>.yaml
python -m src.evaluate --checkpoint outputs/<exp_name>/best-*.ckpt --config configs/<row_config>.yaml
```

---

## Benchmark table

Metrics: **Weighted F1 (WF1)**, **Unweighted F1 / UAR (UF1)**, **Accuracy**.

### RAVDESS (8-class, random 70/10/20 split)

| Approach | Encoder | WF1 | UF1 | Accuracy | Config | Checkpoint |
|---|---|---|---|---|---|---|
| **Multimodal (cross-attn)** | RoBERTa + wav2vec2 | **0.858** | **0.851** | **0.858** | [`multimodal_ravdess.yaml`](../configs/multimodal_ravdess.yaml) | `best-11-0.8271.ckpt` |
| Audio-only | wav2vec2-base | 0.796 | 0.784 | 0.795 | [`audio_only_ravdess.yaml`](../configs/audio_only_ravdess.yaml) | `best-17-0.7869.ckpt` |
| Text-only (ablation) | RoBERTa-base | 0.053 | 0.053 | 0.132 | [`text_only_ravdess.yaml`](../configs/text_only_ravdess.yaml) | `best-01-0.0654.ckpt` |
| Audio-only | WavLM-base | – | – | – | _todo_ | — |
| Multimodal (concat) | RoBERTa + wav2vec2 | – | – | – | _todo_ | — |

> ⚠️ **Methodology note**: the current split is **random**, so the same actor can appear in train and test. This inflates results vs. published actor-independent benchmarks. A speaker-independent split (actors 1-20 train, 21-24 test) will follow once we add that splitter; expect a ~5-10pp drop, which is the honest number.

> ⚠️ **Text-only on RAVDESS is intentionally at chance** — RAVDESS has only 2 fixed sentences ("Kids are talking by the door" / "Dogs are sitting by the door"), so the input text carries zero emotion signal. This row is included as a deliberate ablation to demonstrate why pure-NLP approaches fail on acted, fixed-sentence SER.

### MELD (7-class, official dev/test splits)

| Approach | Encoder | WF1 | UF1 | Accuracy | Config |
|---|---|---|---|---|---|
| Text-only (no context) | RoBERTa-base | – | – | – | _todo_ |
| Text-only (context=2) | RoBERTa-base | – | – | – | [`text_only_meld.yaml`](../configs/text_only_meld.yaml) |
| Audio-only | WavLM-base | – | – | – | [`audio_only_meld.yaml`](../configs/audio_only_meld.yaml) |
| Multimodal (cross-attn) | RoBERTa + WavLM | – | – | – | [`multimodal_meld.yaml`](../configs/multimodal_meld.yaml) |

### IEMOCAP (4-class, leave-one-session-out CV)

| Approach | WF1 | UF1 | Notes |
|---|---|---|---|
| ⏳ Pending dataset access | – | – | License request pending; loader stub at `src/data/iemocap.py` |

---

## Detailed results — RAVDESS

### 🥇 Multimodal cross-attention (RoBERTa + wav2vec2)

**Config**: [`configs/multimodal_ravdess.yaml`](../configs/multimodal_ravdess.yaml)
**Checkpoint**: `outputs/multimodal_ravdess/best-11-0.8271.ckpt`
**Training**: 19 epochs (early-stopped from max 30, best at epoch 11), batch_size=8, accum=2, LR=2e-5, freeze 8/12 audio encoder layers, RoBERTa fully trainable, class-weighted CE loss. **221M params, ~50 min on Apple Silicon MPS.**

| Metric | Value |
|---|---|
| Accuracy | **0.8576** |
| Weighted F1 | **0.8579** |
| Unweighted F1 (UAR) | **0.8513** |

**Per-class F1 with delta vs audio-only**:

| Class | Multimodal F1 | Audio-only F1 | Δ | n |
|---|---|---|---|---|
| neutral | 0.7692 | 0.6667 | **+10.2pp** | 20 |
| calm | 0.8667 | 0.8352 | +3.2pp | 44 |
| happy | 0.8451 | 0.8000 | +4.5pp | 34 |
| sad | 0.8395 | 0.7297 | **+11.0pp** | 39 |
| angry | 0.8608 | 0.8095 | +5.1pp | 42 |
| fearful | 0.8525 | 0.7879 | +6.5pp | 32 |
| disgust | 0.8696 | 0.7541 | **+11.6pp** | 32 |
| surprised | **0.9070** | 0.8916 | +1.5pp | 45 |

**The key observation**: the multimodal gains concentrate on the previously-weakest classes (neutral, sad, disgust — all >+10pp). Classes that audio-only already handled well (surprised, calm, angry) gain only marginally. This is the signature of a useful regularizer / calibrator, not random noise.

### 🥈 Audio-only (wav2vec2-base)

**Config**: [`configs/audio_only_ravdess.yaml`](../configs/audio_only_ravdess.yaml)
**Checkpoint**: `outputs/audio_only_ravdess/best-17-0.7869.ckpt`
**Training**: 25 epochs (early-stopped from max 30, best at epoch 17), batch_size=8, accum=2, LR=2e-5, freeze 8/12 transformer layers, class-weighted CE loss. **94.4M params, ~40 min on MPS.**

| Metric | Value |
|---|---|
| Accuracy | 0.7951 |
| Weighted F1 | 0.7958 |
| Unweighted F1 (UAR) | 0.7843 |

**Per-class F1**:

| Class | F1 | n |
|---|---|---|
| neutral | 0.6667 | 20 |
| calm | 0.8352 | 44 |
| happy | 0.8000 | 34 |
| sad | 0.7297 | 39 |
| angry | 0.8095 | 42 |
| fearful | 0.7879 | 32 |
| disgust | 0.7541 | 32 |
| surprised | 0.8916 | 45 |

High-arousal classes (angry, surprised, fearful) dominate. Low-arousal (neutral, sad, calm) confuse — known acoustic-only prosodic similarity problem.

### 🥉 Text-only RoBERTa-base (deliberate ablation)

**Config**: [`configs/text_only_ravdess.yaml`](../configs/text_only_ravdess.yaml)
**Checkpoint**: `outputs/text_only_ravdess/best-01-0.0654.ckpt`
**Training**: 9 epochs (early-stopped from max 20; best at epoch 1 because val WF1 only got worse). **124M params, ~3 min on MPS.**

| Metric | Value |
|---|---|
| Accuracy | 0.1319 (≈ chance: 0.125) |
| Weighted F1 | 0.0527 |
| Unweighted F1 (UAR) | 0.0530 |

**Per-class F1**:

| Class | F1 | n |
|---|---|---|
| neutral | 0.0000 | 20 |
| calm | 0.0000 | 44 |
| happy | 0.0000 | 34 |
| sad | 0.2286 | 39 |
| angry | 0.0000 | 42 |
| fearful | 0.1957 | 32 |
| disgust | 0.0000 | 32 |
| surprised | 0.0000 | 45 |

**6 of 8 classes scored exactly 0.000** — the model collapsed to predicting `sad` or `fearful` for most inputs. This is the textbook degenerate solution when a classifier has no useful input signal. **Demonstrates conclusively that text-only SER fails on acted, fixed-sentence datasets** — RoBERTa cannot distinguish 8 emotions from 2 distinct sentences.

---

## Cross-experiment observations

A few findings worth highlighting for a future blog post or paper:

1. **Multimodal fusion helps even when one modality is uninformative.** Text-only alone is at chance (0.053 WF1), yet adding the text branch to audio improves WF1 from 0.796 to 0.858. The text branch isn't contributing semantic emotion signal — it's likely acting as either (a) a soft regularizer preventing the audio encoder from overfitting to spurious actor-specific acoustic patterns, or (b) a weak prior over the statement ID, helping calibrate per-input.

2. **Gains concentrate on previously-weakest classes.** This is the signature of a useful intervention rather than noise. If multimodal were just a lucky run, gains would be uniform or random across classes; instead, the model improves most on neutral/sad/disgust, which are the acoustically-ambiguous classes audio-only struggled with.

3. **Hyperparameter sensitivity in SSL fine-tuning is severe.** An early audio-only run with LR=1e-4 (the wav2vec2 pre-training LR) diverged — model unlearned representations, EarlyStopping triggered at epoch 4, "best" checkpoint was effectively initial weights (test WF1 0.27). LR=2e-5 with 8/12 frozen layers reached 0.796 WF1. **5x LR difference → 3x performance gap.** Worth a blog paragraph as a cautionary tale for first-time SSL fine-tuners.

---

## Notes from training

- **MPS backend**: Apple Silicon MPS engaged automatically for all three runs. Speed: ~1.4 it/s for audio-only, ~0.77 it/s for multimodal (2x params).
- **`pin_memory` warning** is benign on MPS — printed once per dataloader instantiation, then ignored.
- **wav2vec2 `UNEXPECTED` keys** (`quantizer.*`, `project_q.*`, `project_hid.*`) and RoBERTa `lm_head.*` keys are pre-training-only heads. They're correctly dropped for classification; ignore the warnings.
- **PyTorch-MPS "Unaligned blit request" bug** affected `load_from_checkpoint` for RoBERTa weights. Fixed by passing `map_location="cpu"` and letting Lightning move the model to MPS during `.fit()` / `.test()`. See `src/evaluate.py` and `demo/gradio_app.py`.
