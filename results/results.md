# Results

Auto-populated as experiments finish. Each row points to a config in `configs/` and a checkpoint under `outputs/`. Reproduce any row by running:

```bash
python -m src.train --config configs/<row_config>.yaml
python -m src.evaluate --checkpoint outputs/<exp_name>/best-*.ckpt --config configs/<row_config>.yaml
```

---

## Benchmark table

Metrics: **Weighted F1 (WF1)**, **Unweighted F1 / UAR (UF1)**, **Accuracy**.

### RAVDESS (8-class, speech-only, random 70/10/20 split)

| Approach | Encoder | WF1 | UF1 | Accuracy | Config | Checkpoint |
|---|---|---|---|---|---|---|
| Audio-only | wav2vec2-base | **0.7958** | **0.7843** | **0.7951** | [`audio_only_ravdess.yaml`](../configs/audio_only_ravdess.yaml) | `best-17-0.7869.ckpt` |
| Text-only | RoBERTa-base | – | – | – | [`text_only_ravdess.yaml`](../configs/text_only_ravdess.yaml) | — |
| Audio-only | WavLM-base | – | – | – | _todo_ | — |
| Multimodal (concat) | RoBERTa + wav2vec2 | – | – | – | _todo_ | — |
| Multimodal (cross-attn) | RoBERTa + wav2vec2 | – | – | – | [`multimodal_ravdess.yaml`](../configs/multimodal_ravdess.yaml) | — |

> ⚠️ **Methodology note**: the current split is **random**, so the same actor can appear in train and test. This inflates results vs. published actor-independent benchmarks. A speaker-independent split (actors 1-20 train, 21-24 test) will follow once we add that splitter; expect a ~5-10pp drop, which is the honest number.

> ⚠️ **Text-only on RAVDESS is expected to be near-chance** — RAVDESS has only 2 fixed sentences ("Kids are talking by the door" / "Dogs are sitting by the door"), so the ASR transcripts carry no lexical emotion signal. This is included as a deliberate ablation.

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

## Detailed results

### Audio-only on RAVDESS — wav2vec2-base

**Config**: [`configs/audio_only_ravdess.yaml`](../configs/audio_only_ravdess.yaml)
**Checkpoint**: `outputs/audio_only_ravdess/best-17-0.7869.ckpt`
**Training**: 25 epochs (early-stopped from max 30, best at epoch 17), batch_size=8, accum=2, LR=2e-5, freeze 8/12 encoder layers, class-weighted CE loss.
**Hardware**: Apple Silicon MPS, ~1.4 it/s, ~40 min total.

**Test metrics**:

| Metric | Value |
|---|---|
| Accuracy | **0.7951** |
| Weighted F1 | **0.7958** |
| Unweighted F1 (UAR) | **0.7843** |

**Per-class F1** (n = test samples):

| Class | F1 | n |
|---|---|---|
| neutral | 0.6667 | 20 |
| calm | 0.8352 | 44 |
| happy | 0.8000 | 34 |
| sad | 0.7297 | 39 |
| angry | 0.8095 | 42 |
| fearful | 0.7879 | 32 |
| disgust | 0.7541 | 32 |
| surprised | **0.8916** | 45 |

**Observations**:
- **High-arousal classes win** — angry, surprised, fearful all score 0.79+. Acoustic prosody (pitch, energy) is highly discriminative for these.
- **Low-arousal classes struggle** — neutral (0.667) and sad (0.730) confuse with calm. This is the well-known prosodic similarity problem: neutral and calm sound acoustically nearly identical in RAVDESS.
- **Neutral has only 20 test samples** — partly a count artifact; small support inflates variance.
- Confusion matrix: see [`outputs/audio_only_ravdess/eval/confusion_matrix.png`](#) (artifact, not committed).

**What's missing** (for the next experiment): the same model with `freeze_encoder_layers: 4` (more capacity) or wav2vec2-**large** (4x the params) would likely close the gap on the low-arousal classes. The current run is intentionally conservative.

---

### Earlier (diverged) run — kept for reference

The first run used LR=1e-4 (the pre-training LR, way too high for fine-tuning) and no frozen layers. Result: model unlearned over epochs, EarlyStopping kicked in at epoch 4, "best" checkpoint was the **initial weights** before training did damage. Test WF1: 0.27. Archived at `outputs/audio_only_ravdess_v1_lr1e4_diverged/` for the blog post on hyperparameter sensitivity in SSL fine-tuning.

**Lesson** (worth a blog paragraph): wav2vec2 fine-tuning has a *much* lower LR than the original pre-training. 2e-5 is the right ballpark; 1e-4 destroys the learned representations. This is documented in HF tutorials but easy to miss when treating the model as a generic encoder.

---

## Notes from training

- **MPS backend**: Apple Silicon MPS engaged automatically for both training and evaluation. Speed: ~1.4 it/s for audio-only with batch_size=8.
- **`pin_memory` warning** is benign on MPS — Lightning prints it once, then ignores.
- **wav2vec2 `UNEXPECTED` keys** at load time (`quantizer.*`, `project_q.*`, `project_hid.*`) are the pre-training-only contrastive heads. They're correctly dropped for classification; ignore the warning.
- **`Modules in eval mode at start of training`** is the frozen feature extractor (CNN front-end) staying in eval. Intentional given `freeze_feature_extractor: true`.
