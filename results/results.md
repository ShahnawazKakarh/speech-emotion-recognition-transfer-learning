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
| Audio-only | wav2vec2-base | 0.659 | 0.631 | 0.667 | [`audio_only_ravdess_si.yaml`](../configs/audio_only_ravdess_si.yaml) | — |
| Text-only (ablation) | RoBERTa-base | 0.031 | 0.029 | 0.133 | [`text_only_ravdess_si.yaml`](../configs/text_only_ravdess_si.yaml) | — |
| Audio-only | WavLM-base | – | – | – | _todo_ | — |
| Multimodal (concat) | RoBERTa + wav2vec2 | – | – | – | _todo_ | — |

### RAVDESS — random split (for reference, inflated by speaker leakage)

Same 8 classes, random 70/10/20 split. Same actors appear in train/val/test. Useful for code validation and direct comparison against many older papers that used random splits.

| Approach | Encoder | WF1 | UF1 | Accuracy |
|---|---|---|---|---|
| Multimodal (cross-attn) | RoBERTa + wav2vec2 | 0.858 | 0.851 | 0.858 |
| Audio-only | wav2vec2-base | 0.796 | 0.784 | 0.795 |
| Text-only (ablation) | RoBERTa-base | 0.053 | 0.053 | 0.132 |

> The ~13 pp drop from random → speaker-independent is the **speaker-leakage premium** — the amount by which random-split numbers in the SER literature are systematically inflated. Worth its own paragraph in any future write-up.

### MELD (7-class, official dev/test splits)

| Approach | Encoder | WF1 | UF1 | Config |
|---|---|---|---|---|
| Text-only (context=2) | RoBERTa-base | – | – | [`text_only_meld.yaml`](../configs/text_only_meld.yaml) |
| Audio-only | WavLM-base | – | – | [`audio_only_meld.yaml`](../configs/audio_only_meld.yaml) |
| Multimodal (cross-attn) | RoBERTa + WavLM | – | – | [`multimodal_meld.yaml`](../configs/multimodal_meld.yaml) |

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

The two classes where multimodal *loses* slightly (angry -3.4, disgust -3.7) are already high-F1 for audio-only — there's less room for the text branch to help, and a bit of representational noise from the cross-attention can edge things down.

### 🥈 Audio-only (wav2vec2-base) — SI

**Test metrics**:

| Metric | Value |
|---|---|
| Accuracy | 0.6667 |
| Weighted F1 | 0.6591 |
| Unweighted F1 (UAR) | 0.6310 |

**Per-class F1**: see table above. **Notable failure mode: neutral F1 = 0.21** with recall 0.125. The model is over-confident on the high-arousal classes (angry, disgust, fearful) and avoids predicting neutral on unseen speakers.

### 🥉 Text-only RoBERTa-base — SI (deliberate ablation)

**Test metrics**:

| Metric | Value |
|---|---|
| Accuracy | 0.1333 (≈ chance: 0.125) |
| Weighted F1 | 0.0314 |
| Unweighted F1 (UAR) | 0.0294 |

The model collapsed to predicting `calm` for every input (recall(calm)=1.00, all other classes 0.00). Even more degenerate than the random-split text-only run — confirms that text-only RAVDESS is structurally unwinnable, not just hard.

---

## Cross-experiment observations (updated)

1. **Multimodal advantage *grows* on speaker-independent split.** Random: multimodal +6.2 pp over audio-only. SI: multimodal **+6.9 pp**. The fusion is more useful, not less, when the test set has unseen speakers. This is the strongest evidence that multimodal cross-attention isn't just learning a different inductive bias — it's genuinely extracting complementary signal that generalizes.

2. **Speaker leakage inflates random-split numbers by ~13 pp.** This is the published-vs-random gap. Worth flagging anytime you compare to papers in the literature: if the paper used a random split on RAVDESS and reports 0.85 WF1, that's roughly equivalent to a 0.72 WF1 on a proper speaker-independent split.

3. **Multimodal especially helps the *neutral* class on unseen speakers.** Random split: +10.2 pp on neutral. SI split: **+56.7 pp on neutral**. Neutral is the hardest class for acoustic-only models on unseen voices because there's so little prosodic distinction from the actor's baseline voice. The text branch (even predicting `calm` constantly) provides a useful prior.

4. **Hyperparameter sensitivity in SSL fine-tuning is severe.** An early run with LR=1e-4 (the wav2vec2 *pre-training* LR) diverged — model unlearned representations, EarlyStopping triggered at epoch 4, test WF1 0.27. LR=2e-5 with 8/12 layers frozen reached 0.796 (random) / 0.659 (SI). **5× LR difference → 3× performance gap.**

5. **MELD's class imbalance is going to dominate when we get there** (~48 % neutral utterances). UF1 will be the more honest metric than WF1, and the confusion matrices will tell the real story.

---

## Notes from training

- **MPS backend**: Apple Silicon MPS engaged automatically for all six RAVDESS runs (3 random + 3 SI).
- **`pin_memory` warning** is benign on MPS — printed once per dataloader instantiation, then ignored.
- **wav2vec2 `UNEXPECTED` keys** (`quantizer.*`, `project_q.*`, `project_hid.*`) and RoBERTa `lm_head.*` keys are pre-training-only heads. They're correctly dropped for classification; ignore the warnings.
- **PyTorch-MPS "Unaligned blit request" bug** affected `load_from_checkpoint` for RoBERTa weights. Fixed by passing `map_location="cpu"` and letting Lightning move the model to MPS during `.fit()` / `.test()`. See `src/evaluate.py` and `demo/gradio_app.py`.
