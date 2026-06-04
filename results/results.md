# Results

This document is **auto-populated** as experiments finish. Run all baselines with:

```bash
bash scripts/run_all_experiments.sh
```

Each experiment writes a checkpoint and metrics JSON under `outputs/<experiment_name>/`.
After training, run `python -m src.evaluate --checkpoint <ckpt> --config <cfg>` to write
confusion matrices and per-class scores.

---

## Benchmark table

> 🚧 Pending — populate as runs complete. Use **Weighted F1 (WF1)** and **Unweighted F1 / UAR (UF1)**.

### RAVDESS (8-class, speech-only, actor-independent split)

| Approach | Encoder | WF1 | UF1 | Accuracy |
|---|---|---|---|---|
| Text-only | RoBERTa-base | – | – | – |
| Audio-only | wav2vec2-base | – | – | – |
| Audio-only | WavLM-base | – | – | – |
| Multimodal (concat) | RoBERTa + wav2vec2 | – | – | – |
| Multimodal (cross-attn) | RoBERTa + wav2vec2 | – | – | – |

> ⚠️ Text-only on RAVDESS is **expected to be near-chance** — RAVDESS has only 2 fixed sentences,
> so the lexical channel carries no emotion signal. This is included as a deliberate ablation.

### MELD (7-class, official dev/test splits)

| Approach | Encoder | WF1 | UF1 | Accuracy |
|---|---|---|---|---|
| Text-only (no context) | RoBERTa-base | – | – | – |
| Text-only (context=2) | RoBERTa-base | – | – | – |
| Audio-only | WavLM-base | – | – | – |
| Multimodal (cross-attn) | RoBERTa + WavLM | – | – | – |

### IEMOCAP (4-class, leave-one-session-out CV)

| Approach | WF1 | UF1 | Notes |
|---|---|---|---|
| ⏳ Pending dataset access | – | – | License request in progress |

---

## Per-class breakdown

Confusion matrices live under `outputs/<experiment>/eval/confusion_matrix.png`.

---

## Notes from training

Add observations here as runs complete — class imbalance behavior, where the model
struggles, how cross-attention compares to concat, ASR quality impact on text-only, etc.
