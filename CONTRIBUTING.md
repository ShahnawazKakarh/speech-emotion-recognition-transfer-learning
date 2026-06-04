# Contributing

Thanks for your interest in this project. Contributions of all sizes are welcome — bug fixes, new model architectures, new dataset loaders, documentation, or just better notebooks.

## Development setup

```bash
git clone https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning.git
cd speech-emotion-recognition-transfer-learning

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,demo]"

# Verify everything is wired correctly
pytest tests/ -v
```

## Code style

This repo uses a light, ML-friendly configuration:

| Tool   | Purpose                          | Run via                     |
|--------|----------------------------------|-----------------------------|
| black  | Code formatting (100-col)        | `black src tests demo`      |
| ruff   | Linting (E, F, I, W, UP, B, SIM) | `ruff check src tests`      |
| pytest | Tests                            | `pytest tests/ -v`          |

All three run in CI on every push and PR. Black is **advisory** in CI (won't fail the build), but please format before pushing.

If you're using VS Code, add this to `.vscode/settings.json`:

```json
{
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
  }
}
```

## Branching and commits

- **Branch from `master`**, name your branch with a short prefix: `feat/`, `fix/`, `docs/`, `ci/`, `chore/`. Examples: `feat/iemocap-loader`, `fix/mps-precision-bug`.
- **Conventional commits** for messages: `feat(models): add HuBERT encoder`, `fix(data): handle empty MELD utterances`, `docs(readme): add iemocap results`.
- **PR title** mirrors the commit style. A short body explains *why*, not just *what*.

## Adding a new dataset

1. Create `src/data/<dataset_name>.py` with a `<Name>Dataset(BaseSERDataset)` class. Use `RAVDESSDataset` or `MELDDataset` as a template.
2. Define the label list, label→id mapping, and id→label mapping as module-level constants.
3. Wire it into `src/data/datamodule.py` `SERDataModule.setup()`.
4. Add label name lookup to `src/train.py` `get_label_names()`.
5. Add a config under `configs/` — one per (modality × dataset) combination.
6. Add a download script under `scripts/` if the dataset is publicly available.
7. Add basic tests under `tests/test_data.py` (label coverage, filename parsing — anything that doesn't require the actual data).

## Adding a new model / encoder / fusion strategy

- **New audio encoder**: extend `AudioEncoder` in `src/models/audio_encoder.py` to support a new HF model id. Most SSL speech models (wav2vec2, HuBERT, WavLM, mHuBERT, w2v-BERT) work out of the box via `AutoModel`.
- **New text encoder**: same pattern in `src/models/text_encoder.py`.
- **New fusion strategy**: add a new `nn.Module` to `src/models/fusion.py` and register it in `build_fusion()`.
- **New training tricks** (e.g. SAM optimizer, layer-wise LR decay): modify `SERLightningModule.configure_optimizers()` in `src/models/lightning_module.py`.

## Reporting results

When you run a new experiment and want to add it to the benchmark table:

1. Make sure you used a config that's committed to the repo (no off-the-cuff hyperparameters).
2. Run `python -m src.evaluate --checkpoint <path> --config <cfg>` to produce `metrics.json` + `confusion_matrix.png`.
3. Update `results/results.md` with the row, and link to the confusion matrix.
4. Open a PR with the config, the metrics JSON (commit it), and the updated table.

## Things we're not looking for

- **No committed checkpoints, datasets, or `*.wav` files.** Use HuggingFace Hub or release assets for models; use download scripts for data.
- **No notebooks with massive cell outputs.** Run `jupyter nbconvert --clear-output` before committing.
- **No proprietary or licensed audio.** RAVDESS, MELD, and IEMOCAP-with-license are the canonical datasets.

## Questions?

Open an issue with the `question` label, or reach out via [skakarh.com](https://www.skakarh.com).
