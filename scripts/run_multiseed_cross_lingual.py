"""Multi-seed cross-lingual SER orchestrator.

Runs the 3-seed cross-lingual study designed in
`research/cross_lingual/multiseed_protocol.md`.

Design (6 trainings + 12 evaluations):

    For each seed s in {0, 1, 2}:
      - Train ONE Punjabi RASA XLS-R model with init seed and val-split seed
        from the seed map.
      - Train ONE URDU-Latif XLS-R model with init seed and held-out show
        from the seed map (split_mode=show_independent).
      - Evaluate the Punjabi model on:
          (cond 1) Punjabi test     -> within-language baseline
          (cond 3) Urdu  test SM_s  -> Pun -> Urdu zero-shot transfer
      - Evaluate the Urdu    model on:
          (cond 2) Urdu  test SM_s  -> within-language ceiling (show-indep)
          (cond 4) Punjabi test     -> Urdu -> Pun zero-shot transfer

Seeds map:
    s=0: init=42,   urdu_held_out=SM2, rasa_val_seed=42
    s=1: init=123,  urdu_held_out=SM3, rasa_val_seed=123
    s=2: init=2024, urdu_held_out=SM4, rasa_val_seed=2024

Outputs:
    outputs/multiseed_cross_lingual/
        configs/   per-(model, seed) YAML files used for training
        pun_s0/    Lightning output dir (ckpts, TB logs) for Pun model, seed 0
        pun_s1/
        pun_s2/
        urdu_s0/
        urdu_s1/
        urdu_s2/
    results/multiseed_cross_lingual/
        pun_s0_on_pun.json      cond 1, seed 0
        pun_s0_on_urdu.json     cond 3, seed 0
        urdu_s0_on_urdu.json    cond 2, seed 0
        urdu_s0_on_pun.json     cond 4, seed 0
        ... (3 seeds x 4 = 12 JSON files)

Usage:
    # Dry run -- print the plan without executing
    python -m scripts.run_multiseed_cross_lingual --dry-run

    # Run everything (long! ~12-18h on M-series MPS)
    python -m scripts.run_multiseed_cross_lingual

    # Resume: skip cells whose output JSON already exists
    python -m scripts.run_multiseed_cross_lingual --resume

    # Run a subset
    python -m scripts.run_multiseed_cross_lingual --seeds 0 1 --skip-eval

Author: Muhammad Shahnawaz Khan
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_PUN_CONFIG = REPO_ROOT / "configs/audio_only_xlsr_punjabi_rasa.yaml"
BASE_URDU_CONFIG = REPO_ROOT / "configs/audio_only_xlsr_urdu_latif.yaml"
RUN_ROOT = REPO_ROOT / "outputs/multiseed_cross_lingual"
RESULTS_ROOT = REPO_ROOT / "results/multiseed_cross_lingual"
CONFIGS_DIR = RUN_ROOT / "configs"

# Seed map. See module docstring.
SEED_MAP = [
    {"s": 0, "init_seed": 42,   "urdu_held_out": "SM2", "rasa_val_seed": 42},
    {"s": 1, "init_seed": 123,  "urdu_held_out": "SM3", "rasa_val_seed": 123},
    {"s": 2, "init_seed": 2024, "urdu_held_out": "SM4", "rasa_val_seed": 2024},
]


def load_yaml_with_defaults(path: Path) -> dict:
    """Inline the `defaults: [base]` chain so the written config is self-contained."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for default in cfg.pop("defaults", []):
        base_path = path.parent / f"{default}.yaml"
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def build_pun_config(seed_entry: dict) -> tuple[Path, dict]:
    cfg = load_yaml_with_defaults(BASE_PUN_CONFIG)
    cfg["seed"] = seed_entry["init_seed"]
    cfg.setdefault("data", {})["split_seed"] = seed_entry["rasa_val_seed"]
    cfg["experiment_name"] = f"multiseed_pun_s{seed_entry['s']}"
    cfg.setdefault("logging", {})["output_dir"] = str(RUN_ROOT)
    cfg["logging"]["use_wandb"] = False  # multi-run, suppress to avoid spam
    out_path = CONFIGS_DIR / f"pun_s{seed_entry['s']}.yaml"
    return out_path, cfg


def build_urdu_config(seed_entry: dict) -> tuple[Path, dict]:
    cfg = load_yaml_with_defaults(BASE_URDU_CONFIG)
    cfg["seed"] = seed_entry["init_seed"]
    cfg.setdefault("data", {})
    cfg["data"]["split_mode"] = "show_independent"
    cfg["data"]["held_out_show"] = seed_entry["urdu_held_out"]
    cfg["data"]["split_seed"] = seed_entry["init_seed"]
    cfg["experiment_name"] = f"multiseed_urdu_s{seed_entry['s']}"
    cfg.setdefault("logging", {})["output_dir"] = str(RUN_ROOT)
    cfg["logging"]["use_wandb"] = False
    out_path = CONFIGS_DIR / f"urdu_s{seed_entry['s']}.yaml"
    return out_path, cfg


def find_best_ckpt(exp_dir: Path) -> Path:
    """Return the best-{epoch}-{wf1}.ckpt with the highest WF1 in the filename."""
    ckpts = sorted(exp_dir.glob("best-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No best-*.ckpt found in {exp_dir}")
    # Filenames look like: best-07-0.9897.ckpt -- the val/wf1 score is the last token.
    def score(p: Path) -> float:
        try:
            return float(p.stem.rsplit("-", 1)[-1])
        except ValueError:
            return -1.0
    return max(ckpts, key=score)


def run_subprocess(cmd: list[str], dry_run: bool) -> None:
    print(f"\n[$] {' '.join(cmd)}")
    if dry_run:
        return
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    dt = time.time() - t0
    if result.returncode != 0:
        raise SystemExit(
            f"[FAIL] command exited with {result.returncode} after {dt:.0f}s: {' '.join(cmd)}"
        )
    print(f"[ok ] took {dt:.0f}s")


def train_one(cfg_path: Path, dry_run: bool, resume: bool) -> None:
    exp_dir = RUN_ROOT / cfg_path.stem.replace("_", "_")  # noop, kept for clarity
    # The experiment_name in the config is what determines the output subfolder.
    cfg = yaml.safe_load(cfg_path.read_text())
    exp_dir = RUN_ROOT / cfg["experiment_name"]
    if resume and exp_dir.exists() and list(exp_dir.glob("best-*.ckpt")):
        print(f"[skip] training already done: {exp_dir}")
        return
    run_subprocess(
        [sys.executable, "-m", "src.train", "--config", str(cfg_path), "--skip-test"],
        dry_run=dry_run,
    )


def eval_one(
    ckpt: Path,
    target_dataset: str,
    output: Path,
    seed_tag: str,
    *,
    urdu_held_out_show: str | None,
    dry_run: bool,
    resume: bool,
) -> None:
    if resume and output.exists():
        print(f"[skip] eval already done: {output}")
        return
    cmd = [
        sys.executable, "-m", "scripts.eval_checkpoint",
        "--ckpt", str(ckpt),
        "--target-dataset", target_dataset,
        "--output", str(output),
        "--seed-tag", seed_tag,
    ]
    if target_dataset == "urdu_latif" and urdu_held_out_show:
        cmd += ["--urdu-held-out-show", urdu_held_out_show]
    run_subprocess(cmd, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print plan, don't execute.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip trainings/evals whose outputs already exist.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Subset of seed indices to run (default: all 3).",
    )
    parser.add_argument(
        "--skip-train", action="store_true", help="Only run evaluations (assumes ckpts exist)."
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Only run trainings (no evaluations)."
    )
    args = parser.parse_args()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    seed_entries = [e for e in SEED_MAP if e["s"] in args.seeds]

    # ---- 1. Write all per-seed configs (always; cheap; needed for resume too) ----
    pun_cfgs: dict[int, Path] = {}
    urdu_cfgs: dict[int, Path] = {}
    for entry in seed_entries:
        pun_path, pun_cfg = build_pun_config(entry)
        urdu_path, urdu_cfg = build_urdu_config(entry)
        pun_path.write_text(yaml.safe_dump(pun_cfg, sort_keys=False))
        urdu_path.write_text(yaml.safe_dump(urdu_cfg, sort_keys=False))
        pun_cfgs[entry["s"]] = pun_path
        urdu_cfgs[entry["s"]] = urdu_path
        print(f"[cfg] wrote {pun_path.relative_to(REPO_ROOT)}")
        print(f"[cfg] wrote {urdu_path.relative_to(REPO_ROOT)}")

    # ---- 2. Train 6 models (3 Pun + 3 Urdu) ----
    if not args.skip_train:
        for entry in seed_entries:
            print(f"\n=== TRAIN Pun s{entry['s']} (init={entry['init_seed']}, val_seed={entry['rasa_val_seed']}) ===")
            train_one(pun_cfgs[entry["s"]], dry_run=args.dry_run, resume=args.resume)
            print(f"\n=== TRAIN Urdu s{entry['s']} (init={entry['init_seed']}, held_out={entry['urdu_held_out']}) ===")
            train_one(urdu_cfgs[entry["s"]], dry_run=args.dry_run, resume=args.resume)

    # ---- 3. Evaluate all 12 cells ----
    if not args.skip_eval:
        for entry in seed_entries:
            s = entry["s"]
            pun_dir = RUN_ROOT / f"multiseed_pun_s{s}"
            urdu_dir = RUN_ROOT / f"multiseed_urdu_s{s}"

            if not args.dry_run:
                pun_ckpt = find_best_ckpt(pun_dir)
                urdu_ckpt = find_best_ckpt(urdu_dir)
            else:
                pun_ckpt = pun_dir / "best-XX-YYYY.ckpt"
                urdu_ckpt = urdu_dir / "best-XX-YYYY.ckpt"

            print(f"\n=== EVAL seed {s} ===")
            # cond 1: Pun -> Pun (within)
            eval_one(
                pun_ckpt, "punjabi_rasa",
                RESULTS_ROOT / f"pun_s{s}_on_pun.json",
                seed_tag=f"s{s}",
                urdu_held_out_show=None,
                dry_run=args.dry_run, resume=args.resume,
            )
            # cond 3: Pun -> Urdu (cross-lingual, held_out=SM_s)
            eval_one(
                pun_ckpt, "urdu_latif",
                RESULTS_ROOT / f"pun_s{s}_on_urdu.json",
                seed_tag=f"s{s}",
                urdu_held_out_show=entry["urdu_held_out"],
                dry_run=args.dry_run, resume=args.resume,
            )
            # cond 2: Urdu -> Urdu (within, show-indep)
            eval_one(
                urdu_ckpt, "urdu_latif",
                RESULTS_ROOT / f"urdu_s{s}_on_urdu.json",
                seed_tag=f"s{s}",
                urdu_held_out_show=entry["urdu_held_out"],
                dry_run=args.dry_run, resume=args.resume,
            )
            # cond 4: Urdu -> Pun (cross-lingual)
            eval_one(
                urdu_ckpt, "punjabi_rasa",
                RESULTS_ROOT / f"urdu_s{s}_on_pun.json",
                seed_tag=f"s{s}",
                urdu_held_out_show=None,
                dry_run=args.dry_run, resume=args.resume,
            )

    print("\n[done] multi-seed orchestration finished.")
    print(f"[done] aggregate with: python -m scripts.aggregate_multiseed")


if __name__ == "__main__":
    main()
