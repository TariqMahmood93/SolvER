#!/usr/bin/env python3
"""Headless SolvER: train Ditto once on the clean split, score every variant.

Same logic as SolvER.ipynb, without Jupyter, so it can be submitted as a batch
job. Every path is derived from --repo (default: this script's parent), so the
checkout can live anywhere on the cluster.

Train and valid are ALWAYS the clean files; only the test file varies across
experiments, so one model is trained and reused for every mechanism, null
percentage, variant and seed.

    python run_solver.py --repo /scratch/$USER/SolvIER-main --retrain
    python run_solver.py --dry-run          # list the experiments, run nothing
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

METRICS = ["load_f1", "real_f1", "TN", "TP", "FN", "FP", "FNR", "REC"]
TEXT_COLUMNS = ["run", "mechanism", "imputer"]
NUM_COLUMNS = ["null_pct", "seed", "isImputed",
               "load_f1", "real_f1", "tn", "tp", "fn", "fp", "fnr", "rec",
               "pairs", "match_s", "ms_per_pair", "train_s"]
TIMING = {"train_s": 0.0}          # training happens once; every row reports the same value
COLUMNS = TEXT_COLUMNS + NUM_COLUMNS
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=here.parent, help="SolvIER-main checkout")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="folder holding the .txt files (default: <repo>/SolvIER/Data)")
    ap.add_argument("--dataset", default="DBLPACM")
    # the data-prep notebook writes these seeds and levels; a value not in the
    # file names matches no file, and every experiment is then skipped
    ap.add_argument("--seeds", nargs="+", type=int, default=[584, 1576])
    ap.add_argument("--mechanisms", nargs="+", default=["MCAR", "MAR", "MNAR"])
    ap.add_argument("--pcts", nargs="+", type=int, default=[5, 10, 20, 30, 40])
    ap.add_argument("--variants", nargs="+",
                    default=["nonimputed", "imputed_senti"],
                    help="file-name variants; the imputation strategy is part of the name")
    ap.add_argument("--no-clean-baseline", action="store_true")
    ap.add_argument("--n-epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="must be <= len(train set), else num_steps is 0")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--train-seed", type=int, default=0, help="--run_id for train_ditto.py")
    ap.add_argument("--lm", default="distilbert")
    # Ditto's optional helpers (paper Sec 3). All off = the paper's "Baseline".
    ap.add_argument("--da", default=None,
                    help="data augmentation op (drop_col, span_del, ...). Also enables MixDA "
                         "and entry_swap. Off by default.")
    ap.add_argument("--dk", default=None, help="domain knowledge: general | product")
    ap.add_argument("--summarize", action="store_true", help="TF-IDF summarization")
    ap.add_argument("--threshold-mode", dest="threshold_mode", default="valid",
                    choices=["valid", "test", "fixed"],
                    help="threshold source: valid (default, leak-free) | test (matcher.py's "
                         "original behaviour, inflates scores) | fixed")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--retrain", action="store_true", help="retrain even if a checkpoint exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug", action="store_true", help="echo subprocess stdout/stderr")
    return ap.parse_args()


def experiment(a, data_dir, mech=None, pct=None, seed=None, variant=None) -> dict:
    """One run: its name and its train/valid/test files."""
    suffix = a.dataset if pct is None else f"{a.dataset}_{mech}_{pct}_{seed}_{variant}"
    files = {}
    for split in SPLITS:
        # train/valid always come from the clean files - only the test file varies
        stem = suffix if split == "test" else a.dataset
        base = data_dir / f"{split}_{stem}"
        files[split] = next((base.with_suffix(e) for e in (".txt", ".text") if base.with_suffix(e).exists()),
                            base.with_suffix(".txt"))
    return {"name": suffix,
            "model": a.dataset,                       # one checkpoint, shared by every experiment
            "mechanism": "clean" if mech is None else mech,
            "null_pct": 0 if pct is None else int(pct),
            "seed": -1 if seed is None else int(seed),
            "imputer": "clean" if variant is None else ("none" if variant == "nonimputed"
                                                        else variant.replace("imputed_", "")),
            "isImputed": -1 if variant is None else int(variant.startswith("imputed")),
            "files": files,
            "missing": [f for f in files.values() if not f.exists()]}


def write_config(exp, config_path: Path) -> None:
    """Rewrite configs.json so train_ditto.py / matcher.py see this run's files.

    Written fresh on every call, so the absolute paths it holds are always this
    machine's - a stale configs.json from another host cannot leak in.
    """
    config_path.write_text(json.dumps([{
        "name": exp["model"],
        "task_type": "classification",
        "vocab": ["0", "1"],
        "trainset": str(exp["files"]["train"]),
        "validset": str(exp["files"]["valid"]),
        "testset": str(exp["files"]["test"]),
    }], indent=4), encoding="utf-8")


def run(cmd, cwd: Path, debug: bool) -> str:
    if debug:
        print("$", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if debug or proc.returncode != 0:
        print(proc.stdout, flush=True)
        print(proc.stderr, file=sys.stderr, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {cmd[1]}")
    return proc.stdout


def train(a, exp, solver_dir: Path, ckpt_dir: Path, config_path: Path) -> None:
    ckpt = ckpt_dir / exp["model"] / "model.pt"
    if ckpt.exists() and not a.retrain:
        print(f"  reusing checkpoint {ckpt}", flush=True)
        return
    n_train = sum(1 for line in exp["files"]["train"].open(encoding="utf-8") if line.strip())
    steps = (n_train // a.batch_size) * a.n_epochs
    if steps == 0:
        raise SystemExit(f"train set has {n_train} rows and batch_size is {a.batch_size}, so "
                         f"num_steps = (n // batch) * epochs = 0 and the LR scheduler gets no "
                         f"steps. Pass --batch-size {max(1, n_train // 2)} or lower.")
    print(f"  training on {n_train} rows, batch {a.batch_size}, {a.n_epochs} epochs "
          f"({steps} steps)", flush=True)
    write_config(exp, config_path)
    cmd = [sys.executable, "train_ditto.py", "--task", exp["model"], "--run_id", a.train_seed,
           "--batch_size", a.batch_size, "--max_len", a.max_len, "--lr", a.lr,
           "--n_epochs", a.n_epochs, "--finetuning", "--lm", a.lm, "--save_model",
           "--logdir", str(ckpt_dir) + "/"]
    if a.da:                                  # augmentation + MixDA + entry_swap, training only
        cmd += ["--da", a.da]
    if a.dk:
        cmd += ["--dk", a.dk]
    if a.summarize:
        cmd.append("--summarize")
    if a.train_size:
        cmd += ["--size", a.train_size]
    t0 = time.perf_counter()
    out = run(cmd, solver_dir, a.debug)
    TIMING["train_s"] = round(time.perf_counter() - t0, 1)
    epochs = [l for l in out.splitlines() if l.startswith("epoch")]
    print("  " + (epochs[-1] if epochs else "no epoch output"), flush=True)
    print(f"  trained in {TIMING['train_s']:.1f}s", flush=True)
    if not ckpt.exists():
        raise RuntimeError(f"training finished but {ckpt} was not written")


def evaluate(a, exp, solver_dir: Path, ckpt_dir: Path, out_dir: Path, config_path: Path, use_gpu: bool):
    write_config(exp, config_path)
    cmd = [sys.executable, "matcher.py", "--task", exp["model"],
           "--input_path", exp["files"]["test"],
           "--output_path", out_dir / f"output_{exp['name']}.jsonl",
           "--lm", a.lm, "--max_len", a.max_len,
           "--threshold_mode", a.threshold_mode, "--threshold", a.threshold,
           "--checkpoint_path", str(ckpt_dir) + "/"]
    if a.dk:
        cmd += ["--dk", a.dk]
    if a.summarize:
        cmd.append("--summarize")
    if use_gpu:
        cmd.append("--use_gpu")
    t0 = time.perf_counter()
    out = run(cmd, solver_dir, a.debug)
    match_s = round(time.perf_counter() - t0, 1)   # this file: model load + threshold + predict
    vals = [float(m.group(1)) if (m := re.search(rf"{k}\s*=\s*([0-9.eE+-]+)", out)) else -1 for k in METRICS]
    pairs = sum(1 for line in exp["files"]["test"].open(encoding="utf-8") if line.strip())
    print(f"  real_f1 = {vals[1]:.4f}   match_s = {match_s:.1f}  ({pairs} pairs)", flush=True)
    return (exp["name"], exp["mechanism"], exp["imputer"], exp["null_pct"], exp["seed"],
            exp["isImputed"], *vals, pairs, match_s,
            round(1000 * match_s / pairs, 2) if pairs else -1, TIMING["train_s"])


def main() -> int:
    a = parse_args()

    repo = a.repo.resolve()
    solver_dir = repo / "SolvIER" / "SolvER"
    data_dir = (a.data_dir or repo / "SolvIER" / "Data").resolve()
    for label, path, hint in (("repo", repo, ""),
                              ("SolvER folder", solver_dir, ""),
                              ("data dir", data_dir, "  (pass --data-dir)")):
        if not path.is_dir():
            raise SystemExit(f"{label} does not exist: {path}{hint}")
    if not (solver_dir / "train_ditto.py").exists():
        raise SystemExit(f"train_ditto.py not found in {solver_dir}")

    ckpt_dir = solver_dir / "checkpoints"
    out_dir = solver_dir / "output"
    out_dir.mkdir(exist_ok=True)
    config_path = solver_dir / "configs.json"

    # torch is needed only to pick the device, and pandas only to write the results
    # table. --dry-run resolves file names and nothing else, so it stays runnable
    # before the environment is built - which is exactly when it is most useful.
    use_gpu = False
    if not a.dry_run:
        import torch
        use_gpu = torch.cuda.is_available()

    print(f"repo    : {repo}")
    print(f"data    : {data_dir}")
    print(f"dataset : {a.dataset}")
    helpers = [n for n, on in (("DA=" + str(a.da), a.da), ("DK=" + str(a.dk), a.dk),
                               ("SU", a.summarize)) if on]
    print(f"helpers : {', '.join(helpers) if helpers else 'none (paper Baseline)'}"
          f" | threshold from: {a.threshold_mode}")
    print("gpu     : " + ("not checked (dry run)" if a.dry_run else
                          f"{use_gpu}" + ("" if use_gpu else "  (CPU only - training will be slow)")))
    print(f"HF cache: {os.environ.get('HF_HOME', '(default ~/.cache/huggingface)')}"
          + ("  [offline]" if os.environ.get("HF_HUB_OFFLINE") else ""))

    experiments = [] if a.no_clean_baseline else [experiment(a, data_dir)]
    for seed in a.seeds:
        for mech in a.mechanisms:
            for pct in a.pcts:
                for variant in a.variants:
                    experiments.append(experiment(a, data_dir, mech, pct, seed, variant))

    runnable = [e for e in experiments if not e["missing"]]
    skipped = [e for e in experiments if e["missing"]]
    if skipped:
        print(f"\nWARNING: {len(skipped)} experiment(s) skipped, missing files:")
        for e in skipped:
            print("   " + e["name"] + ": " + ", ".join(f.name for f in e["missing"]))
    if not runnable:
        raise SystemExit(f"\nNo complete train/valid/test triple found in {data_dir} "
                         f"for dataset={a.dataset}, seeds={a.seeds}, mechanisms={a.mechanisms}")
    print(f"\n{len(runnable)} experiment(s) to run")

    if a.dry_run:
        for e in runnable:
            print(f"  {e['name']:<42} train={e['files']['train'].name:<22} test={e['files']['test'].name}")
        print("\n(dry run - nothing executed)")
        return 0

    results, trained = [], set()
    for i, exp in enumerate(runnable, 1):
        print(f"\n[{i}/{len(runnable)}] {exp['name']}  (model: {exp['model']})", flush=True)
        if exp["model"] not in trained:
            train(a, exp, solver_dir, ckpt_dir, config_path)
            trained.add(exp["model"])
        results.append(evaluate(a, exp, solver_dir, ckpt_dir, out_dir, config_path, use_gpu))

    import pandas as pd

    df = pd.DataFrame(results, columns=COLUMNS)
    results_csv = out_dir / f"Results_{a.dataset}.csv"
    if results_csv.exists():
        df = pd.concat([pd.read_csv(results_csv).reindex(columns=COLUMNS), df], ignore_index=True)
    for col in NUM_COLUMNS:            # text columns stay text - to_numeric would blank them
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = (df.dropna(subset=["run", "seed", "null_pct"])
            .astype({"null_pct": int, "seed": int, "isImputed": int})
            .drop_duplicates(subset=["run"], keep="last")
            .sort_values(["mechanism", "imputer", "seed", "null_pct", "isImputed"])
            .reset_index(drop=True))
    df.to_csv(results_csv, index=False)
    print(f"\nsaved {results_csv}")

    real = df[df.seed >= 0]
    if not real.empty:
        print("\nmean real_f1 by mechanism / null % (columns: isImputed)")
        print(real.groupby(["mechanism", "null_pct", "isImputed"])["real_f1"].mean().unstack("isImputed").round(4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
