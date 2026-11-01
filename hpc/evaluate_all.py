#!/usr/bin/env python3
"""Score every test variant with ONE model load.

SolvER.ipynb (and run_solver.py) launch `matcher.py` as a separate subprocess per
experiment. Each launch re-imports torch and transformers and re-reads the
DistilBERT checkpoint off the filesystem - on a cluster that startup cost dwarfs
the actual inference, which is a few hundred pairs. This script pays it once.

The metrics replicate matcher.tune_threshold exactly: `load_f1` from
ditto_light.ditto.evaluate with threshold tuning, then `real_f1` and the
confusion matrix from a full prediction pass at the tuned threshold.

    python evaluate_all.py --repo ~/SolvIER-main
    python evaluate_all.py --repo ~/SolvIER-main --mechanisms MCAR --pcts 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=here.parent)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--dataset", default="DBLPACM")
    ap.add_argument("--seeds", nargs="+", type=int, default=[584, 1576])
    ap.add_argument("--mechanisms", nargs="+", default=["MCAR", "MAR", "MNAR"])
    ap.add_argument("--pcts", nargs="+", type=int, default=[5, 10, 20, 30, 40])
    ap.add_argument("--variants", nargs="+",
                    default=["nonimputed", "imputed_senti"],
                    help="file-name variants; the imputation strategy is part of the name")
    ap.add_argument("--no-clean-baseline", action="store_true")
    ap.add_argument("--lm", default="distilbert")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--task", default=None, help="checkpoint folder name (default: --dataset)")
    ap.add_argument("--threshold-mode", dest="threshold_mode", default="valid",
                    choices=["valid", "test", "fixed"],
                    help="threshold source: valid (default, leak-free) | test (matcher.py's "
                         "original behaviour, inflates scores) | fixed")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is visible")
    ap.add_argument("--out", type=Path, default=None,
                    help="results CSV (default: <SolvER>/output/Results_<dataset>_fast.csv)")
    return ap.parse_args()


def test_files(a, data_dir: Path):
    """(name, mechanism, imputer, pct, seed, isImputed, path) for every experiment."""
    out = []
    if not a.no_clean_baseline:
        out.append((a.dataset, "clean", "clean", 0, -1, -1, data_dir / f"test_{a.dataset}.txt"))
    for seed in a.seeds:
        for mech in a.mechanisms:
            for pct in a.pcts:
                for variant in a.variants:
                    stem = f"{a.dataset}_{mech}_{pct}_{seed}_{variant}"
                    imputer = ("none" if variant == "nonimputed"
                               else variant.replace("imputed_", ""))
                    out.append((stem, mech, imputer, pct, seed, int(variant.startswith("imputed")),
                                data_dir / f"test_{stem}.txt"))
    return out


def main() -> int:
    a = parse_args()
    repo = a.repo.resolve()
    solver_dir = repo / "SolvIER" / "SolvER"
    data_dir = (a.data_dir or repo / "SolvIER" / "Data").resolve()
    task = a.task or a.dataset

    if not (solver_dir / "matcher.py").exists():
        raise SystemExit(f"matcher.py not found in {solver_dir}  (check --repo)")
    if not data_dir.is_dir():
        raise SystemExit(f"data dir does not exist: {data_dir}  (pass --data-dir)")

    # matcher.py and ditto_light resolve imports, log.txt and tmp.jsonl relative to cwd
    os.chdir(solver_dir)
    sys.path.insert(0, str(solver_dir))

    import pandas as pd
    import sklearn.metrics
    import torch
    from torch.utils import data as torch_data

    import jsonlines
    import matcher
    from ditto_light.dataset import DittoDataset
    from ditto_light.ditto import DittoModel, evaluate as ditto_evaluate

    use_gpu = torch.cuda.is_available() and not a.cpu
    device = "cuda" if use_gpu else "cpu"
    ckpt = solver_dir / "checkpoints" / task / "model.pt"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}  (train first, e.g. via run_solver.py)")

    print(f"repo    : {repo}")
    print(f"data    : {data_dir}")
    print(f"threshold: from {a.threshold_mode}"
          + ("  (tuned on each TEST file - inflates scores)" if a.threshold_mode == "test" else ""))
    print(f"device  : {device}" + ("" if use_gpu else "  (CPU)"))
    print(f"loading {ckpt} ... ", end="", flush=True)
    t0 = time.time()
    model = DittoModel(device=device, lm=a.lm)
    model.load_state_dict(torch.load(ckpt, map_location=lambda s, _: s)["model"])
    model = model.to(device).eval()
    load_s = round(time.time() - t0, 1)
    print(f"{load_s:.1f}s  (loaded ONCE for every experiment)\n", flush=True)

    experiments = test_files(a, data_dir)
    # test_files yields 7-tuples: (name, mechanism, imputer, pct, seed, isImputed, path)
    for name, *_, path in experiments:
        if not path.exists():
            print(f"skip (missing): {path.name}")
    experiments = [e for e in experiments if e[6].exists()]
    if not experiments:
        raise SystemExit(f"no test files found in {data_dir}")

    out_dir = solver_dir / "output"
    out_dir.mkdir(exist_ok=True)
    valid_fp = data_dir / f"valid_{a.dataset}.txt"
    if a.threshold_mode == "valid" and not valid_fp.is_file():
        raise SystemExit(f"--threshold-mode valid needs {valid_fp}")
    tuned_threshold = [None]

    rows = []
    for i, (name, mech, imputer, pct, seed, is_imp, path) in enumerate(experiments, 1):
        t0 = time.time()
        config = {"name": task, "task_type": "classification", "vocab": ["0", "1"],
                  "trainset": str(path), "validset": str(path), "testset": str(path)}

        # --- decide the threshold ---
        matcher.set_seed(123)
        ds = DittoDataset(str(path), max_len=a.max_len, lm=a.lm)
        it = torch_data.DataLoader(dataset=ds, batch_size=64, shuffle=False,
                                   num_workers=0, collate_fn=DittoDataset.pad)
        if a.threshold_mode == "test":
            load_f1, threshold = ditto_evaluate(model, it, threshold=None)  # tuned ON this file
        else:
            if a.threshold_mode == "valid":
                if tuned_threshold[0] is None:                 # tune once, on the valid file
                    vds = DittoDataset(str(valid_fp), max_len=a.max_len, lm=a.lm)
                    vit = torch_data.DataLoader(dataset=vds, batch_size=64, shuffle=False,
                                                num_workers=0, collate_fn=DittoDataset.pad)
                    _, tuned_threshold[0] = ditto_evaluate(model, vit, threshold=None)
                    print(f"threshold tuned on {valid_fp.name}: {tuned_threshold[0]:.2f}\n")
                threshold = tuned_threshold[0]
            else:
                threshold = a.threshold
            load_f1 = ditto_evaluate(model, it, threshold=threshold)

        # --- full prediction pass at that threshold -> real_f1 + confusion matrix ---
        matcher.set_seed(123)
        pred_path = out_dir / f"output_{name}.jsonl"
        matcher.predict(str(path), str(pred_path), config, model,
                        summarizer=None, max_len=a.max_len, lm=a.lm,
                        dk_injector=None, threshold=threshold)

        with jsonlines.open(pred_path) as reader:
            predicts = [int(line["match"]) for line in reader]
        labels = [int(l.split("\t")[-1]) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

        match_s = round(time.time() - t0, 1)      # this file, with the model already loaded
        real_f1 = sklearn.metrics.f1_score(labels, predicts)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(labels, predicts).ravel()
        rows.append({"run": name, "mechanism": mech, "imputer": imputer,
                     "null_pct": pct, "seed": seed,
                     "isImputed": is_imp, "load_f1": load_f1, "real_f1": real_f1,
                     "tn": float(tn), "tp": float(tp), "fn": float(fn), "fp": float(fp),
                     "fnr": fn / (fn + tp) if (fn + tp) else -1.0,
                     "rec": tp / (fn + tp) if (fn + tp) else -1.0,
                     "pairs": len(labels), "match_s": match_s,
                     "ms_per_pair": round(1000 * match_s / len(labels), 2) if labels else -1.0,
                     "model_load_s": load_s})
        print(f"[{i}/{len(experiments)}] {name:<42} real_f1 = {real_f1:.4f}  "
              f"(tp={tp} fp={fp} tn={tn} fn={fn})  {match_s:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    out_csv = a.out or out_dir / f"Results_{a.dataset}_fast.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")
    print(f"matching time: {df.match_s.sum():.0f}s over {len(df)} files, "
          f"{df.match_s.mean():.1f}s per file, {df.ms_per_pair.mean():.2f} ms per pair "
          f"(model loaded once in {load_s:.1f}s)")

    real = df[df.seed >= 0]
    if not real.empty:
        print("\nmean real_f1 by mechanism / null % (columns: isImputed)")
        print(real.groupby(["mechanism", "null_pct", "isImputed"])["real_f1"].mean().unstack("isImputed").round(4))
    if df.real_f1.nunique() == 1:
        print("\nNOTE: every run has the same F1. Check tp/fp/tn/fn above - if fn=0 and tn=0 the\n"
              "model predicts 'match' for every pair, so the test file's content is irrelevant.\n"
              "That means the model collapsed, usually from too small a training set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
