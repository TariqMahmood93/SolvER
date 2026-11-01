# SentI_static.py
# ------------------------------------------------------------
# Static (single-snapshot) variant of SENTI's FAISS imputation.
#
# SentI.run_imputation treats a dataset as a growing database: it globs the
# chunked *_nonimputed.csv files and extends one FAISS index across them. This
# module runs that first-chunk case only - one file in, one file out - and adds
# what the SolvIER data-prep pipeline needs:
#
#   * files are selected by missingness mechanism (MCAR / MAR / MNAR)
#   * bookkeeping columns (split, pair_id, side, label) are excluded from the
#     embedded text and are never imputed
#   * every non-bookkeeping column is a target, not only object-dtype ones, so
#     numeric attributes such as `year` are imputed too
#
# The neighbour search, thresholds and re-embedding are unchanged from SentI.py.
# ------------------------------------------------------------

from __future__ import annotations  # PEP 604 unions (X | Y) on Python 3.9

import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from faiss_utils import build_vector_DB, embed_list, load_model, update_vector_DB
from Logs import save_evaluation
from similarity import (DEFAULT_EMBED_MODEL, SUMMARY_FIELDS, format_report, score_logs,
                        to_native as _to_native)

__all__ = ["senti_fill", "impute_file", "run_static_imputation"]


def _row_texts(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """One string per row, built from `cols` only - nulls become empty."""
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()


def senti_fill(
    df: pd.DataFrame,
    target_cols: list[str],
    ground_truth: pd.DataFrame,
    model,
    median_k: int = 100,
    top_k: int = 25,
    sim_th: float = 0.65,
    high_th: float = 1.0,
    desc: str = "SENTI",
) -> tuple[list[dict], float, float]:
    """Fill every null in `target_cols` in place. Returns (logs, index_time, impute_time).

    The search, thresholds and re-embedding are SentI.py's. Split out of
    `impute_file` so other callers (Data_preparation/code/imputers.py) can reuse
    the search without SENTI's file naming.
    """
    texts = _row_texts(df, target_cols)
    t0 = time.time()
    emb_matrix = embed_list(model, texts)
    faiss_db = build_vector_DB(emb_matrix)
    faiss_time = time.time() - t0

    missing_rows = np.where(df[target_cols].isna().any(axis=1))[0]
    logs = []
    t0 = time.time()
    for ridx in tqdm(missing_rows, desc=f"  {desc}", leave=False):
        D, I = faiss_db.search(emb_matrix[ridx : ridx + 1], median_k)
        sims, neigh = ((D[0] + 1) / 2).tolist(), I[0].tolist()
        query = texts[ridx]

        for c in target_cols:
            ci = df.columns.get_loc(c)
            if not pd.isna(df.iat[ridx, ci]):
                continue
            # neighbours that actually carry a value for this column
            valid = [(nj, s) for nj, s in zip(neigh, sims) if pd.notna(df.iat[nj, ci])][:top_k]
            direct = next((df.iat[nj, ci] for nj, s in valid if s >= high_th), None)
            if direct is not None:
                imp = direct
            else:
                cand = [df.iat[nj, ci] for nj, s in valid if s >= sim_th]
                imp = Counter(cand).most_common(1)[0][0] if cand else None

            logs.append({
                "row_index": int(ridx),
                "column": c,
                "query": query,
                "top_neighbors": [texts[nj] for nj, _ in valid],
                "extracted_values": [_to_native(df.iat[nj, ci]) for nj, _ in valid],
                "similarity_scores": [s for _, s in valid],
                "imputed_value": _to_native(imp),
                "ground_truth_value": _to_native(ground_truth.iat[ridx, ground_truth.columns.get_loc(c)]),
            })
            df.iat[ridx, ci] = imp

        # the repaired row re-enters the index, as in SentI.py
        new_txt = " ".join(df.loc[ridx, target_cols].fillna("").astype(str).tolist())
        new_emb = embed_list(model, [new_txt])[0]
        update_vector_DB(faiss_db, new_emb, ridx)
        emb_matrix[ridx], texts[ridx] = new_emb, new_txt
    imputation_time = time.time() - t0
    gc.collect()
    return logs, faiss_time, imputation_time


def impute_file(
    nonimputed_fp: Path,
    ground_truth: pd.DataFrame,
    id_cols: list[str],
    model,
    median_k: int = 100,
    top_k: int = 25,
    sim_th: float = 0.65,
    high_th: float = 1.0,
    save_log: bool = True,
    score_similarity: bool = True,
) -> dict:
    """Impute one *_nonimputed.csv, write *_imputed.csv beside it, return metrics."""
    out_dir = nonimputed_fp.parent
    tag_imp = nonimputed_fp.stem.replace("nonimputed", "imputed")

    df = pd.read_csv(nonimputed_fp)
    target_cols = [c for c in df.columns if c not in id_cols]  # dtype-agnostic, unlike SentI.cat_cols
    n_nulls = int(df[target_cols].isna().to_numpy().sum())

    logs, faiss_time, imputation_time = senti_fill(
        df, target_cols, ground_truth, model, median_k=median_k, top_k=top_k,
        sim_th=sim_th, high_th=high_th, desc=tag_imp)

    # restore each column's original dtype where it still fits (keeps `year` an int)
    for c in target_cols:
        if pd.api.types.is_integer_dtype(ground_truth[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    out_csv = out_dir / f"{tag_imp}.csv"
    df.to_csv(out_csv, index=False)

    # scored on the values, not on the retrieval - see similarity.py's header
    numeric_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(ground_truth[c])]
    metrics = score_logs(logs, target_cols, numeric_cols, n_nulls=n_nulls, model=model,
                         score_similarity=score_similarity)

    if save_log:
        (out_dir / f"{tag_imp}.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    save_evaluation(
        file_name=out_csv.name, start_index=0, end_index=len(df) - 1,
        nulls=n_nulls, exact_matches=metrics["exact_matches"],
        avg_similarity=metrics["avg_semantic_sim"],
        processing_time=faiss_time + imputation_time, faiss_time=faiss_time,
        imputation_time=imputation_time, total_processing_time=faiss_time + imputation_time,
        out_csv=str(out_dir / f"{tag_imp}_evaluation.csv"),
        per_column=metrics["per_column"],
        metrics=metrics,
    )

    gc.collect()
    print(format_report(metrics), flush=True)
    return {
        "file": out_csv.name, "rows": len(df),
        **{k: metrics[k] for k in SUMMARY_FIELDS},
        "faiss_s": round(faiss_time, 1), "impute_s": round(imputation_time, 1),
        **{f"{c}_{k}": v[k] for c, v in metrics["per_column"].items()
           for k in ("nulls", "exact", "wrong", "avg_sim")},
    }


def run_static_imputation(
    data_dir: str | Path,
    dataset: str,
    mechanisms: list[str],
    pcts: list[int],
    seed: int,
    id_cols: list[str],
    ground_truth_file: str | Path | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    **kwargs,
) -> pd.DataFrame:
    """Impute every {dataset}_{MECH}_{pct}_{seed}_nonimputed.csv in `data_dir`.

    `ground_truth_file` defaults to {dataset}.csv and is used for scoring only -
    it is never read during imputation.
    """
    data_dir = Path(data_dir)
    gt_fp = Path(ground_truth_file) if ground_truth_file else data_dir / f"{dataset}.csv"
    ground_truth = pd.read_csv(gt_fp)
    print(f"ground truth : {gt_fp.name} ({len(ground_truth)} rows)")
    print(f"embedded cols: {[c for c in ground_truth.columns if c not in id_cols]}")
    print(f"excluded cols: {id_cols}\n")

    model = load_model(embed_model)
    print(f"model {embed_model} on {model.device}\n")

    rows = []
    for mech in mechanisms:
        for pct in pcts:
            fp = data_dir / f"{dataset}_{mech}_{pct}_{seed}_nonimputed.csv"
            if not fp.is_file():
                print(f"skip (missing): {fp.name}")
                continue
            assert len(pd.read_csv(fp)) == len(ground_truth), f"{fp.name} row count != ground truth"
            print(f"  {fp.name.replace('nonimputed', 'imputed')}")
            rows.append({"mechanism": mech, "pct": pct, **impute_file(
                fp, ground_truth, id_cols, model, **kwargs)})
    return pd.DataFrame(rows)
