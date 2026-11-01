"""Imputation for the SolvIER pipeline: senti, knn, simple.

Three strategies, one driver, one output contract. Every run reads
`{dataset}_{MECH}_{pct}_{seed}_nonimputed.csv` and writes, beside it:

    {dataset}_{MECH}_{pct}_{seed}_imputed_{strategy}.csv
    {dataset}_{MECH}_{pct}_{seed}_imputed_{strategy}.json             (per-cell decisions)
    {dataset}_{MECH}_{pct}_{seed}_imputed_{strategy}_evaluation.csv   (metrics)
    {dataset}_{MECH}_{pct}_{seed}_imputed_{strategy}_tuple_similarity.csv

    senti   SENTI's LaBSE + FAISS search, reused through `SentI_static.senti_fill`.
    knn     TF-IDF char n-gram neighbours; k nearest with a value vote.
    simple  per-column mean (numeric) or mode (categorical).

The strategy is in the file name, so the variants coexist and Ditto can be run
on any of them without re-imputing.

The ground-truth file is used for scoring only and is never read while filling.
Scoring lives in SENTI/Code/similarity.py: `avg_semantic_sim` is the cosine
between the imputed value and the ground-truth value, and every rate names the
denominator it used. See that module's header.
"""
from __future__ import annotations

import gc
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# the scorer lives beside SENTI's embedding helpers
SENTI_CODE = Path(__file__).resolve().parents[2] / "SENTI" / "Code"
if SENTI_CODE.is_dir() and str(SENTI_CODE) not in sys.path:
    sys.path.insert(0, str(SENTI_CODE))
from similarity import (DEFAULT_EMBED_MODEL, SUMMARY_FIELDS, format_report,  # noqa: E402
                        score_logs, to_native, tuple_similarities,
                        value_similarity_rows)

# the 10-column slim schema. ORDER MATTERS - every downstream reader relies on it.
# Names are deliberately human-readable (spaces, parens); plot_eval renames them
# to short identifiers on load, so downstream code never has to type the long one
SIM_IMP = "semantic_similarity (Average semantic similarity over imputed values)"
SIM_ALL = "semantic_similarity (Average semantic similarity over all nulls)"
SIM_NEX = "semantic_similarity (Average over imputed values that are not exact matches)"
EM_ALL  = "exact matches (over all nulls)"
EM_IMP  = "exact matches (over imputed values)"
IMP_PCT = "imputed % (of all nulls)"
IMP_PCT_TOTAL = "imputed % (of total cells M*N)"
EVAL_COLUMNS = [
    "file", "imputer", "mechanism", "null_pct", "seed",
    "total_nulls", "total_imputed", "exact_matches",
    IMP_PCT, IMP_PCT_TOTAL,
    SIM_IMP, SIM_ALL, SIM_NEX,
    EM_ALL, EM_IMP,
]
# `tag` is   {dataset}_{MECH}_{pct}_{seed}_imputed_{strategy}  (see run_imputation)
# so this regex finds pct and seed by their position relative to _imputed_, which
# is anchored and unambiguous even when the dataset part contains underscores
_PARSE = re.compile(r"_(?P<mechanism>MCAR|MAR|MNAR)_(?P<null_pct>\d+)_(?P<seed>\d+)_imputed_")

STRATEGIES = ("senti", "knn", "simple")

__all__ = ["STRATEGIES", "run_imputation", "save_evaluation",
           "simple_fill", "knn_fill"]


# ----------------------------------------------------------------- fill helpers
def _row_texts(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """One string per row from `cols` only - nulls become empty."""
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()


def _column_default(df: pd.DataFrame, col: str, numeric: bool, integer: bool):
    """The column's own mean or mode, from the observed cells of the damaged file."""
    observed = df[col].dropna()
    if not len(observed):
        return None
    if numeric:
        mean = pd.to_numeric(observed, errors="coerce").mean()
        if pd.isna(mean):
            return None
        return int(round(mean)) if integer else float(mean)
    modes = observed.mode()
    return modes.iloc[0] if len(modes) else None


def _log(ridx: int, col: str, imputed, ground_truth, **extra) -> dict:
    """One per-cell log entry, matching the shape senti_fill emits so score_logs
    can read all three strategies through the same path."""
    return {"row_index": int(ridx), "column": col,
            "imputed_value": to_native(imputed),
            "ground_truth_value": to_native(ground_truth), **extra}


# --------------------------------------------------------------------- fills
def simple_fill(df: pd.DataFrame, target_cols: list[str], ground_truth: pd.DataFrame,
                desc: str = "simple") -> tuple[list[dict], float, float]:
    """Per-column mean (numeric) or mode (categorical). Fills `df` in place."""
    t0 = time.time()
    logs = []
    for c in target_cols:
        numeric = pd.api.types.is_numeric_dtype(ground_truth[c])
        value = _column_default(df, c, numeric,
                                pd.api.types.is_integer_dtype(ground_truth[c]))
        ci, gi = df.columns.get_loc(c), ground_truth.columns.get_loc(c)
        for ridx in np.where(df[c].isna().to_numpy())[0]:
            logs.append(_log(int(ridx), c, value, ground_truth.iat[ridx, gi],
                             source="column mean" if numeric else "column mode"))
            df.iat[ridx, ci] = value
    return logs, 0.0, time.time() - t0


def knn_fill(df: pd.DataFrame, target_cols: list[str], ground_truth: pd.DataFrame,
             knn_k: int = 10, ngram_range: tuple[int, int] = (2, 4),
             desc: str = "knn") -> tuple[list[dict], float, float]:
    """Vote among the k nearest rows (TF-IDF char n-grams, cosine). Fills `df` in place."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from tqdm.auto import tqdm

    t0 = time.time()
    texts = _row_texts(df, target_cols)
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range).fit_transform(texts)
    # extra neighbours: some may be null in the target column
    n_neigh = min(len(df), knn_k * 4 + 1)
    nn = NearestNeighbors(n_neighbors=n_neigh, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    index_time = time.time() - t0

    t0 = time.time()
    logs = []
    defaults = {c: _column_default(df, c,
                                   pd.api.types.is_numeric_dtype(ground_truth[c]),
                                   pd.api.types.is_integer_dtype(ground_truth[c]))
                for c in target_cols}
    missing_rows = np.where(df[target_cols].isna().any(axis=1).to_numpy())[0]
    for ridx in tqdm(missing_rows, desc=f"  {desc}", leave=False):
        neigh = [(int(nj), float(1.0 - d))
                 for nj, d in zip(idx[ridx], dist[ridx]) if int(nj) != int(ridx)]
        for c in target_cols:
            ci, gi = df.columns.get_loc(c), ground_truth.columns.get_loc(c)
            if not pd.isna(df.iat[ridx, ci]):
                continue
            valid = [(nj, s) for nj, s in neigh if pd.notna(df.iat[nj, ci])][:knn_k]
            vals = [df.iat[nj, ci] for nj, _ in valid]
            if not vals:
                imp, source = defaults[c], "column fallback"
            elif pd.api.types.is_numeric_dtype(ground_truth[c]):
                mean = float(np.mean(pd.to_numeric(pd.Series(vals), errors="coerce").dropna()))
                imp = (int(round(mean)) if pd.api.types.is_integer_dtype(ground_truth[c])
                       else mean)
                source = f"mean of {len(vals)} neighbours"
            else:
                imp, source = Counter(vals).most_common(1)[0][0], f"vote of {len(vals)} neighbours"
            logs.append(_log(int(ridx), c, imp, ground_truth.iat[ridx, gi],
                             source=source,
                             extracted_values=[to_native(v) for v in vals],
                             similarity_scores=[s for _, s in valid]))
            df.iat[ridx, ci] = imp
    return logs, index_time, time.time() - t0


# --------------------------------------------------------------------- bookkeeping
def _r4(v: float | None) -> float:
    """Round to 4 decimals; None collapses to 0.0 so no NaN reaches the CSV."""
    return round(float(v), 4) if v is not None else 0.0


def save_evaluation(out_csv: Path, *, file_name: str, imputer: str, rows: int, metrics: dict,
                    index_time: float, imputation_time: float,
                    n_target_cols: int) -> None:
    """One-row 15-column metrics CSV. The slim schema the downstream tools read.

    `index_time` and `imputation_time` are no longer written - the shape is held
    deliberately narrow (see EVAL_COLUMNS). The driver still measures them for
    the stdout `format_report`, which prints them per file. `n_target_cols` is
    the attribute count (N columns in the long CSV without the id columns) so
    the dataset-wide fill rate K / (M * N) can be reported alongside K / nulls.
    """
    m = _PARSE.search(file_name)
    assert m, (f"could not parse mechanism/null_pct/seed from {file_name!r}; "
               f"expected ..._{{MECH}}_{{pct}}_{{seed}}_imputed_{{strategy}}")

    N = int(metrics["nulls"])            # every injected null
    K = int(metrics["imputed"])          # the cells the imputer filled
    E = int(metrics["exact_matches"])    # of those, string-exact against the truth
    total_cells = int(rows) * int(n_target_cols)   # M * N - every value in the data block

    # EM_all and EM_imp are rates in percent; S_imp and S_all are cosines in [0, 1]
    row = {
        "file": file_name,
        "imputer": imputer,
        "mechanism": m["mechanism"],
        "null_pct": int(m["null_pct"]),
        "seed": int(m["seed"]),
        "total_nulls": N,
        "total_imputed": K,
        "exact_matches": E,
        IMP_PCT: _r4(100 * K / N) if N else 0.0,              # K / N x 100
        IMP_PCT_TOTAL: _r4(100 * K / total_cells) if total_cells else 0.0,  # K / (M * N) x 100
        # similarities as percentages (cosine x 100) to match the exact-match rates
        SIM_IMP: _r4(100 * metrics["avg_semantic_sim"]) if metrics["avg_semantic_sim"] is not None else 0.0,
        SIM_ALL: _r4(100 * metrics["avg_semantic_sim_all_nulls"]) if metrics["avg_semantic_sim_all_nulls"] is not None else 0.0,
        SIM_NEX: _r4(100 * metrics["avg_semantic_sim_nonexact"]) if metrics["avg_semantic_sim_nonexact"] is not None else 0.0,
        EM_ALL: _r4(100 * E / N) if N else 0.0,               # E / N x 100
        EM_IMP: _r4(100 * E / K) if K else 0.0,               # E / K x 100
    }
    # DataFrame holds column order AND catches a schema drift (new/dropped key)
    pd.DataFrame([row], columns=EVAL_COLUMNS).to_csv(out_csv, index=False)


# ------------------------------------------------------------------------- driver
def run_imputation(data_dir: str | Path, dataset: str, mechanisms: list[str], pcts: list,
                   seed: int, id_cols: list[str], strategy: str = "senti",
                   ground_truth_file: str | Path | None = None, save_log: bool = True,
                   knn_k: int = 10,
                   senti_code: str | Path | None = None,
                   embed_model: str = DEFAULT_EMBED_MODEL, median_k: int = 100,
                   top_k: int = 25, sim_th: float = 0.65, high_th: float = 1.0,
                   score_similarity: bool = True,
                   sim_embed_model: str | None = None) -> pd.DataFrame:
    """Impute every {dataset}_{MECH}_{pct}_{seed}_nonimputed.csv.

    `strategy` picks the fill: "senti" (SENTI's LaBSE + FAISS, loads the model),
    "knn" (TF-IDF char n-gram neighbours, no transformer) or "simple" (per-column
    mean/mode). The evaluation schema is identical across the three.

    `score_similarity` embeds every imputed value and its ground-truth value to
    score the fill; pass False to skip that pass and leave the similarity columns
    empty. `sim_embed_model` defaults to `embed_model`. For knn and simple the
    scorer loads the embedder on demand since there is no fill-time model to reuse.
    """
    assert strategy in STRATEGIES, f"strategy must be one of {STRATEGIES}, got {strategy!r}"
    data_dir = Path(data_dir)
    gt_fp = Path(ground_truth_file) if ground_truth_file else data_dir / f"{dataset}.csv"
    ground_truth = pd.read_csv(gt_fp)

    print(f"imputer      : {strategy}")
    print(f"ground truth : {gt_fp.name} ({len(ground_truth)} rows), scoring only")
    print(f"target cols  : {[c for c in ground_truth.columns if c not in id_cols]}")
    print(f"excluded cols: {id_cols}")
    print(f"similarity   : {'scored' if score_similarity else 'not computed'}\n")

    # pick the fill and only load SENTI when strategy == "senti"
    if strategy == "senti":
        senti_code = Path(senti_code) if senti_code else SENTI_CODE
        assert (senti_code / "SentI_static.py").is_file(), f"SENTI code not found at {senti_code}"
        if str(senti_code) not in sys.path:
            sys.path.insert(0, str(senti_code))
        from faiss_utils import load_model            # noqa: E402
        from SentI_static import senti_fill           # noqa: E402
        model = load_model(embed_model)
        print(f"model {embed_model} on {model.device}\n")
        fill = senti_fill
        fill_kwargs = {"model": model, "median_k": median_k, "top_k": top_k,
                       "sim_th": sim_th, "high_th": high_th}
        # the scorer reuses the model SENTI loaded, unless a different one was asked for
        score_model = model if sim_embed_model in (None, embed_model) else None
    elif strategy == "knn":
        fill, fill_kwargs = knn_fill, {"knn_k": knn_k}
        score_model = None   # similarity.py will lazy-load the embedder by name
    else:  # simple
        fill, fill_kwargs = simple_fill, {}
        score_model = None

    rows = []
    for mech in mechanisms:
        for pct in pcts:
            src = data_dir / f"{dataset}_{mech}_{pct}_{seed}_nonimputed.csv"
            if not src.is_file():
                print(f"skip (missing): {src.name}")
                continue
            tag = f"{dataset}_{mech}_{pct}_{seed}_imputed_{strategy}"
            df = pd.read_csv(src)
            nonimputed = df.copy()                  # keep the pre-fill snapshot for tuple scoring
            assert len(df) == len(ground_truth), f"{src.name} row count != ground truth"
            target_cols = [c for c in df.columns if c not in id_cols]
            n_nulls = int(df[target_cols].isna().to_numpy().sum())

            logs, index_time, impute_time = fill(df, target_cols, ground_truth,
                                                 desc=tag, **fill_kwargs)

            # restore each column's original dtype where it still fits (keeps `year` an int)
            for c in target_cols:
                if pd.api.types.is_integer_dtype(ground_truth[c]):
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            df.to_csv(data_dir / f"{tag}.csv", index=False)

            numeric_cols = [c for c in target_cols
                            if pd.api.types.is_numeric_dtype(ground_truth[c])]
            metrics = score_logs(logs, target_cols, numeric_cols, n_nulls=n_nulls,
                                 model=score_model,
                                 embed_model=sim_embed_model or embed_model,
                                 score_similarity=score_similarity)
            if save_log:
                (data_dir / f"{tag}.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
            save_evaluation(data_dir / f"{tag}_evaluation.csv", file_name=f"{tag}.csv",
                            imputer=strategy, rows=len(df), metrics=metrics,
                            index_time=index_time, imputation_time=impute_time,
                            n_target_cols=len(target_cols))

            # second CSV: per-tuple cosine, (imputed vs truth) and (nonimputed vs truth).
            # Only rows with at least one null in the nonimputed are included.
            if score_similarity:
                tdf = tuple_similarities(ground_truth, df, nonimputed, target_cols,
                                         model=score_model,
                                         embed_model=sim_embed_model or embed_model)
                # prepend the same identifiers the main eval CSV carries, so each
                # tuple row is self-describing when files are merged across runs
                tdf.insert(0, "file", f"{tag}.csv")
                tdf.insert(1, "imputer", strategy)
                tdf.insert(2, "mechanism", mech)
                tdf.insert(3, "null_pct", int(pct))
                tdf.insert(4, "seed", int(seed))
                tdf.to_csv(data_dir / f"{tag}_tuple_similarity.csv", index=False)
                sim_col = "semantic_similarity (imputed vs truth)"
                non_col = "semantic_similarity (nonimputed vs truth)"
                if len(tdf):
                    print(f"    tuple sim      : {len(tdf)} rows with nulls, "
                          f"mean imputed={tdf[sim_col].mean():.2f}%, "
                          f"mean nonimputed={tdf[non_col].mean():.2f}%")

                # third CSV: per-CELL cosine (ground truth vs imputed), one row
                # per null in the file. Same metadata header as the tuple CSV.
                # `ground_truth` is passed so each cell row carries the pair's
                # match label from the long CSV.
                vdf = value_similarity_rows(logs, ground_truth,
                                            model=score_model,
                                            embed_model=sim_embed_model or embed_model)
                vdf.insert(0, "file", f"{tag}.csv")
                vdf.insert(1, "imputer", strategy)
                vdf.insert(2, "mechanism", mech)
                vdf.insert(3, "null_pct", int(pct))
                vdf.insert(4, "seed", int(seed))
                vdf.to_csv(data_dir / f"{tag}_value_similarity.csv", index=False)
                if len(vdf):
                    print(f"    value sim      : {len(vdf)} cells, "
                          f"mean={vdf['semantic_similarity (imputed vs truth)'].mean():.2f}%")

            rows.append({"imputer": strategy, "mechanism": mech, "pct": pct, "file": f"{tag}.csv",
                         "rows": len(df), **{k: metrics[k] for k in SUMMARY_FIELDS},
                         "index_s": round(index_time, 1), "impute_s": round(impute_time, 1)})
            print(f"  {tag}.csv")
            print(format_report(metrics), flush=True)
            gc.collect()
    return pd.DataFrame(rows)
