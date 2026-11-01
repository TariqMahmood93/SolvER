# similarity.py
# ------------------------------------------------------------
# Imputation quality, scored on the VALUES rather than on the search.
#
# `similarity_scores` in the per-cell logs are cosines between the query ROW and
# a neighbour ROW: they say how confident the retrieval was, and stay just above
# sim_th however wrong the value it carried turns out to be. On AmazonGoogle that
# read 0.99 while three quarters of the fills were wrong.
#
# `value_similarities` embeds the imputed value and the ground-truth value
# themselves, so an exact hit is 1.0 and an unrelated fill is near 0.
#
# Scale: raw cosine in [-1, 1], NOT SentI's (ip + 1) / 2 rescaling. That
# rescaling exists so sim_th / high_th sit on [0, 1]; nothing thresholds this
# metric, and the raw cosine is the readable one (1.0 identical, ~0 unrelated).
#
# `score_logs` is the single place both imputation drivers compute their numbers,
# so Data_preparation/code/imputers.py and SentI_static.py cannot drift apart.
# ------------------------------------------------------------

from __future__ import annotations  # PEP 604 unions (X | Y) on Python 3.9

import numpy as np
import pandas as pd

DEFAULT_EMBED_MODEL = "sentence-transformers/LaBSE"

__all__ = ["DEFAULT_EMBED_MODEL", "to_native", "norm_value", "is_empty", "get_model",
           "value_similarities", "score_logs", "format_report", "SUMMARY_FIELDS",
           "tuple_similarities", "TUPLE_COLUMNS",
           "value_similarity_rows", "VALUE_COLUMNS"]

# the second evaluation CSV written beside the main one: one row per TUPLE that
# had at least one null in the nonimputed, three concatenated strings and the
# cosine of (imputed vs truth) and (nonimputed vs truth). Column names are the
# literal headers written to disk - keep them in sync with tuple_similarities
TUPLE_COLUMNS = [
    "row_index",
    "label",
    "ground_truth_tuple",
    "imputed_tuple",
    "nonimputed_tuple",
    "semantic_similarity (imputed vs truth)",
    "semantic_similarity (nonimputed vs truth)",
    "improvement (imputed - nonimputed)",
]

# the per-cell counterpart: one row per null cell, carrying the ground-truth
# value, the imputed value and their cosine. Written beside the main eval CSV
# as `{tag}_value_similarity.csv` - keep in sync with value_similarity_rows
VALUE_COLUMNS = [
    "row_index",
    "label",
    "column",
    "ground_truth_value",
    "imputed_value",
    "semantic_similarity (imputed vs truth)",
]

# the scalar fields score_logs returns, in the order the evaluation CSV lists them
SUMMARY_FIELDS = [
    "nulls", "no_truth", "scorable", "imputed", "not_imputed", "imputed_no_truth",
    "exact_matches", "wrong", "exact_matches_incl_empty", "imputed_sim_gt_94",
    "exact%_of_nulls", "exact%_of_scorable", "exact%_of_imputed",
    "avg_semantic_sim", "avg_semantic_sim_scorable", "avg_semantic_sim_all_nulls",
    "avg_semantic_sim_text", "avg_semantic_sim_nonexact",
]

# cells that are NOT exact matches but whose cosine with the ground-truth value
# clears this threshold are counted as `imputed_sim_gt_94` - the near-exact band
NEAR_EXACT_THRESHOLD = 0.94

_MODEL_CACHE: dict[str, object] = {}


def to_native(val):
    """JSON-safe scalar: numpy types unwrapped, NaN -> None."""
    if val is None or (np.isscalar(val) and pd.isna(val)):
        return None
    return val.item() if isinstance(val, np.generic) else val


def norm_value(val):
    """Compare values across dtypes: 1996, 1996.0 and '1996' are one value."""
    if val is None or (np.isscalar(val) and pd.isna(val)):
        return None
    if isinstance(val, (int, float, np.generic)):
        f = float(val)
        return str(int(f)) if f.is_integer() else str(f)
    return str(val).strip()


def is_empty(val) -> bool:
    """True for None, NaN and the empty/whitespace string.

    The long CSVs come from Ditto lines where a missing attribute is `COL x VAL `,
    so an attribute can be empty in the ground truth itself. Such a cell carries
    nothing to recover, and counting the imputer's blank as an "exact match"
    against it is what inflated the old numbers.
    """
    return norm_value(val) in (None, "")


def get_model(embed_model: str = DEFAULT_EMBED_MODEL, model=None):
    """Reuse the caller's SentenceTransformer, else load it once per name.

    faiss_utils is imported here rather than at module scope so that a caller
    that only needs norm_value or the exact-match counts can import this module
    without pulling in faiss and torch.
    """
    if model is not None:
        return model
    if embed_model not in _MODEL_CACHE:
        from faiss_utils import load_model
        _MODEL_CACHE[embed_model] = load_model(embed_model)
    return _MODEL_CACHE[embed_model]


def value_similarities(pairs: list[tuple], *, model=None, embed_model: str = DEFAULT_EMBED_MODEL,
                       batch_size: int = 64) -> list[float | None]:
    """[(imputed, ground_truth), ...] -> [cosine or None, ...], one per pair.

    None where either side is empty - there is nothing to compare. Pairs that
    normalise equal score 1.0 without being embedded, and every remaining string
    is embedded once however often it repeats, which is what keeps this cheap:
    the columns an imputer gets right are the ones it repeats most.
    """
    out: list[float | None] = [None] * len(pairs)
    todo: list[tuple[int, str, str]] = []
    for i, (imputed, truth) in enumerate(pairs):
        if is_empty(imputed) or is_empty(truth):
            continue
        if norm_value(imputed) == norm_value(truth):
            out[i] = 1.0
        else:
            todo.append((i, str(imputed).strip(), str(truth).strip()))

    if todo:
        from faiss_utils import embed_list
        unique = sorted({text for _, a, b in todo for text in (a, b)})
        emb = embed_list(get_model(embed_model, model), unique, batch_size=batch_size)
        at = {text: k for k, text in enumerate(unique)}
        for i, a, b in todo:
            out[i] = float(np.dot(emb[at[a]], emb[at[b]]))  # embed_list L2-normalises
    return out


def _mean(values: list[float], denominator: int) -> float | None:
    """Sum of `values` spread over `denominator` cells, the cells without a score counting 0.

    None when nothing was scored at all - then the metric is unavailable, which
    is not the same as a similarity of zero.
    """
    if not denominator or not values:
        return None
    return round(float(np.sum(values) / denominator), 4)


def score_logs(logs: list[dict], target_cols: list[str], numeric_cols=(), *,
               n_nulls: int | None = None, model=None,
               embed_model: str = DEFAULT_EMBED_MODEL, score_similarity: bool = True) -> dict:
    """Every imputation-quality number, from the per-cell logs.

    One log entry per null cell, carrying `column`, `imputed_value` and
    `ground_truth_value`. The cells partition two independent ways:

        nulls    = imputed + not_imputed                  what the imputer did
        nulls    = scorable + no_truth                    what there was to recover
        imputed  = exact_matches + wrong + imputed_no_truth

    `imputed_no_truth` is a value invented where the source attribute is empty.
    It can be neither right nor wrong, so it belongs in neither count, but it is
    not nothing either: those fills reach Ditto as content.

    `exact_matches` counts only cells with a non-empty ground truth that the
    imputer reproduced. `exact_matches_incl_empty` is the old definition, which
    also credits a blank fill against a blank truth - kept so earlier summary
    CSVs remain comparable, and reported beside the honest count rather than in
    place of it.

    Similarity is averaged three ways over the same cosines, so the denominator
    is never ambiguous: over the cells that were scored, over every recoverable
    cell, and over every null. The last two count an unfilled or unrecoverable
    cell as 0.0.
    """
    numeric_cols = set(numeric_cols)
    nulls = len(logs) if n_nulls is None else int(n_nulls)

    sims = (value_similarities([(e["imputed_value"], e["ground_truth_value"]) for e in logs],
                               model=model, embed_model=embed_model)
            if score_similarity else [None] * len(logs))

    imputed = no_truth = imputed_no_truth = exact = wrong = exact_incl_empty = 0
    imputed_sim_gt_94 = 0
    scored: list[float] = []
    scored_text: list[float] = []
    scored_nonexact: list[float] = []    # sims for cells filled but not exact match
    blank = lambda: {"nulls": 0, "no_truth": 0, "scorable": 0, "filled": 0,
                     "filled_no_truth": 0, "exact": 0, "wrong": 0, "sims": []}
    per_column: dict[str, dict] = {col: blank() for col in target_cols}

    for entry, sim in zip(logs, sims):
        col = entry["column"]
        stats = per_column.setdefault(col, blank())
        filled = not is_empty(entry["imputed_value"])
        truth_empty = is_empty(entry["ground_truth_value"])
        hit = filled and not truth_empty and \
            norm_value(entry["imputed_value"]) == norm_value(entry["ground_truth_value"])

        stats["nulls"] += 1
        imputed += filled
        stats["filled"] += filled
        no_truth += truth_empty
        stats["no_truth"] += truth_empty
        stats["scorable"] += not truth_empty
        exact += hit
        stats["exact"] += hit
        exact_incl_empty += norm_value(entry["imputed_value"]) == norm_value(entry["ground_truth_value"])
        if filled and truth_empty:          # invented where the source has nothing
            imputed_no_truth += 1
            stats["filled_no_truth"] += 1
        elif filled and not hit:
            wrong += 1
            stats["wrong"] += 1
        if sim is not None:
            scored.append(sim)
            stats["sims"].append(sim)
            if col not in numeric_cols:
                scored_text.append(sim)
            # cells SENTI filled whose value did not match the truth exactly -
            # the semantically-close-but-not-identical fills
            if filled and not hit and not truth_empty:
                scored_nonexact.append(sim)
            # the near-exact band: not an exact string hit, but the embedding
            # cosine is tight enough that the imputer is semantically on target
            if not hit and sim > NEAR_EXACT_THRESHOLD:
                imputed_sim_gt_94 += 1

    scorable = nulls - no_truth
    for col, stats in per_column.items():
        col_sims = stats.pop("sims")
        stats["exact%"] = round(100 * stats["exact"] / stats["nulls"], 1) if stats["nulls"] else None
        stats["avg_sim"] = round(float(np.mean(col_sims)), 4) if col_sims else None
        stats["avg_sim_all"] = _mean(col_sims, stats["nulls"])

    return {
        "nulls": nulls,
        "no_truth": no_truth,                      # ground truth itself empty: nothing to recover
        "scorable": scorable,
        "imputed": imputed,
        "not_imputed": nulls - imputed,
        "imputed_no_truth": imputed_no_truth,      # filled where the source attribute is empty
        "exact_matches": exact,
        "wrong": wrong,                            # filled, recoverable, not the truth
        "exact_matches_incl_empty": exact_incl_empty,
        "imputed_sim_gt_94": imputed_sim_gt_94,    # near-exact: cosine > 0.94, not exact
        "exact%_of_nulls": round(100 * exact / nulls, 1) if nulls else None,
        "exact%_of_scorable": round(100 * exact / scorable, 1) if scorable else None,
        "exact%_of_imputed": round(100 * exact / imputed, 1) if imputed else None,
        "avg_semantic_sim": round(float(np.mean(scored)), 4) if scored else None,
        "avg_semantic_sim_scorable": _mean(scored, scorable),
        "avg_semantic_sim_all_nulls": _mean(scored, nulls),
        "avg_semantic_sim_text": round(float(np.mean(scored_text)), 4) if scored_text else None,
        # avg cosine over the K - E cells that were filled but not exact: a
        # similarity read of how close the "wrong" fills still are to the truth
        "avg_semantic_sim_nonexact": round(float(np.mean(scored_nonexact)), 4) if scored_nonexact else None,
        "scored_cells": len(scored),
        # False when the caller skipped the embedding pass: the similarity columns
        # are then absent, which is not the same as a similarity that came out low
        "similarity_computed": bool(score_similarity),
        "per_column": per_column,
    }


def _tuple_text(values, use_space_for_na: bool) -> str:
    """Concat the attribute values of one row into a single string.

    `use_space_for_na=True` renders a missing cell as a literal single space, so
    the tuple string still carries a placeholder where the attribute is absent -
    that is the explicit rule for the nonimputed CSV. For the ground truth and
    for the imputed frame it defaults to an empty slot: on a clean source the
    truth never has a NaN, and a cell SENTI declined to fill is honestly missing.
    """
    parts = []
    for v in values:
        if v is None or (np.isscalar(v) and pd.isna(v)):
            parts.append(" " if use_space_for_na else "")
        else:
            parts.append(str(v).strip())
    return " ".join(parts)


def tuple_similarities(ground_truth: pd.DataFrame, imputed: pd.DataFrame,
                       nonimputed: pd.DataFrame, target_cols: list[str], *,
                       model=None, embed_model: str = DEFAULT_EMBED_MODEL,
                       batch_size: int = 64) -> pd.DataFrame:
    """Cosine between the ground-truth tuple and (a) the imputed one, (b) the nonimputed one.

    Only rows with at least one null in `nonimputed[target_cols]` are included -
    tuples with nothing missing have nothing to measure. Nulls in the nonimputed
    frame are rendered as single spaces when the tuple is concatenated, so the
    comparison is honest about the missing attribute instead of silently closing
    the gap. The three frames must share row ordering and column names (both the
    long CSVs the pipeline writes and the damaged frame SENTI mutated satisfy
    this - see Data_preparation/code/imputers.run_imputation).

    Returns a DataFrame with the columns listed in TUPLE_COLUMNS. Similarities
    are percentages (cosine x 100), 4 decimals, matching the main eval CSV.
    """
    assert len(ground_truth) == len(imputed) == len(nonimputed), \
        f"row counts differ: gt={len(ground_truth)} imp={len(imputed)} non={len(nonimputed)}"

    mask = nonimputed[target_cols].isna().any(axis=1).to_numpy()
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=TUPLE_COLUMNS)

    # build the three strings once per included row
    gt_block = ground_truth[target_cols]
    imp_block = imputed[target_cols]
    non_block = nonimputed[target_cols]
    gt_texts = [_tuple_text(gt_block.iloc[i].tolist(), use_space_for_na=False) for i in idx]
    imp_texts = [_tuple_text(imp_block.iloc[i].tolist(), use_space_for_na=False) for i in idx]
    non_texts = [_tuple_text(non_block.iloc[i].tolist(), use_space_for_na=True) for i in idx]

    # embed every distinct string once - gt and imp overlap heavily where SENTI
    # recovered the exact value, so this keeps the pass cheap
    from faiss_utils import embed_list    # imported here, same reason as value_similarities
    unique = sorted({s for triple in (gt_texts, imp_texts, non_texts) for s in triple})
    emb = embed_list(get_model(embed_model, model), unique, batch_size=batch_size)
    at = {s: k for k, s in enumerate(unique)}

    # the per-pair label from the long CSV, so each tuple row carries the match
    # verdict it belongs to - absent when the ground truth has no `label` column
    labels = ground_truth["label"] if "label" in ground_truth.columns else None

    rows = []
    for i, gt_t, imp_t, non_t in zip(idx, gt_texts, imp_texts, non_texts):
        sim_imp = float(np.dot(emb[at[imp_t]], emb[at[gt_t]]))
        sim_non = float(np.dot(emb[at[non_t]], emb[at[gt_t]]))
        rows.append({
            "row_index": int(i),
            "label": to_native(labels.iat[i]) if labels is not None else None,
            "ground_truth_tuple": gt_t,
            "imputed_tuple": imp_t,
            "nonimputed_tuple": non_t,
            "semantic_similarity (imputed vs truth)": round(100 * sim_imp, 4),
            "semantic_similarity (nonimputed vs truth)": round(100 * sim_non, 4),
            "improvement (imputed - nonimputed)": round(100 * (sim_imp - sim_non), 4),
        })
    return pd.DataFrame(rows, columns=TUPLE_COLUMNS)


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def format_report(m: dict, indent: str = "    ") -> str:
    """The metrics block printed per file. Every rate names its denominator."""
    n, imp, scorable = m["nulls"], m["imputed"], m["scorable"]
    sim, sim_s, sim_all = (m["avg_semantic_sim"], m["avg_semantic_sim_scorable"],
                           m["avg_semantic_sim_all_nulls"])
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
    lines = [
        f"nulls          : {n}",
        f"imputed        : {imp} ({_pct(imp, n)} of nulls)   not imputed: "
        f"{m['not_imputed']} ({_pct(m['not_imputed'], n)})",
        f"exact matches  : {m['exact_matches']} ({_pct(m['exact_matches'], n)} of nulls, "
        f"{_pct(m['exact_matches'], scorable)} of scorable, {_pct(m['exact_matches'], imp)} of imputed)",
        f"wrong          : {m['wrong']} ({_pct(m['wrong'], n)} of nulls)",
        (f"semantic sim   : {fmt(sim)} over the {m['scored_cells']} cells scored, "
         f"{fmt(sim_s)} over scorable, {fmt(sim_all)} over all nulls")
        if m.get("similarity_computed", True) else
        "semantic sim   : not computed (no embedding pass for this strategy)",
    ]
    if m["no_truth"]:
        lines.append(f"no ground truth: {m['no_truth']} ({_pct(m['no_truth'], n)} of nulls) - the "
                     f"attribute is empty in the source, so nothing is recoverable; "
                     f"{m['imputed_no_truth']} of them were filled anyway")
    return "\n".join(indent + line for line in lines)


def value_similarity_rows(logs: list[dict], ground_truth: pd.DataFrame | None = None,
                          *, model=None, embed_model: str = DEFAULT_EMBED_MODEL,
                          batch_size: int = 64) -> pd.DataFrame:
    """Cosine between the ground-truth value and the imputed value, cell by cell.

    One row per per-cell log entry, so this is the per-cell counterpart of
    tuple_similarities (which concatenates every target column of a tuple).
    The DataFrame carries the columns listed in VALUE_COLUMNS; similarity is a
    percentage (cosine x 100) and 0.0 when either side is empty, so the file
    is self-contained even where there was nothing to score.

    When `ground_truth` is given its `label` column is pulled in per log entry
    (via `row_index`), so each cell row carries the pair's match label. The
    column stays in the schema even without a frame; the value is then None.
    """
    if not logs:
        return pd.DataFrame(columns=VALUE_COLUMNS)
    pairs = [(e["imputed_value"], e["ground_truth_value"]) for e in logs]
    sims = value_similarities(pairs, model=model, embed_model=embed_model,
                              batch_size=batch_size)
    labels = (ground_truth["label"]
              if ground_truth is not None and "label" in ground_truth.columns
              else None)
    rows = []
    for e, sim in zip(logs, sims):
        ridx = int(e["row_index"])
        rows.append({
            "row_index": ridx,
            "label": to_native(labels.iat[ridx]) if labels is not None else None,
            "column": e["column"],
            "ground_truth_value": e["ground_truth_value"],
            "imputed_value": e["imputed_value"],
            # percentage to match the main eval CSV; empty-side pairs score 0.0
            "semantic_similarity (imputed vs truth)": round(100 * sim, 4) if sim is not None else 0.0,
        })
    return pd.DataFrame(rows, columns=VALUE_COLUMNS)
