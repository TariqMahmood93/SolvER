# Logs.py

from __future__ import annotations  # PEP 604 unions (X | Y) on Python 3.9

import os
import pandas as pd

def save_evaluation(
    file_name: str,
    start_index: int,
    end_index: int,
    nulls: int,
    exact_matches: int,
    avg_similarity: float | None,
    processing_time: float,
    faiss_time: float,
    imputation_time: float,
    total_processing_time: float,
    out_csv: str,
    per_column: dict | None = None,
    metrics: dict | None = None,
) -> None:
    """One-row metrics CSV.

    `metrics` is similarity.score_logs' output: when given, every count and rate
    it holds is written too, each under its own name, so the denominator a rate
    used is never in doubt. Omitted by the older chunked path in Evaluation.py,
    which still writes the four original columns only.
    """
    df = pd.DataFrame(
        [{
            "file":                  file_name,
            "start_index":           start_index,
            "end_index":             end_index,
            "nulls":                 nulls,
            "exact_matches_SENTI":         exact_matches,
            # cosine(imputed value, ground-truth value) over the cells that were scored
            "avg_semantic_sim_SENTI":        avg_similarity,
            "processing_time_SENTI":       processing_time,
            "faiss_time_SENTI":            faiss_time,
            "imputation_time_SENTI":       imputation_time,
            "total_time_SENTI": total_processing_time
        }]
    )
    if metrics:
        for key, val in metrics.items():
            if key != "per_column":
                df[f"{key}_SENTI"] = val
    if per_column:
        # nulls_/filled_/exact_/exact%_ per attribute, so a problem column is visible
        for col, stats in per_column.items():
            for key, val in stats.items():
                df[f"{col}_{key}"] = val
    df.to_csv(out_csv, index=False)
    print("    Saved evaluation CSV:", os.path.basename(out_csv))
