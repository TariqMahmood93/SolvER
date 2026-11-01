# evaluation.py

import os
import time
import numpy as np
from Logs import save_evaluation

def evaluate(
    df_ground_truth,
    cum_df,
    logs,
    n_prev,
    n_cur,
    nulls_in_new,
    startcsv_path,
    tag_imp,
    faiss_time,
    imputation_time,
    total_processing_time,
):
    new_logs = [e for e in logs if e["row_index"] >= n_prev]
    exact_matches = sum(1 for e in new_logs if e["imputed_value"] == e["ground_truth_value"])

    sim_scores = []
    for e in new_logs:
        try:
            idx = e["extracted_values"].index(e["imputed_value"])
            sim_scores.append(e["similarity_scores"][idx])
        except ValueError:
            continue
    avg_sim = float(np.mean(sim_scores)) if sim_scores else None

    out_cum  = os.path.join(startcsv_path, f"{tag_imp}.csv")
    eval_csv = os.path.join(startcsv_path, f"{tag_imp}_evaluation.csv")

    save_evaluation(
        file_name=os.path.basename(out_cum),
        start_index=n_prev,
        end_index=n_cur - 1,
        nulls=nulls_in_new,
        exact_matches=exact_matches,
        avg_similarity=avg_sim,
        processing_time=total_processing_time,
        faiss_time=faiss_time,
        imputation_time=imputation_time,
        total_processing_time=total_processing_time,
        out_csv=eval_csv,
    )
    print("    Saved evaluation CSV:", os.path.basename(eval_csv))
