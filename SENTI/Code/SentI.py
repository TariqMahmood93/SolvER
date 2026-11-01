# SentI.py
import os
import glob
import re
import time
import gc
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from faiss_utils import load_model, embed_list, build_vector_DB, update_vector_DB
from Saving_output import save_outputs
from Evaluation import evaluate

def _to_native(val):
    if pd.isna(val):
        return None
    if isinstance(val, (np.generic,)):
        return val.item()
    return val

def run_imputation(
    startcsv_path: str,
    dataset: str,
    seeds: list[int],
    cum_pcts: list[float],
    initial_size: int,
    step: int,
    embed_model: str = "sentence-transformers/LaBSE",
) -> None:
    model = load_model(embed_model)
    print(f"\nLoaded embedding model on device: {model.device}", flush=True)

    MEDIAN_K, SIM_TH, HIGH_TH, TOP_K = 100, 0.65, 1.0, 25
    df_ground_truth = pd.read_csv(os.path.join(startcsv_path, f"{dataset}.csv"))
    cat_cols = [
        c for c in df_ground_truth.columns
        if df_ground_truth[c].dtype == object
        or isinstance(df_ground_truth[c].dtype, pd.CategoricalDtype)
    ]

    for seed in seeds:
        for pct in cum_pcts:
            print(f"\n--- Seed {seed}, {int(pct*100)}% → imputation ---", flush=True)
            pattern = os.path.join(
                startcsv_path,
                # f"{dataset}_subset_*_{int(pct*100)}_{seed}_nonimputed.csv"
                f"{dataset}_*_{int(pct*100)}_{seed}_nonimputed.csv"

            )
            files = sorted(
                [(f, int(re.search(r"_*_(\d+)_", f).group(1)))
                 for f in glob.glob(pattern)
                 if re.search(r"_*_(\d+)_", f)],
                key=lambda x: x[1],
            )
            files = [f for f, _ in files]

            faiss_db = None
            emb_matrix = None
            texts = []
            cum_df = None

            for fp in files:
                tag_nonimp = os.path.basename(fp).rsplit(".",1)[0]
                tag_imp    = tag_nonimp.replace("nonimputed","imputed")
                print(f"\n  Processing {tag_nonimp}", flush=True)

                df    = pd.read_csv(fp)
                n_cur = len(df)
                n_prev= len(texts)
                nulls_in_new = int(df.iloc[n_prev:n_cur][cat_cols].isna().sum().sum())

                if cum_df is not None and len(cum_df) >= n_prev:
                    df.iloc[:n_prev] = cum_df.iloc[:n_prev].values

                # FAISS index build/update
                print(f"    → FAISS build/update for rows {n_prev}–{n_cur-1}", flush=True)
                start_faiss = time.time()
                if faiss_db is None:
                    texts      = df.fillna("").astype(str).agg(" ".join, axis=1).tolist()
                    emb_matrix = embed_list(model, texts)
                    faiss_db   = build_vector_DB(emb_matrix)
                else:
                    new_texts = df.iloc[n_prev:n_cur].fillna("").astype(str).agg(" ".join, axis=1).tolist()
                    new_emb   = embed_list(model, new_texts)
                    ids       = np.arange(n_prev, n_cur, dtype=np.int64)
                    faiss_db.add_with_ids(new_emb, ids)
                    texts.extend(new_texts)
                    emb_matrix = np.vstack([emb_matrix, new_emb])
                faiss_time = time.time() - start_faiss
                print(f"    → FAISS time: {faiss_time:.2f}s", flush=True)

                # Imputation
                all_missing = np.where(df[cat_cols].isna().any(axis=1))[0]
                missing     = [r for r in all_missing if n_prev <= r < n_cur]
                print(f"    → Imputing {len(missing)} rows", flush=True)

                logs = []
                start_imp = time.time()
                for ridx in tqdm(missing, desc="      Imputing", leave=False):
                    D, I = faiss_db.search(emb_matrix[ridx:ridx+1], MEDIAN_K)
                    sims  = ((D[0] + 1) / 2).tolist()
                    neigh = I[0].tolist()
                    query = texts[ridx]

                    for c in cat_cols:
                        ci = df.columns.get_loc(c)
                        if pd.isna(df.iat[ridx, ci]):
                            valid_neighbors = [
                                (nj, s) for nj, s in zip(neigh, sims)
                                if pd.notna(df.iat[nj, ci])
                            ][:TOP_K]

                            direct = next(
                                (df.iat[nj, ci] for nj, s in valid_neighbors if s >= HIGH_TH),
                                None
                            )
                            if direct is not None:
                                imp = direct
                            else:
                                cand = [df.iat[nj, ci] for nj, s in valid_neighbors if s >= SIM_TH]
                                imp = Counter(cand).most_common(1)[0][0] if cand else None

                            logs.append({
                                "row_index":         int(ridx),
                                "query":             query,
                                "top_neighbors":     [texts[nj] for nj, _ in valid_neighbors],
                                "extracted_values":  [_to_native(df.iat[nj, ci]) for nj, _ in valid_neighbors],
                                "similarity_scores": [s for _, s in valid_neighbors],
                                "imputed_value":     imp,
                                "ground_truth_value": _to_native(
                                    df_ground_truth.iat[
                                        ridx,
                                        df_ground_truth.columns.get_loc(c)
                                    ]
                                )
                            })
                            df.iat[ridx, ci] = imp

                    # add updated embedding for this row
                    new_txt = " ".join(df.iloc[ridx].fillna("").astype(str).tolist())
                    new_emb = embed_list(model, [new_txt])[0]
                    update_vector_DB(faiss_db, new_emb, ridx)
                    emb_matrix[ridx] = new_emb
                    texts[ridx]      = new_txt

                imputation_time = time.time() - start_imp
                print(f"    → Imputation time: {imputation_time:.2f}s", flush=True)

                # Save + evaluate
                cum_df = save_outputs(df, cum_df, n_prev, n_cur, startcsv_path, tag_imp, logs)
                total = faiss_time + imputation_time
                evaluate(
                    df_ground_truth, cum_df, logs,
                    n_prev, n_cur, nulls_in_new,
                    startcsv_path, tag_imp,
                    faiss_time, imputation_time, total
                )
                gc.collect()
