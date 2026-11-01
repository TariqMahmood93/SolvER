# null_injection.py
# ------------------------------------------------------------
# Generates incrementally larger CSVs with injected categorical
# nulls. For every subset it writes TWO outputs:
#   1) {dataset}_{i}_{pct}_{seed}_nonimputed.csv
#   2) {dataset}_{i}_{pct}_{seed}_nonimputed_chunk{i}.csv
# ------------------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
from Chunks_1 import create_chunks

def inject_nulls(df, pct, seed, cat_cols, prev_nulls, new_idx):
    np.random.seed(seed)
    random.seed(seed)
    out = df.copy()
    desired = int(len(new_idx) * pct)  # Only consider new rows!

    # Reapply earlier nulls (if any) to the current subset
    for c, idxs in prev_nulls.items():
        out.loc[list(idxs), c] = np.nan

    updated = {c: set(prev_nulls[c]) for c in cat_cols}

    print(f"    → Injecting into rows {new_idx[0]}–{new_idx[-1]}")
    for c in cat_cols:
        existing = len([i for i in prev_nulls[c] if i in new_idx])
        need     = max(0, desired - existing)
        if need == 0:
            continue
        print(f"    → Column {c!r}: injecting {need} nulls")

        candidates = [i for i in new_idx if pd.notna(out.at[i, c]) and i not in prev_nulls[c]]
        pick = random.sample(candidates, min(need, len(candidates)))

        out.loc[pick, c] = np.nan
        updated[c].update(pick)

    return out, updated


def run_null_injection(
    startcsv_path: str,
    dataset: str,
    seeds: list[int],
    cum_pcts: list[float],
    initial_size: int,
    step: int,
) -> None:

    print(f"\n=== Dataset: {dataset} ===")
    df_full = pd.read_csv(os.path.join(startcsv_path, f"{dataset}.csv"))

    cat_cols = [
        c for c in df_full.columns
        if df_full[c].dtype == object or isinstance(df_full[c].dtype, pd.CategoricalDtype)
    ]

    subset_ends = create_chunks(len(df_full), initial_size, step)
    print("  Chunk end-indices:", subset_ends)

    for seed in seeds:
        print(f"\n--- Seed {seed} (injection) ---")
        prev_end = 0
        base_df = pd.DataFrame()

        for i, end in enumerate(subset_ends, start=1):
            print(f"\n  Building chunk {i}: rows 0…{end - 1}")
            if i == 1:
                base_df = df_full.iloc[:end].reset_index(drop=True)
            else:
                new_rows = df_full.iloc[prev_end:end].reset_index(drop=True)
                base_df = pd.concat([base_df, new_rows], ignore_index=True)
            new_idx = list(range(prev_end, end))
            prev_end = end

            # Fresh tracker for each cumulative % in this chunk
            chunk_nulls = {c: set() for c in cat_cols}

            for pct in cum_pcts:
                print(f"  Injecting {int(pct * 100)}% nulls into new rows")
                base_df, chunk_nulls = inject_nulls(
                    base_df, pct, seed, cat_cols, chunk_nulls, new_idx
                )

                full_fname = f"{dataset}_{i}_{int(pct*100)}_{seed}_nonimputed.csv"
                delta_fname = f"{dataset}_{i}_{int(pct*100)}_{seed}_nonimputed_chunk{i}.csv"

                base_df.to_csv(os.path.join(startcsv_path, full_fname), index=False)
                base_df.iloc[new_idx].to_csv(os.path.join(startcsv_path, delta_fname), index=False)

                print("    Saved:", full_fname)
                print("    Saved new-chunk CSV:", delta_fname)
