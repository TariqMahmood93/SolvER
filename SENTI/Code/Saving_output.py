# save_outputs.py
import json
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

__all__ = ["save_outputs"]


def _try_load_master(startcsv_path: str, dataset: str) -> Optional[pd.DataFrame]:
    """Load existing master imputed CSV if it exists."""
    master_fp = Path(startcsv_path) / f"{dataset}_imputed.csv"
    if master_fp.is_file():
        return pd.read_csv(master_fp)
    return None


def save_outputs(
    df: pd.DataFrame,
    cum_df: Optional[pd.DataFrame],
    n_prev: int,
    n_cur: int,
    startcsv_path: str,
    tag_imp: str,
    logs: List[Dict],
) -> pd.DataFrame:
    """Save artefacts for one imputation batch and return updated `cum_df`."""
    out_dir = Path(startcsv_path)

    # 1) Delta imputed CSV
    new_imputed = df.iloc[n_prev:n_cur].reset_index(drop=True)
    out_new = out_dir / f"{tag_imp}.csv"
    new_imputed.to_csv(out_new, index=False)
    print("    Saved chunked imputed CSV:", out_new.name)

    # 2) Cumulative subset CSV
    cum_df = new_imputed.copy() if cum_df is None else pd.concat([cum_df, new_imputed], ignore_index=True)
    out_cum = out_dir / f"{tag_imp}.csv"
    cum_df.to_csv(out_cum, index=False)
    print("    Saved cumulative subset CSV:", out_cum.name)

    # # 3) Master imputed CSV
    # dataset = tag_imp.split("_subset_")[0]
    # master_fp = out_dir / f"{dataset}_imputed.csv"
    # master_df = _try_load_master(startcsv_path, dataset)
    # if master_df is None:
    #     master_df = new_imputed.copy()
    # else:
    #     master_df = pd.concat([master_df, new_imputed], ignore_index=True)
    # master_df.to_csv(master_fp, index=False)
    # print("    Saved master imputed CSV:", master_fp.name)

    # 4) JSON log
    out_json = out_dir / f"{tag_imp}.json"
    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(logs, jf, indent=2)
    print("    Saved log JSON:", out_json.name)

    return cum_df
