"""Shared settings, paths and schema for merge_splits.ipynb.

Every parameter is defined once, in the notebook's **first cell**. That cell calls
`write_params()`, which stores them in `<split>_<DATASET>_params.json` next to the
data. Any later cell can then be run on its own - restarted kernel, or "run this
cell" without the ones above - because its guard calls `load_context()` and gets
the same values back, instead of failing with `NameError: name 'ROOT' is not defined`.
"""
from __future__ import annotations

import json
from itertools import accumulate
from pathlib import Path

SPLITS = ["train", "valid", "test"]
ID_COLS = ["split", "pair_id", "label"]   # + "side" identify an entity row in the long CSV
SIDES = ["left", "right"]

# parameter name in the json  ->  the notebook global it becomes
PARAM_GLOBALS = {
    "dataset": "DATASET", "source_split": "SOURCE_SPLIT", "seed": "SEED", "ratios": "RATIOS",
    "mechanisms": "MECHANISMS", "increments": "PCTS", "null_seed": "NULL_SEED",
    "strength": "STRENGTH", "mar_observed": "MAR_OBSERVED", "drop_stale": "DROP_STALE",
    "imputer": "IMPUTER",
    "embed_model": "EMBED_MODEL", "median_k": "MEDIAN_K", "top_k": "TOP_K",
    "sim_th": "SIM_TH", "high_th": "HIGH_TH",
}


def as_level(frac: float) -> float:
    """A cumulative fraction -> the percentage used in the file names (int when whole)."""
    p = round(100 * frac, 6)
    return int(p) if float(p).is_integer() else p


def levels_of(increments: list[float]) -> list[float]:
    """Increments -> the cumulative levels. [0.05, 0.05, 0.1, 0.2] -> [5, 10, 20, 40]."""
    return [as_level(c) for c in accumulate(increments)]


def steps_of(increments: list[float]) -> dict[float, float]:
    """Cumulative level -> the increment that produced it."""
    return dict(zip(levels_of(increments), (as_level(i) for i in increments)))


def data_dir_of(root: Path, dataset: str) -> Path:
    return Path(root) / "Data_preparation" / "Data" / dataset


def params_path(root: Path, dataset: str, source_split: str) -> Path:
    return data_dir_of(root, dataset) / f"{source_split}_{dataset}_params.json"


def find_root(explicit: str | None = None) -> Path:
    """The folder holding Data_preparation/Data, from an explicit path, $SOLVIER_ROOT or the cwd."""
    import os

    hint = explicit or os.environ.get("SOLVIER_ROOT")
    starts = [Path(hint).expanduser().resolve()] if hint else [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for p in starts:
        if (p / "Data_preparation" / "Data").is_dir():
            return p
    raise FileNotFoundError(
        "could not find the project root. Run the first cell (it has PROJECT_ROOT), "
        "or export SOLVIER_ROOT=/path/to/SolvIER-main before starting Jupyter."
    )


def write_params(root: Path, params: dict) -> Path:
    """Store the first cell's parameters next to the data. Returns the file path."""
    missing = set(PARAM_GLOBALS) - set(params)
    assert not missing, f"write_params is missing {sorted(missing)}"
    fp = params_path(root, params["dataset"], params["source_split"])
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return fp


def load_params(root: Path, dataset: str | None = None, source_split: str | None = None) -> tuple[dict, Path]:
    """The stored parameters. With no dataset, the most recently written file is used."""
    if dataset and source_split:
        fp = params_path(root, dataset, source_split)
    else:
        base = Path(root) / "Data_preparation" / "Data"
        found = sorted(base.glob("*/*_params.json"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise FileNotFoundError(f"no *_params.json under {base} - run the first cell of the notebook first")
        fp = found[-1]
    if not fp.is_file():
        raise FileNotFoundError(f"{fp.name} not found - run the first cell of the notebook first")
    return json.loads(fp.read_text(encoding="utf-8")), fp


def load_context(root: Path | None = None, dataset: str | None = None,
                 source_split: str | None = None) -> dict:
    """Every name the notebook cells share, rebuilt from the stored parameters."""
    import pandas as pd

    root = Path(root) if root else find_root()
    params, fp = load_params(root, dataset, source_split)
    ctx = {PARAM_GLOBALS[k]: v for k, v in params.items() if k in PARAM_GLOBALS}
    data_dir = data_dir_of(root, ctx["DATASET"])

    long_fp = data_dir / f"{ctx['SOURCE_SPLIT']}_{ctx['DATASET']}_long.csv"
    attrs = ([c for c in pd.read_csv(long_fp, nrows=0).columns if c not in ID_COLS + ["side"]]
             if long_fp.is_file() else None)

    return {**ctx, "ROOT": root, "DATA_DIR": data_dir, "PARAMS_FP": fp,
            "SPLITS": list(SPLITS), "ID_COLS": list(ID_COLS), "SIDES": list(SIDES), "ATTRS": attrs,
            "PCT_LEVELS": levels_of(ctx["PCTS"]), "STEPS": steps_of(ctx["PCTS"])}
