"""Read the imputation evaluation CSVs and plot them, one line per dataset.

Loaded by plot_imputation.ipynb, which is run AFTER merge_splits.ipynb has been
run for every dataset: this module only reads what data prep wrote, and computes
nothing about imputation itself.

    {split}_{DATASET}_long_{MECH}_{pct}_{seed}_imputed_{strategy}_evaluation.csv

One such file per (dataset, mechanism, null level, seed, strategy), each a single
row of metrics. `load_evaluations` globs them into one tidy frame; `metric_figure`
draws it as x = null %, y = the metric, colour = dataset, panel = strategy x
mechanism.

Colours are the validated categorical palette, assigned by the dataset's position
in a FIXED order, so filtering datasets out never repaints the ones that remain.
Each dataset also carries its own marker, because colour alone is not an identity
channel: one adjacent pair in this palette sits in the tritan CVD band, and three
light-mode slots fall below 3:1 against the surface. That is why every figure has
a `table_view` twin - no value is reachable only by eye.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

__all__ = ["DATASET_ORDER", "SIM_METRICS", "EXACT_METRICS", "REQUIRED_COLUMNS",
           "find_root", "load_evaluations", "metric_figure", "table_view", "describe"]

# the fixed colour order: a dataset keeps its hue however the selection changes
DATASET_ORDER = ["DBLPACM", "DBLPGoogleScholar", "AmazonGoogle", "WalmartAmazon",
                 "FodorsZagats", "iTunesAmazon", "Beer"]

# categorical slots 1-8, light and dark, from the validated palette. Both columns
# pass the lightness band, chroma floor, adjacent CVD and normal-vision gates in
# their own mode; the dark column is its own set of steps, never a flip of light.
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500",
             "#d55181", "#008300", "#9085e9", "#e66767"],
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]        # secondary encoding
THEME = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e5e4e0"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#33322f"},
}

SIM_METRICS = {
    "avg_sim_imputed": "cosine(imputed, truth) %, over the K cells SENTI filled",
    "avg_sim_all_nulls": "cosine(imputed, truth) %, over every null (unfilled count 0)",
}
EXACT_METRICS = {
    "exact_pct_of_nulls": "exact matches, % of all injected nulls",
    "exact_pct_of_imputed": "exact matches, % of the cells actually filled",
}

# the 10-column slim CSV's human-readable names, mapped to short identifiers that
# downstream code can type. Rename happens on load; the CSV on disk keeps the
# descriptive names, the frame in memory uses the short ones
_CSV_TO_SHORT = {
    "total_imputed": "imputed",
    "semantic_similarity (Average semantic similarity over imputed values)": "avg_sim_imputed",
    "semantic_similarity (Average semantic similarity over all nulls)":      "avg_sim_all_nulls",
    "exact matches (over all nulls)":                                        "exact_pct_of_nulls",
    "exact matches (over imputed values)":                                   "exact_pct_of_imputed",
}
# any one of these present identifies a slim-schema CSV
REQUIRED_COLUMNS = list(_CSV_TO_SHORT)

_STEM = re.compile(r"_long_(?P<mechanism>[A-Z]+)_(?P<pct>\d+(?:\.\d+)?)_(?P<seed>\d+)"
                   r"_imputed_(?P<imputer>[a-z]+)_evaluation$")


def find_root(explicit: str | Path | None = None) -> Path:
    """The folder holding Data_preparation/Data - as context.py resolves it."""
    import os

    hint = explicit or os.environ.get("SOLVIER_ROOT")
    starts = ([Path(hint).expanduser().resolve()] if hint
              else [Path.cwd().resolve(), *Path.cwd().resolve().parents])
    for p in starts:
        if (p / "Data_preparation" / "Data").is_dir():
            return p
    raise FileNotFoundError(
        "could not find the project root. Set PROJECT_ROOT in the first cell, or "
        "export SOLVIER_ROOT=/path/to/SolvIER2 before starting Jupyter.")


def load_evaluations(root: str | Path | None = None, *, datasets: list[str] | None = None,
                     seeds: list[int] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Every *_imputed_*_evaluation.csv under Data_preparation/Data, as one frame.

    The dataset comes from the folder the file sits in, the rest from its name.
    Files written before the value-based scoring are listed and skipped rather
    than mixed in, because their `avg_semantic_sim` is a different measurement.
    """
    root = find_root(root)
    data_root = root / "Data_preparation" / "Data"

    rows, stale, unparsed = [], [], []
    for fp in sorted(data_root.glob("*/*_imputed_*_evaluation.csv")):
        match = _STEM.search(fp.stem)
        if not match:
            unparsed.append(fp)
            continue
        dataset = fp.parent.name
        if datasets and dataset not in datasets:
            continue
        meta = match.groupdict()
        if seeds and int(meta["seed"]) not in seeds:
            continue
        df = pd.read_csv(fp).rename(columns=_CSV_TO_SHORT)
        missing = [c for c in _CSV_TO_SHORT.values() if c not in df.columns]
        if missing:
            stale.append((fp, missing))
            continue
        # the slim CSV already carries mechanism/null_pct/seed, so trust those
        # over the filename parse; dataset still comes from the folder name
        rows.append({"dataset": dataset, "mechanism": meta["mechanism"],
                     "null_pct": float(meta["pct"]), "seed": int(meta["seed"]),
                     "imputer": meta["imputer"], **df.iloc[0].to_dict(),
                     "source": str(fp.relative_to(root))})

    if verbose:
        if unparsed:
            print(f"{len(unparsed)} file(s) did not match the expected name and were skipped, "
                  f"e.g. {unparsed[0].name}")
        if stale:
            print(f"WARNING: {len(stale)} evaluation file(s) predate the value-based scoring "
                  f"and were skipped - re-run Step 4 of merge_splits.ipynb for them:")
            for fp, missing in stale[:5]:
                print(f"   {fp.parent.name}/{fp.name}  (no {', '.join(missing[:3])})")
            if len(stale) > 5:
                print(f"   ... and {len(stale) - 5} more")

    if not rows:
        raise FileNotFoundError(
            f"no usable *_imputed_*_evaluation.csv under {data_root}. Run Step 4 of "
            f"merge_splits.ipynb for each dataset first.")

    out = pd.DataFrame(rows)

    # datasets in the fixed colour order, then any name the order does not know
    known = [d for d in DATASET_ORDER if d in set(out.dataset)]
    extra = sorted(set(out.dataset) - set(DATASET_ORDER))
    out["dataset"] = pd.Categorical(out.dataset, categories=known + extra, ordered=True)
    out["imputer"] = pd.Categorical(out.imputer, categories=(
        ["senti"] if set(out.imputer) == {"senti"} else sorted(set(out.imputer))), ordered=True)
    out["mechanism"] = pd.Categorical(out.mechanism, categories=[m for m in
                                      ("MCAR", "MAR", "MNAR") if m in set(out.mechanism)],
                                      ordered=True)
    return out.sort_values(["imputer", "mechanism", "dataset", "null_pct", "seed"]).reset_index(drop=True)


def describe(df: pd.DataFrame) -> None:
    """What the loaded frame actually covers - read this before trusting a figure."""
    print(f"{len(df)} evaluation rows")
    print(f"  datasets  : {list(df.dataset.cat.categories)}")
    print(f"  imputers  : {list(df.imputer.cat.categories)}")
    print(f"  mechanisms: {list(df.mechanism.cat.categories)}")
    print(f"  null %    : {sorted(df.null_pct.unique())}")
    print(f"  seeds     : {sorted(df.seed.unique())}")
    size = (df.groupby("dataset", observed=True)[["imputed"]].sum()
              .rename(columns={"imputed": "K filled"}))
    print("  totals per dataset:")
    print("\n".join("    " + line for line in size.to_string().splitlines()))
    holes = (df.groupby(["imputer", "mechanism", "dataset"], observed=True).null_pct
               .nunique().pipe(lambda s: s[s < df.null_pct.nunique()]))
    if len(holes):
        print(f"  NOTE: {len(holes)} (imputer, mechanism, dataset) group(s) are missing some "
              f"null levels - their lines stop early")


def _colour_of(dataset: str, categories: list[str], mode: str) -> tuple[str, str]:
    """(colour, marker) for a dataset - by its position in the fixed order."""
    i = categories.index(dataset)
    if i >= len(PALETTE[mode]):
        raise ValueError(
            f"{len(categories)} datasets exceed the {len(PALETTE[mode])} validated colour slots. "
            f"Pass datasets=[...] to plot a subset; a generated 9th hue is not colourblind-safe.")
    return PALETTE[mode][i], MARKERS[i]


def metric_figure(df: pd.DataFrame, metric: str, *, dark: bool = False,
                  title: str | None = None, ylabel: str | None = None,
                  ylim: tuple[float, float] | None = None, share_y: str = "row",
                  band: bool = True, save_to: str | Path | None = None, dpi: int = 150):
    """x = null %, y = `metric`, one coloured line per dataset.

    One panel per (imputer, mechanism): the two are never averaged over, because
    a mechanism changes which cells go missing and a strategy changes how they
    are filled. Where several seeds exist the line is their mean and the wash
    around it their min-max spread.

    `share_y="row"` gives each strategy its own y-range, so a row is not flattened
    by another strategy's scale; the three mechanisms in a row always share it, so
    reading across a row is exact. Pass `share_y="all"` or an explicit `ylim` to
    compare rows directly instead.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    assert metric in df.columns, f"{metric!r} is not a column; have {sorted(df.columns)[:12]}..."
    mode = "dark" if dark else "light"
    th = THEME[mode]
    datasets = list(df.dataset.cat.categories)
    # a strategy with nothing to show for this metric gets no row rather than an
    # empty one - e.g. a run made with score_similarity=False
    has = df[df[metric].notna()]
    if has.empty:
        raise ValueError(f"every row has an empty {metric!r} - nothing to plot. "
                         f"For a similarity metric, only strategies that were scored have one.")
    imputers = [i for i in df.imputer.cat.categories if (has.imputer == i).any()]
    mechanisms = [m for m in df.mechanism.cat.categories if (has.mechanism == m).any()]
    skipped = [i for i in df.imputer.cat.categories if i not in imputers]
    if skipped:
        print(f"no {metric} for {', '.join(map(str, skipped))} - omitted from this figure")
    df = df[df.imputer.isin(imputers) & df.mechanism.isin(mechanisms)]

    fig, axes = plt.subplots(len(imputers), len(mechanisms), squeeze=False,
                             figsize=(3.6 * len(mechanisms) + 1.2, 2.5 * len(imputers) + 1.2),
                             sharex=True, sharey=("all" if ylim else share_y))
    fig.patch.set_facecolor(th["surface"])

    levels = sorted(df.null_pct.unique())
    for r, imputer in enumerate(imputers):
        for c, mech in enumerate(mechanisms):
            ax = axes[r][c]
            ax.set_facecolor(th["surface"])
            panel = df[(df.imputer == imputer) & (df.mechanism == mech)]
            for name in datasets:
                series = panel[panel.dataset == name]
                if series.empty:
                    continue
                colour, marker = _colour_of(name, datasets, mode)
                agg = (series.groupby("null_pct", observed=True)[metric]
                             .agg(["mean", "min", "max", "count"]).reset_index()
                             .sort_values("null_pct"))
                if band and agg["count"].max() > 1:
                    ax.fill_between(agg.null_pct, agg["min"], agg["max"],
                                    color=colour, alpha=0.10, linewidth=0)
                ax.plot(agg.null_pct, agg["mean"], color=colour, linewidth=2,
                        solid_capstyle="round", solid_joinstyle="round",
                        marker=marker, markersize=7, markeredgecolor=th["surface"],
                        markeredgewidth=1.5, label=name, zorder=3)

            ax.grid(axis="y", color=th["grid"], linewidth=0.8, linestyle="-")
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(th["grid"])
                ax.spines[side].set_linewidth(0.8)
            ax.tick_params(colors=th["muted"], labelsize=9, length=0)
            ax.set_xticks(levels)
            if ylim:
                ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(mech, color=th["text"], fontsize=11, pad=8)
            if c == 0:
                ax.set_ylabel(f"{imputer}", color=th["text"], fontsize=11)
            if r == len(imputers) - 1:
                ax.set_xlabel("nulls injected (%)", color=th["muted"], fontsize=9)

    # marker-only keys: a short line stub either side of the marker reads as a
    # dashed line, and nothing in these charts is dashed
    present = [d for d in datasets if (df.dataset == d).any()]
    if len(present) >= 2:         # identity is never colour-alone
        handles = [Line2D([0], [0], linestyle="none", marker=_colour_of(d, datasets, mode)[1],
                          color=_colour_of(d, datasets, mode)[0], markersize=8,
                          markeredgecolor=th["surface"], markeredgewidth=1.5) for d in present]
        fig.legend(handles, present, loc="lower center", ncol=min(len(present), 4),
                   frameon=False, fontsize=9, labelcolor=th["text"],
                   handletextpad=0.5, columnspacing=1.6, bbox_to_anchor=(0.5, -0.01))
    head = title or SIM_METRICS.get(metric) or EXACT_METRICS.get(metric) or metric
    fig.suptitle(head, color=th["text"], fontsize=13, y=0.995)
    fig.supylabel(ylabel or metric, color=th["muted"], fontsize=10, x=0.005)
    fig.tight_layout(rect=(0.02, 0.06, 1, 0.97))

    if save_to:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=dpi, facecolor=th["surface"], bbox_inches="tight")
        print(f"saved {save_to}")
    return fig


def table_view(df: pd.DataFrame, metric: str, *, decimals: int = 4) -> pd.DataFrame:
    """The figure's values as a table - the WCAG-clean twin of every chart here."""
    assert metric in df.columns, f"{metric!r} is not a column"
    out = (df.pivot_table(index=["imputer", "mechanism", "dataset"], columns="null_pct",
                          values=metric, aggfunc="mean", observed=True)
             .round(decimals)
             .dropna(how="all"))      # a strategy that was never scored has no row
    out.columns = [f"{c:g}%" for c in out.columns]
    return out
