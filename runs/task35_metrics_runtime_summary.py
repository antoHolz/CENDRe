import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import json
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Same heuristics as task32_runtime_summary.py / task31_plot_metrics.py:
# - CENDRE-prefixed methods determine n_concepts internally → only cnr_0 exists.
# - CENDREkMeans is the exception: it uses n_concepts directly, like ECLAD.
CENDRE_PREFIXES = ("CENDRE",)
CENDRE_USES_N_CONCEPTS = ("CENDREkMeans",)

CE_DISPLAY_NAMES = {
    "ECLADts": "ECLAD-ts",
}

STYLE_RC = {
    "axes.grid": True,
    "axes.facecolor": "white",
    "grid.linestyle": "-",
    "grid.alpha": 0.5,
    "grid.color": "tab:gray",
    "figure.facecolor": "white",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["computer modern roman"],
    "font.size": 25,
    "axes.titlesize": 30,
    "axes.labelsize": 25,
    "xtick.labelsize": 17.5,
    "ytick.labelsize": 17.5,
    "legend.fontsize": 15,
    "legend.title_fontsize": 15,
}


def setup_style():
    sns.set_theme(rc=STYLE_RC)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["computer modern roman"],
    })


def n_concepts_for_method(ce: str, n_concepts: list[int]) -> list[int]:
    """Return the cnr values whose metrics/run_info.json should be summed for this method.

    AC-style CENDRE methods only have cnr_0; ECLAD/MultiVISION/CENDREkMeans have
    one metrics run per n_concepts value. Summing over the returned list gives
    the "fair" metrics-compute time to compare against the AC variants (which
    score all internally-discovered concepts in a single metrics run).
    """
    if ce in CENDRE_USES_N_CONCEPTS:
        return list(n_concepts)
    if ce.startswith(CENDRE_PREFIXES):
        return [0]
    return list(n_concepts)


def read_runtime(run_info_path: Path) -> float | None:
    """Return total_time_seconds from a run_info.json, or None if missing/bad."""
    if not run_info_path.is_file():
        return None
    try:
        with open(run_info_path, "r") as f:
            info = json.load(f)
        return float(info.get("total_time_seconds"))
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"[WARN] Could not read {run_info_path}: {e}")
        return None


def load_runtimes(cfg) -> pd.DataFrame:
    """Walk metrics/run_info.json files and aggregate per (dataset, model, method, seed).

    Returns a long-format DataFrame:
      [Dataset, Model, CE Method, Seed, TotalTimeSeconds, NCnrs, NCnrsExpected]

    For each entity, TotalTimeSeconds is the SUM across the n_concepts values
    that apply to that method (see n_concepts_for_method). The runtime read
    here is the metrics-computation time only — task30 loads an already-fit CE
    when run_info.json exists at the CE level, so this excludes CE fit cost.
    """
    n_seeds = cfg.seeds
    models = cfg.models
    methods = cfg.methods
    n_concepts = cfg.n_concepts
    outnr = cfg.outnr
    dataset_names = list(cfg.datasets) if hasattr(cfg, "datasets") else [cfg.dataset.name]
    ce_base = getattr(cfg, "ce_base_path", "./data/ConceptExtraction")

    rows = []
    for dataset_name in dataset_names:
        for model in models:
            folder_structure = (
                f"{dataset_name}_{model}_{cfg.transforms.name}"
                f"_{cfg.loader.name}_{cfg.training.name}"
            )
            for ce in methods:
                cnrs = n_concepts_for_method(ce, n_concepts)
                cex = CE_DISPLAY_NAMES.get(ce, ce)
                save_dir = Path(f"{ce_base}/{ce}")
                for seed in n_seeds:
                    folder = (
                        save_dir
                        / f"{folder_structure}__{seed}_{cfg.suffix}"
                        / f"out{outnr}"
                    )
                    total = 0.0
                    found = 0
                    for n_concept in cnrs:
                        rt = read_runtime(
                            folder / f"cnr_{n_concept}" / "metrics" / "run_info.json"
                        )
                        if rt is not None:
                            total += rt
                            found += 1
                    if found == 0:
                        print(
                            f"[WARN] No metrics/run_info.json found under {folder} "
                            f"for {ce} (expected cnrs={cnrs}). Skipping entity."
                        )
                        continue
                    if found < len(cnrs):
                        print(
                            f"[WARN] Partial coverage for {ce} {dataset_name} "
                            f"{model} seed={seed}: {found}/{len(cnrs)} cnrs found."
                        )
                    rows.append({
                        "Dataset": dataset_name,
                        "Model": model,
                        "CE Method": cex,
                        "Seed": seed,
                        "TotalTimeSeconds": total,
                        "NCnrs": found,
                        "NCnrsExpected": len(cnrs),
                    })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / count of total metrics runtime per CE method, across (dataset, model, seed)."""
    if df.empty:
        return df
    summary = (
        df.groupby("CE Method", as_index=False)["TotalTimeSeconds"]
        .agg(
            MeanTimeSeconds="mean",
            StdTimeSeconds="std",
            MedianTimeSeconds="median",
            MinTimeSeconds="min",
            MaxTimeSeconds="max",
            N="count",
        )
        .sort_values("MeanTimeSeconds")
        .reset_index(drop=True)
    )
    return summary


def compute_palette(n_methods: int):
    c = 0.4
    colors = (1.0 - c) * plt.get_cmap("viridis")(
        np.linspace(0.0, 1.0, n_methods)
    ) + c * np.ones((n_methods, 4))
    return sns.color_palette(colors, n_methods)


def plot_runtime_bar(summary: pd.DataFrame, df: pd.DataFrame, cfg, out_dir: str):
    """Bar of mean metrics runtime per method with std error bars; overlaid strip of raw entities."""
    if summary.empty:
        print("[WARN] No metrics runtime data to plot.")
        return
    save_pdf = cfg.get("save_pdf", True)
    methods_sorted = list(summary["CE Method"])
    palette = dict(zip(methods_sorted, compute_palette(len(methods_sorted))))

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(methods_sorted)), 6),
                           constrained_layout=True)
    ax.bar(
        summary["CE Method"],
        summary["MeanTimeSeconds"],
        yerr=summary["StdTimeSeconds"].fillna(0.0),
        color=[palette[m] for m in methods_sorted],
        capsize=6,
        edgecolor="black",
    )
    sns.stripplot(
        data=df, x="CE Method", y="TotalTimeSeconds",
        order=methods_sorted, color="black", size=4, alpha=0.5, jitter=0.2, ax=ax,
    )
    ax.set_ylabel("Metrics runtime [s] (sum over n\\_concepts)")
    ax.set_xlabel("")
    ax.set_title(cfg.exp_name)
    ax.tick_params(axis="x", rotation=30)
    sns.despine(ax=ax)

    stem = f"{out_dir}/metrics_runtime_{cfg.exp_name}_out{cfg.outnr}"
    fig.savefig(f"{stem}.png", bbox_inches="tight")
    if save_pdf:
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


@hydra.main(config_path="../data/config", config_name="CE_plots", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_style()
    ce_base = getattr(cfg, "ce_base_path", "./data/ConceptExtraction")
    out_dir = f"{ce_base}/comparison"
    os.makedirs(out_dir, exist_ok=True)

    df = load_runtimes(cfg)
    if df.empty:
        print("[WARN] No metrics runtime data loaded. Exiting.")
        return

    df.to_csv(f"{out_dir}/metrics_runtime_entities_{cfg.exp_name}.csv", index=False)

    summary = summarize(df)
    summary.to_csv(f"{out_dir}/metrics_runtime_summary_{cfg.exp_name}.csv", index=False)
    print("\nMetrics runtime summary (seconds, summed over n_concepts):")
    print(summary.to_string(index=False))

    plot_runtime_bar(summary, df, cfg, out_dir)


if __name__ == "__main__":
    main()
