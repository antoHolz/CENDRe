import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import shutil
import os
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from modules.utils.hydraqol import run_decorator


# CENDRE variants that determine n_concepts internally (use [0] for file lookup).
# Exception: CENDREkMeans uses n_concepts directly as cluster count.
CENDRE_PREFIXES = ("CENDRE",)
CENDRE_USES_N_CONCEPTS = ("CENDREkMeans",)

# Display name overrides for CE methods
CE_DISPLAY_NAMES = {
    "ECLADts": "ECLAD-ts",
}

# Centralized style dict (applied once, not repeated per plot)
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
    """Apply matplotlib/seaborn style settings once at startup.
    Falls back to mathtext if no LaTeX binary is on PATH (e.g. cluster image)."""
    use_tex = shutil.which("latex") is not None
    rc = dict(STYLE_RC)
    rc["text.usetex"] = use_tex
    sns.set_theme(rc=rc)
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["computer modern roman"],
    })


def get_metric_bases(cfg) -> list[str]:
    """Return metric base names to plot. Default: soft variants only (sIC, sRC).
    Set cfg.soft_only=False to also include hard variants (IC, RC)."""
    soft_only = cfg.get("soft_only", True)
    if soft_only:
        return ["sIC", "sRC"]
    return ["IC", "sIC", "RC", "sRC"]


def compute_palette(n_methods: int):
    """Compute a light-tinted viridis palette for the given number of CE methods."""
    c = 0.4
    colors = (1.0 - c) * plt.get_cmap("viridis")(
        np.linspace(0.0, 1.0, n_methods)
    ) + c * np.ones((n_methods, 4))
    return sns.color_palette(colors, n_methods)


def load_all_data(cfg) -> pd.DataFrame:
    """
    Load metrics CSVs across all (dataset, model, method, n_concept, seed)
    combinations into a single long-format DataFrame.

    Columns: [Dataset, Model, CE Method, Seed, Metric, Value, n_concepts]
    """
    n_seeds = cfg.seeds
    models = cfg.models
    methods = cfg.methods
    n_concepts = cfg.n_concepts
    outnr = cfg.outnr
    dataset_names = list(cfg.datasets) if hasattr(cfg, "datasets") else [cfg.dataset.name]
    eval_metrics = list(cfg.eval_metrics) if hasattr(cfg, "eval_metrics") else ["dst_eclad"]

    rows = []
    for dataset_name in dataset_names:
        for model in models:
            folder_structure = (
                f"{dataset_name}_{model}_{cfg.transforms.name}"
                f"_{cfg.loader.name}_{cfg.training.name}"
            )
            for ce in methods:
                # CENDRE variants determine n_concepts internally,
                # except CENDREkMeans which uses n_concepts directly.
                if ce in CENDRE_USES_N_CONCEPTS:
                    n_concepts_CE = n_concepts
                elif ce.startswith(CENDRE_PREFIXES):
                    n_concepts_CE = [0]
                else:
                    n_concepts_CE = n_concepts
                cex = CE_DISPLAY_NAMES.get(ce, ce)

                ce_base = getattr(cfg, "ce_base_path", "./data/ConceptExtraction")
                for n_concept in n_concepts_CE:
                    save_dir = Path(f"{ce_base}/{ce}")
                    for seed in n_seeds:
                        folder = (
                            f"{save_dir}/{folder_structure}__{seed}"
                            f"_{cfg.suffix}/out{outnr}"
                        )
                        csv_path = f"{folder}/metrics_{n_concept}.csv"
                        try:
                            metric_df = pd.read_csv(csv_path)
                            for em in eval_metrics:
                                for col_base in ["sRC", "RC", "IC", "ICA", "ICU", "sIC", "sICA", "sICU"]:
                                    col_name = f"{col_base}_{em}"
                                    if col_name in metric_df.columns:
                                        rows.append({
                                            "Dataset": dataset_name,
                                            "Model": model,
                                            "CE Method": cex,
                                            "Seed": seed,
                                            "Metric": col_name,
                                            "Value": metric_df[col_name].iloc[0],
                                            "n_concepts": n_concept,
                                        })
                        except FileNotFoundError:
                            print(f"File not found: {csv_path}")
                            continue

    return pd.DataFrame(rows)


def apply_k_mode(df: pd.DataFrame, k_mode: str) -> pd.DataFrame:
    """
    Aggregate across n_concepts (k) values per (Dataset, Model, CE Method, Seed, Metric).

    k_mode:
        "all"  -- keep all rows as-is (each k is a separate data point)
        "mean" -- average Value over k per group
        "best" -- max Value over k per group
                  (max is correct for both IC and RC regardless of higher_is_better)
    """
    if k_mode == "all":
        return df

    group_cols = ["Dataset", "Model", "CE Method", "Seed", "Metric"]
    if k_mode == "mean":
        return df.groupby(group_cols, as_index=False)["Value"].mean()
    elif k_mode == "best":
        # Pick the k with best RC, then keep all metrics at that k.
        # RC metrics start with "RC_"; find the best k per entity using RC.
        entity_cols = ["Dataset", "Model", "CE Method", "Seed"]
        rc_df = df.loc[df["Metric"].str.startswith("RC_")]
        if rc_df.empty:
            # Fallback: no RC metrics, just take max per metric independently
            return df.groupby(group_cols, as_index=False)["Value"].max()
        # For each entity + n_concepts, get mean RC across all RC metrics
        best_k = (
            rc_df.groupby(entity_cols + ["n_concepts"], as_index=False)["Value"]
            .mean()
            .sort_values("Value", ascending=False)
            .drop_duplicates(subset=entity_cols, keep="first")
            [entity_cols + ["n_concepts"]]
        )
        # Keep only rows matching the best k per entity
        return df.merge(best_k, on=entity_cols + ["n_concepts"])
    else:
        raise ValueError(f"Invalid k_mode '{k_mode}'. Must be one of: all, mean, best")


def plot_single_boxplot(ax, data, palette):
    """Draw a single boxplot on the given axes with x=Model, hue=CE Method."""
    sns.boxplot(
        data=data, x="Model", y="Value", hue="CE Method",
        palette=palette, ax=ax,
    )
    sns.despine(ax=ax)


def save_figure(fig, path_stem: str, save_pdf: bool):
    """Save figure as PNG always, and as PDF if save_pdf is True."""
    fig.savefig(f"{path_stem}.png", bbox_inches="tight")
    if save_pdf:
        fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_summary_boxplots(df: pd.DataFrame, cfg, palette, out_dir: str):
    """
    Produce M separate figures (one per IC/RC x eval_metric), aggregated
    across all datasets. Auto-scaled y-axis, no hardcoded limits.
    """
    eval_metrics = list(cfg.eval_metrics) if hasattr(cfg, "eval_metrics") else ["dst_eclad"]
    save_pdf = cfg.get("save_pdf", True)
    outnr = cfg.outnr
    metric_names = [f"{base}_{em}" for em in eval_metrics for base in get_metric_bases(cfg)]

    for m_name in metric_names:
        plot_df = df.loc[df["Metric"] == m_name].dropna(subset=["Value", "Model", "CE Method"])
        if plot_df.empty:
            print(f"[WARN] No data for metric={m_name} in summary. Skipping.")
            continue

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        plot_single_boxplot(ax, plot_df, palette)
        ax.set_ylabel(m_name)
        ax.legend(ncol=2)
        ax.set_title(cfg.exp_name)

        path_stem = f"{out_dir}/boxplotFULL_{cfg.exp_name}_out{outnr}_{m_name}"
        save_figure(fig, path_stem, save_pdf)


def plot_grid_figure(df: pd.DataFrame, cfg, palette, out_dir: str):
    """
    One big figure with M rows x DS columns.
    M = 2 * len(eval_metrics) (IC and RC per eval_metric).
    DS = number of datasets.
    Each cell: boxplot with x=Model, hue=CE Method.
    """
    eval_metrics = list(cfg.eval_metrics) if hasattr(cfg, "eval_metrics") else ["dst_eclad"]
    save_pdf = cfg.get("save_pdf", True)
    outnr = cfg.outnr
    datasets = sorted(df["Dataset"].unique())
    metric_names = [f"{base}_{em}" for em in eval_metrics for base in get_metric_bases(cfg)]

    n_rows = len(metric_names)
    n_cols = len(datasets)
    if n_rows == 0 or n_cols == 0:
        print("[WARN] No data for grid figure. Skipping.")
        return

    cell_w = max(3.0, 12.0 / n_cols * 2)
    cell_h = 2.5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols, cell_h * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    legend_handles = None
    legend_labels = None

    for row_idx, m_name in enumerate(metric_names):
        for col_idx, ds_name in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            cell_df = df.loc[
                (df["Metric"] == m_name) & (df["Dataset"] == ds_name)
            ].dropna(subset=["Value", "Model", "CE Method"])

            if not cell_df.empty:
                plot_single_boxplot(ax, cell_df, palette)
                if legend_handles is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)

            # Remove per-cell legends
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

            # Column headers on top row only
            if row_idx == 0:
                ax.set_title(ds_name, fontsize=12)
            else:
                ax.set_title("")

            # Row labels on left column only
            if col_idx == 0:
                ax.set_ylabel(m_name, fontsize=11)
            else:
                ax.set_ylabel("")

            # x-tick labels on bottom row only
            if row_idx < n_rows - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=45)

    # Shared legend at bottom
    if legend_handles is not None:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            ncol=min(len(legend_labels), 5),
            bbox_to_anchor=(0.5, -0.02),
            fontsize=10,
        )

    path_stem = f"{out_dir}/grid_{cfg.exp_name}_out{outnr}"
    save_figure(fig, path_stem, save_pdf)


def plot_summary_grid(df: pd.DataFrame, cfg, palette, out_dir: str):
    """
    One big figure with rows = eval_metrics, columns = [IC, RC].
    Each cell: boxplot (mean over datasets) with x=Model, hue=CE Method.
    """
    eval_metrics = list(cfg.eval_metrics) if hasattr(cfg, "eval_metrics") else ["dst_eclad"]
    save_pdf = cfg.get("save_pdf", False)
    outnr = cfg.outnr
    bases = get_metric_bases(cfg)

    n_rows = len(eval_metrics)
    n_cols = len(bases)
    if n_rows == 0:
        print("[WARN] No eval_metrics for summary grid. Skipping.")
        return

    cell_w = max(5.0, 12.0 / n_cols * 2)
    cell_h = 3.5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols, cell_h * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    legend_handles = None
    legend_labels = None

    for row_idx, em in enumerate(eval_metrics):
        for col_idx, base in enumerate(bases):
            ax = axes[row_idx, col_idx]
            m_name = f"{base}_{em}"
            cell_df = df.loc[df["Metric"] == m_name].dropna(subset=["Value", "Model", "CE Method"])

            if not cell_df.empty:
                # Mean over datasets per (Model, CE Method, Seed)
                mean_df = cell_df.groupby(
                    ["Model", "CE Method", "Seed"], as_index=False, observed=False
                )["Value"].mean()
                plot_single_boxplot(ax, mean_df, palette)
                if legend_handles is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)

            # Remove per-cell legends
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

            # Column headers on top row only
            if row_idx == 0:
                ax.set_title(base, fontsize=14)
            else:
                ax.set_title("")

            # Row labels on left column only
            if col_idx == 0:
                ax.set_ylabel(em, fontsize=12)
            else:
                ax.set_ylabel("")

            # x-tick labels on bottom row only
            if row_idx < n_rows - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=45)

    # Shared legend at bottom
    if legend_handles is not None:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            ncol=min(len(legend_labels), 5),
            bbox_to_anchor=(0.5, -0.02),
            fontsize=10,
        )

    path_stem = f"{out_dir}/summary_grid_{cfg.exp_name}_out{outnr}"
    save_figure(fig, path_stem, save_pdf)


@hydra.main(config_path="../data/config", config_name="CE_plots", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # --- Setup ---
    setup_style()
    ce_base = getattr(cfg, "ce_base_path", "./data/ConceptExtraction")
    out_dir = f"{ce_base}/comparison"
    os.makedirs(out_dir, exist_ok=True)

    k_mode = cfg.get("k_mode", "all")

    # --- Load all data ---
    df = load_all_data(cfg)
    if df.empty:
        print("[WARN] No data loaded. Exiting.")
        return

    # --- Save combined CSV (before k-mode aggregation, for full detail) ---
    df.to_csv(f"{out_dir}/dfboxplot_{cfg.exp_name}.csv", index=False)

    # --- Apply k-mode aggregation ---
    df = apply_k_mode(df, k_mode)

    # --- Enforce alphabetical CE Method order for consistent plotting ---
    ce_methods_sorted = sorted(df["CE Method"].unique())
    df["CE Method"] = pd.Categorical(df["CE Method"], categories=ce_methods_sorted, ordered=True)
    palette = dict(zip(ce_methods_sorted, compute_palette(len(ce_methods_sorted))))

    # --- Summary boxplots (aggregated across all datasets) ---
    plot_summary_boxplots(df, cfg, palette, out_dir)

    # --- Grid figure (M rows x DS columns) ---
    plot_grid_figure(df, cfg, palette, out_dir)

    # --- Summary grid (eval_metrics rows x IC/RC columns, mean over datasets) ---
    plot_summary_grid(df, cfg, palette, out_dir)


if __name__ == "__main__":
    main()
