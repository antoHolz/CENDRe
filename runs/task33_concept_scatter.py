import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import json
import glob
import hydra
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from cel.utils.correctness import RC, METRICS

from task32_alignment_tables import (
    compute_max_penalties,
    get_input_len,
    load_importances_raw,
    importances_scalar,
)
from task31_plot_metrics import (
    CENDRE_PREFIXES,
    CENDRE_USES_N_CONCEPTS,
    CE_DISPLAY_NAMES,
    STYLE_RC,
    setup_style,
    compute_palette,
    save_figure,
)


# Marker per model for the scatter plot
MODEL_MARKERS = {
    "InceptionTime10": "X",
    "ResNet1D18": "o",
    "DenseNet1D121": "D",
}
DEFAULT_MARKER = "s"


def load_concept_data(cfg, scatter_metric: str) -> pd.DataFrame:
    """
    Load per-concept (similarity, importance) data across all
    (dataset, model, method, seed) combinations for a given metric.

    For methods with multiple n_concepts, pick the best k by RC.
    Returns a DataFrame with columns:
        [Dataset, Model, CE Method, Seed, concept_id, similarity, importance]
    """
    n_seeds = cfg.seeds
    models = cfg.models
    methods = cfg.methods
    n_concepts_list = cfg.n_concepts
    outnr = cfg.outnr
    dataset_names = list(cfg.datasets) if hasattr(cfg, "datasets") else [cfg.dataset.name]

    # Compute max penalties
    input_len = get_input_len(cfg)
    max_penalties = compute_max_penalties(input_len, [scatter_metric])
    max_penalty = max_penalties[scatter_metric]
    higher_is_better = METRICS[scatter_metric]["higher_is_better"]

    rows = []
    for dataset_name in dataset_names:
        for model in models:
            folder_structure = (
                f"{dataset_name}_{model}_{cfg.transforms.name}"
                f"_{cfg.loader.name}_{cfg.training.name}"
            )
            for ce in methods:
                cex = CE_DISPLAY_NAMES.get(ce, ce)

                # Resolve n_concepts candidates
                if ce in CENDRE_USES_N_CONCEPTS:
                    n_concepts_CE = n_concepts_list
                elif ce.startswith(CENDRE_PREFIXES):
                    n_concepts_CE = [0]
                else:
                    n_concepts_CE = n_concepts_list

                for seed in n_seeds:
                    base_dir = (
                        f"./data/ConceptExtraction/{ce}"
                        f"/{folder_structure}__{seed}"
                        f"_{cfg.suffix}/out{outnr}"
                    )

                    # Find best k by RC across candidates
                    best_k = None
                    best_rc = -1.0
                    for k in n_concepts_CE:
                        scores_path = f"{base_dir}/scores_{scatter_metric}_{k}.npy"
                        if not Path(scores_path).exists():
                            continue
                        scores = torch.tensor(
                            np.load(scores_path), dtype=torch.float32
                        )
                        rc, _, _ = RC(
                            scores, mode="rel", max_penalty=max_penalty,
                            higher_is_better=higher_is_better,
                        )
                        if rc > best_rc:
                            best_rc = rc
                            best_k = k

                    if best_k is None:
                        print(f"[SKIP] No scores for {ce}/{folder_structure}__{seed}")
                        continue

                    # Load scores and importances at best k
                    scores_path = f"{base_dir}/scores_{scatter_metric}_{best_k}.npy"
                    scores = torch.tensor(
                        np.load(scores_path), dtype=torch.float32
                    )

                    # Compute similarity: (n_primitives, n_concepts, n_channels)
                    if higher_is_better:
                        sim = scores.clamp(0.0, max_penalty) / max(max_penalty, 1e-12)
                    else:
                        sim = 1.0 - scores.clamp(0.0, max_penalty) / max(max_penalty, 1e-12)

                    # Best similarity per concept: max over primitives, then max over channels
                    best_sim = sim.amax(dim=0).amax(dim=-1)  # (n_concepts,)

                    # Load importances
                    raw_imp = load_importances_raw(base_dir, k_str=str(best_k))
                    if not raw_imp:
                        print(f"[SKIP] No importances for {ce}/{folder_structure}__{seed}")
                        continue
                    imp_dict = importances_scalar(raw_imp)

                    n_concepts = best_sim.shape[0]
                    for c in range(n_concepts):
                        rows.append({
                            "Dataset": dataset_name,
                            "Model": model,
                            "CE Method": cex,
                            "Seed": seed,
                            "concept_id": c,
                            "similarity": float(best_sim[c]),
                            "importance": float(imp_dict.get(c, 0.0)),
                        })

    return pd.DataFrame(rows)


def plot_scatter(df: pd.DataFrame, cfg, palette, out_dir: str, scatter_metric: str, title_suffix: str = ""):
    """
    Scatter plot: x = best similarity, y = importance.
    Color = CE Method, marker = Model.
    """
    save_pdf = cfg.get("save_pdf", False)
    outnr = cfg.outnr

    methods_sorted = sorted(df["CE Method"].unique())
    models_sorted = sorted(df["Model"].unique())

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    # Plot each (method, model) combo separately for correct color + marker
    for ce in methods_sorted:
        for model in models_sorted:
            subset = df.loc[
                (df["CE Method"] == ce) & (df["Model"] == model)
            ]
            if subset.empty:
                continue
            ax.scatter(
                subset["similarity"],
                subset["importance"],
                c=[palette[ce]],
                marker=MODEL_MARKERS.get(model, DEFAULT_MARKER),
                s=60,
                alpha=0.7,
                edgecolors="none",
            )

    # Build two separate legends
    # 1. Color legend (methods)
    method_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[m],
                   markersize=10, label=m)
        for m in methods_sorted
    ]
    # 2. Marker legend (models)
    model_handles = [
        plt.Line2D([0], [0], marker=MODEL_MARKERS.get(m, DEFAULT_MARKER),
                   color="gray", markerfacecolor="gray", markersize=10,
                   linestyle="None", label=m)
        for m in models_sorted
    ]

    all_handles = method_handles + model_handles
    all_labels = [h.get_label() for h in all_handles]
    fig.legend(
        all_handles, all_labels,
        loc="lower center",
        ncol=min(len(all_handles), 8),
        bbox_to_anchor=(0.5, -0.05),
        framealpha=0.9,
    )

    ax.set_xlabel("Best Similarity")
    ax.set_ylabel("Importance")
    title = cfg.exp_name
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    sns.despine(ax=ax)

    suffix_str = f"_{title_suffix}" if title_suffix else ""
    path_stem = f"{out_dir}/scatter_{cfg.exp_name}_out{outnr}_{scatter_metric}{suffix_str}"
    save_figure(fig, path_stem, save_pdf)


@hydra.main(config_path="../data/config", config_name="CE_plots", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_style()
    out_dir = "./data/ConceptExtraction/comparison"
    os.makedirs(out_dir, exist_ok=True)

    scatter_mode = cfg.get("scatter_mode", "all")
    scatter_metrics = list(cfg.get("scatter_metric", ["rma"]))
    if isinstance(scatter_metrics, str):
        scatter_metrics = [scatter_metrics]

    for scatter_metric in scatter_metrics:
        # Load concept-level data for this metric
        df = load_concept_data(cfg, scatter_metric)
        if df.empty:
            print(f"[WARN] No data loaded for {scatter_metric}. Skipping.")
            continue

        # Save raw data
        df.to_csv(f"{out_dir}/scatter_data_{cfg.exp_name}_{scatter_metric}.csv", index=False)

        # Build palette (consistent with task31)
        ce_methods_sorted = sorted(df["CE Method"].unique())
        palette = dict(zip(ce_methods_sorted, compute_palette(len(ce_methods_sorted))))

        if scatter_mode == "per_method":
            for ce in sorted(df["CE Method"].unique()):
                ce_df = df.loc[df["CE Method"] == ce]
                plot_scatter(ce_df, cfg, palette, out_dir, scatter_metric, title_suffix=ce)
        elif scatter_mode == "per_dataset":
            for ds_name in sorted(df["Dataset"].unique()):
                ds_df = df.loc[df["Dataset"] == ds_name]
                plot_scatter(ds_df, cfg, palette, out_dir, scatter_metric, title_suffix=ds_name)
        else:
            plot_scatter(df, cfg, palette, out_dir, scatter_metric)


if __name__ == "__main__":
    main()
