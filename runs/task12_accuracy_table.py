import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import json
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

import modules.utils.hydraqol  # noqa: F401  (registers the OmegaConf resolvers)

DASH_PLAIN = "—"
PM_PLAIN = "±"
MEAN_ROW_LABEL = "Mean"


def read_accuracy(results_path: Path, checkpoint: str, metric: str) -> float | None:
    """Return training_results.json[checkpoint][metric], or None if missing/bad."""
    if not results_path.is_file():
        return None
    try:
        with open(results_path, "r") as f:
            results = json.load(f)
        return float(results[checkpoint][metric])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"[WARN] Could not read {results_path}: {e}")
        return None


def load_accuracies(cfg) -> pd.DataFrame:
    """Walk training_results.json files, one per (dataset, model, seed).

    Returns a long-format DataFrame: [Dataset, Model, Seed, Value].
    """
    models_base = Path(cfg.path.base_path_models)
    rows = []
    for dataset in cfg.datasets:
        for model in cfg.models:
            for seed in cfg.seeds:
                folder = (
                    f"{dataset}_{model}_{cfg.transforms.name}"
                    f"_{cfg.loader.name}_{cfg.training.name}__{seed}_{cfg.suffix}"
                )
                value = read_accuracy(
                    models_base / folder / "training_results.json", cfg.checkpoint, cfg.metric
                )
                if value is None:
                    print(f"[WARN] No {cfg.checkpoint}/{cfg.metric} for {folder}. Skipping entity.")
                    continue
                rows.append({
                    "Dataset": dataset,
                    "Model": model,
                    "Seed": seed,
                    "Value": value,
                })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / count over seeds per (Dataset, Model)."""
    return df.groupby(["Dataset", "Model"], as_index=False)["Value"].agg(
        Mean="mean", Std="std", N="count",
    )


def build_grid(stats: pd.DataFrame, datasets: list[str], models: list[str]) -> list[dict]:
    """One entry per dataset row, holding a {model: (mean, std, n)} dict, plus a final
    summary row. The summary averages over complete-case datasets only (those with a
    value for every model), so the columns stay comparable; its label says so whenever
    some dataset had to be dropped."""
    lookup = {
        (r["Dataset"], r["Model"]): (r["Mean"], r["Std"], r["N"])
        for _, r in stats.iterrows()
    }

    grid = []
    for dataset in datasets:
        cells = {model: lookup.get((dataset, model), (np.nan, np.nan, 0)) for model in models}
        grid.append({"Dataset": dataset, "cells": cells})

    complete = [
        row for row in grid
        if all(not pd.isna(row["cells"][model][0]) for model in models)
    ]
    mean_cells = {
        model: (
            np.mean([row["cells"][model][0] for row in complete]) if complete else np.nan,
            np.nan,
            len(complete),
        )
        for model in models
    }
    label = MEAN_ROW_LABEL
    if len(complete) < len(datasets):
        label = f"{MEAN_ROW_LABEL} ({len(complete)}/{len(datasets)} datasets)"
    grid.append({"Dataset": label, "cells": mean_cells, "is_summary": True})
    return grid


def _fmt_value(mean, std, n, decimals: int, latex: bool) -> str:
    """Plain mean for a single seed, mean +- std once several seeds are aggregated."""
    if pd.isna(mean):
        return "--" if latex else DASH_PLAIN
    if n is None or n < 2 or pd.isna(std):
        return f"{mean:.{decimals}f}"
    pm = r"$\pm$" if latex else PM_PLAIN
    return f"{mean:.{decimals}f} {pm} {std:.{decimals}f}"


def format_csv(grid: list[dict], models: list[str], decimals: int) -> pd.DataFrame:
    records = []
    for row in grid:
        rec = {"Dataset": row["Dataset"]}
        for model in models:
            mean, std, n = row["cells"][model]
            rec[model] = _fmt_value(mean, std, n, decimals, latex=False)
        records.append(rec)
    return pd.DataFrame(records)


def escape_latex(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def build_latex_table(grid: list[dict], models: list[str], decimals: int,
                       checkpoint: str, metric: str, label_suffix: str = "") -> str:
    """Booktabs table, datasets as rows and models as columns, best model per dataset
    bolded (ties keep the first model in config order)."""
    label = f"tab:model_accuracy{('_' + label_suffix) if label_suffix else ''}"
    caption = (
        f"Test accuracy ({escape_latex(metric)}, {escape_latex(checkpoint)} checkpoint) "
        "of the trained models per dataset."
    )

    lines = [
        r"% Requires \usepackage{booktabs} in the preamble.",
        r"\begin{table}[h]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\centering",
        f"\\begin{{tabular}}{{@{{}}l{'c' * len(models)}@{{}}}}",
        r"\toprule",
        " & ".join(["Dataset"] + [escape_latex(m) for m in models]) + r" \\",
        r"\midrule",
    ]

    for row in grid:
        candidates = [(m, row["cells"][m][0]) for m in models]
        candidates = [(m, v) for m, v in candidates if not pd.isna(v)]
        best_model = max(candidates, key=lambda mv: mv[1])[0] if candidates else None

        if row.get("is_summary"):
            lines.append(r"\midrule")

        row_label = escape_latex(row["Dataset"])
        if row.get("is_summary"):
            row_label = f"\\textit{{{row_label}}}"
        cells = [row_label]
        for model in models:
            mean, std, n = row["cells"][model]
            cell = _fmt_value(mean, std, n, decimals, latex=True)
            if model == best_model:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


@hydra.main(config_path="../data/config", config_name="model_accuracy_table", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    datasets = list(cfg.datasets)
    models = list(cfg.models)

    df = load_accuracies(cfg)
    if df.empty:
        print("[WARN] No accuracies loaded. Exiting.")
        return
    df.to_csv(f"{out_dir}/accuracy_entities_{cfg.exp_name}.csv", index=False)

    grid = build_grid(summarize(df), datasets, models)

    table_csv = format_csv(grid, models, cfg.decimals)
    csv_path = f"{out_dir}/table_accuracy_{cfg.exp_name}.csv"
    table_csv.to_csv(csv_path, index=False)

    tex_path = f"{out_dir}/table_accuracy_{cfg.exp_name}.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(build_latex_table(grid, models, cfg.decimals, cfg.checkpoint, cfg.metric,
                                   label_suffix=cfg.exp_name))

    print(f"[OK] Accuracy table -> {csv_path} / {tex_path}")
    print(table_csv.to_string(index=False))


if __name__ == "__main__":
    main()
