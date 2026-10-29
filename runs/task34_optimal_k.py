import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import glob
import math
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


# Mirrors task31 classification: CENDRE variants determine n_concepts internally,
# except CENDREkMeans which uses n_concepts as the fixed cluster count.
CENDRE_PREFIXES = ("CENDRE",)
CENDRE_USES_N_CONCEPTS = ("CENDREkMeans",)


def is_auto_k(method: str) -> bool:
    return method.startswith(CENDRE_PREFIXES) and method not in CENDRE_USES_N_CONCEPTS


def instance_folder(method, dataset, model, seed, cfg) -> str:
    return (
        f"./data/ConceptExtraction/{method}/"
        f"{dataset}_{model}_{cfg.transforms.name}_{cfg.loader.name}_{cfg.training.name}"
        f"__{seed}_{cfg.suffix}/out{cfg.outnr}"
    )


def optimal_k(method, dataset, model, seed, cfg):
    """Return the optimal k for one (method, dataset, model, seed) instance.

    Fixed-k methods: argmax of RC_{selection_metric} over cfg.n_concepts.
    Auto-k methods: the actual concept count the method produced,
    read from the shape of scores_{selection_metric}_0.npy.
    Returns None if no usable data is found.
    """
    folder = instance_folder(method, dataset, model, seed, cfg)
    rc_col = f"RC_{cfg.selection_metric}"

    if is_auto_k(method):
        scores_files = sorted(glob.glob(f"{folder}/scores_{cfg.selection_metric}_*.npy"))
        if not scores_files:
            return None
        return int(np.load(scores_files[0]).shape[1])

    best_k, best_rc = None, -math.inf
    for k in cfg.n_concepts:
        csv_path = f"{folder}/metrics_{k}.csv"
        if not Path(csv_path).exists():
            continue
        df = pd.read_csv(csv_path)
        if rc_col not in df.columns:
            continue
        rc = float(df[rc_col].iloc[0])
        if rc > best_rc:
            best_rc, best_k = rc, k
    return best_k


def gather_rows(cfg) -> pd.DataFrame:
    """Walk the full (dataset x model x method x seed) grid and collect optimal k."""
    datasets = list(cfg.datasets) if hasattr(cfg, "datasets") else [cfg.dataset.name]
    rows = []
    for dataset in datasets:
        for model in cfg.models:
            for method in cfg.methods:
                for seed in cfg.seeds:
                    k = optimal_k(method, dataset, model, seed, cfg)
                    if k is None:
                        print(f"[SKIP] {method} / {dataset} / {model} / seed={seed}")
                        continue
                    rows.append({
                        "Dataset": dataset,
                        "Model": model,
                        "CE Method": method,
                        "Seed": seed,
                        "optimal_k": k,
                        "selection_metric": cfg.selection_metric,
                    })
    return pd.DataFrame(rows)


def save_per_model_tables(df: pd.DataFrame, cfg, out_dir: str):
    """One CSV per model: Dataset, CE Method, Seed, optimal_k, selection_metric."""
    for model in cfg.models:
        model_df = df.loc[df["Model"] == model, [
            "Dataset", "CE Method", "Seed", "optimal_k", "selection_metric",
        ]].reset_index(drop=True)
        path = f"{out_dir}/optimal_k_{cfg.exp_name}_{model}.csv"
        model_df.to_csv(path, index=False)
        print(f"[OK] {path}  ({len(model_df)} rows)")


def save_agreement_matrix(df: pd.DataFrame, cfg, out_dir: str):
    """Symmetric method x method matrix of mean |k_A - k_B| across instances."""
    methods = list(cfg.methods)
    if df.empty:
        print("[WARN] No data for agreement matrix.")
        return

    # Pivot to (instance -> method -> k). Instance = (Dataset, Model, Seed).
    wide = df.pivot_table(
        index=["Dataset", "Model", "Seed"],
        columns="CE Method",
        values="optimal_k",
        aggfunc="first",
    )
    # Ensure all requested methods appear as columns even if absent from data
    for m in methods:
        if m not in wide.columns:
            wide[m] = np.nan
    wide = wide[methods]

    mat = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            if a == b:
                mat.loc[a, b] = 0.0
            else:
                mat.loc[a, b] = (wide[a] - wide[b]).abs().mean(skipna=True)

    path = f"{out_dir}/agreement_matrix_{cfg.exp_name}.csv"
    mat.to_csv(path)
    print(f"[OK] {path}")


@hydra.main(config_path="../data/config", config_name="CE_kselect", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = "./data/ConceptExtraction/comparison"
    os.makedirs(out_dir, exist_ok=True)

    df = gather_rows(cfg)
    if df.empty:
        print("[WARN] No (method, dataset, model, seed) combo produced a valid k. Exiting.")
        return

    save_per_model_tables(df, cfg, out_dir)
    save_agreement_matrix(df, cfg, out_dir)


if __name__ == "__main__":
    main()
