import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import glob
import json
import os

import hydra
import pandas as pd
from omegaconf import DictConfig

from task36_sensitivity_plot import load_sensitivity_data

# clusterMode -> initial used by CENDRE.naming() in the fit-file suffix (<initial>_<n_clusters>)
METHOD_TAGS = {"ACsilhouette": "A", "HDBSCAN": "H"}


def parse_instance(folder_name: str, model_names) -> tuple:
    """'{dataset}_{model}_{transforms}_{loader}_{training}__{seed}_{suffix}' ->
    (dataset, model, seed); the dataset may contain underscores, so the split anchors on the
    model segment. Returns None when no known model name occurs in the folder name."""
    left, right = folder_name.split("__")
    seed = int(right.split("_")[0])
    for model in model_names:
        marker = f"_{model}_"
        if marker in left:
            return left.split(marker)[0], model, seed
    return None


@hydra.main(config_path="../data/config", config_name="CE_sensitivity_plot", version_base="1.3")
def main(cfg: DictConfig) -> None:
    df = load_sensitivity_data(cfg)
    if df.empty:
        print("[WARN] No data loaded. Exiting.")
        return

    # 1) One K row per completed run (seed x n_clusters level); 0 concepts is a real value
    k = df[df["Metric"] == "K"].copy()
    k["reached"] = k["Value"] >= cfg.k_target

    # 2) One table per method, mirroring the response plots: rows = datasets, columns = the
    #    swept J levels, cells = models reaching the target / models run at that level
    out_dir = f"{cfg.ce_base_path}/comparison"
    os.makedirs(out_dir, exist_ok=True)
    for method in cfg.methods:
        sub = k[k["CE Method"] == method]
        reached = sub.pivot_table(index="Dataset", columns="n_clusters",
                                  values="reached", aggfunc="sum")
        total = sub.pivot_table(index="Dataset", columns="n_clusters",
                                values="reached", aggfunc="size")
        table = (reached.reindex(list(cfg.datasets)).fillna(0).astype(int).astype(str)
                 + "/" + total.reindex(list(cfg.datasets)).fillna(0).astype(int).astype(str))
        path = f"{out_dir}/kcount_{cfg.exp_name}_{method}.csv"
        table.to_csv(path)
        print(f"\n{method}: models with K >= {cfg.k_target} per n_clusters level (reached/total):")
        print(table.to_string())
        print(f"[OK] {path}")

    ##############################
    # 3) Normal J=n_clusters_ref runs: every fit found at the out<outnr> level (no nc segment),
    #    across all datasets, models and seeds on disk, pooled per dataset
    ##############################
    model_names = [p.stem for p in
                   (Path(__file__).resolve().parents[1] / "data/config/model").glob("*.yaml")]
    rows = []
    for method, tag in METHOD_TAGS.items():
        pattern = (f"{cfg.ce_base_path}/CENDRE{method}/*/out{cfg.outnr}/"
                   f"importances_{tag}_{cfg.n_clusters_ref}.json")
        for imp in sorted(glob.glob(pattern)):
            instance = Path(imp).parents[1].name
            parsed = parse_instance(instance, model_names)
            if parsed is None:
                print(f"[WARN] cannot parse instance folder name: {instance}")
                continue
            dataset, model, seed = parsed
            with open(imp) as handle:
                rows.append({"Dataset": dataset, "Model": model, "Seed": seed,
                             "CE Method": method, "reached": len(json.load(handle)) >= cfg.k_target})
    normal = pd.DataFrame(rows)
    if normal.empty:
        print(f"[WARN] no normal J={cfg.n_clusters_ref} fits found under {cfg.ce_base_path}.")
        return
    reached = normal.pivot_table(index="Dataset", columns="CE Method",
                                 values="reached", aggfunc="sum")
    total = normal.pivot_table(index="Dataset", columns="CE Method",
                               values="reached", aggfunc="size")
    methods = [m for m in METHOD_TAGS if m in reached.columns]
    summary = (reached[methods].fillna(0).astype(int).astype(str)
               + "/" + total[methods].fillna(0).astype(int).astype(str))
    summary.loc["Total"] = [f"{int(reached[m].sum())}/{int(total[m].sum())}" for m in methods]
    path = f"{out_dir}/kcount_normal_J{cfg.n_clusters_ref}.csv"
    summary.to_csv(path)
    print(f"\nNormal J={cfg.n_clusters_ref} runs with K >= {cfg.k_target}, "
          f"pooled over models and seeds (reached/total):")
    print(summary.to_string())
    print(f"[OK] {path}")


if __name__ == "__main__":
    main()
