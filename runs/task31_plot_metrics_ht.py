"""Summary boxplots with pairwise significance brackets.

Same figures as task31, plus the task38 test (paired Wilcoxon signed-rank,
Holm-corrected) drawn as brackets between the boxes of one x group. Uses the
CE_plots config, so the usual `+experiment=...` overrides apply unchanged.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import matplotlib.pyplot as plt

from task31_plot_metrics import (
    apply_k_mode,
    compute_palette,
    get_metric_bases,
    load_all_data,
    plot_single_boxplot,
    save_figure,
    setup_style,
)
from task38_pairwise_significance_tests import pairwise_tests


SIG_LEVELS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))


def sig_label(p: float, alpha: float) -> str:
    if pd.isna(p) or p >= alpha:
        return "ns"
    for threshold, stars in SIG_LEVELS:
        if p < threshold:
            return stars
    return "*"


def keep_pair(p: float, alpha: float, mode: str) -> bool:
    """sig_mode: 'significant' brackets only pairs below alpha, 'nonsignificant'
    only those above (useful when nearly everything is significant), 'all' both."""
    if pd.isna(p):
        return False
    if mode == "significant":
        return p < alpha
    if mode == "nonsignificant":
        return p >= alpha
    return True


def stack_brackets(pairs: list[tuple[float, float, str]]) -> list[tuple[float, float, str, int]]:
    """Assign each (x1, x2, label) bracket the lowest level on which it does not
    overlap an already-placed bracket. Short spans go first so they end up
    underneath the wider ones."""
    placed = []
    levels: list[list[tuple[float, float]]] = []
    for x1, x2, label in sorted(pairs, key=lambda p: p[1] - p[0]):
        level = 0
        while level < len(levels) and any(x1 <= b2 and x2 >= b1 for b1, b2 in levels[level]):
            level += 1
        if level == len(levels):
            levels.append([])
        levels[level].append((x1, x2))
        placed.append((x1, x2, label, level))
    return placed


def annotate_pairwise_significance(ax, data, block_cols, alpha=0.05, mode="significant",
                                   fontsize=13, x_col="Model", hue_col="CE Method",
                                   box_width=0.8, line_width=1.0, verbose=True):
    """Draw significance brackets between the hue-dodged boxes of each x group.

    `block_cols` defines the matched blocks handed to the paired test; Holm
    correction runs over the pairs within one x group, matching the scope of a
    single task38 matrix. `mode` selects which pairs get a bracket.
    """
    x_levels = [t.get_text() for t in ax.get_xticklabels()]
    _, hue_levels = ax.get_legend_handles_labels()
    if not x_levels or len(hue_levels) < 2:
        return

    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo
    tip = 0.015 * span
    step = 0.075 * span
    top = y_hi

    for xi, x_level in enumerate(x_levels):
        sub = data.loc[data[x_col] == x_level]
        if sub.empty:
            continue
        wide = sub.pivot_table(index=block_cols, columns=hue_col, values="Value", observed=True)
        methods = [m for m in hue_levels if m in wide.columns and wide[m].notna().any()]
        if len(methods) < 2:
            continue

        pair_info, _ = pairwise_tests(wide[methods], methods)
        brackets = []
        for info in pair_info:
            p = info["p_holm"]
            if not keep_pair(p, alpha, mode):
                continue
            mi, mj = methods[info["i"]], methods[info["j"]]
            offsets = [
                xi + (hue_levels.index(m) - (len(hue_levels) - 1) / 2)
                * box_width / len(hue_levels)
                for m in (mi, mj)
            ]
            brackets.append((min(offsets), max(offsets), sig_label(p, alpha)))
            if verbose:
                print(f"      {x_level}: {mi} vs {mj}  p_holm={p:.4g} "
                      f"r={info['r']:+.2f} n={info['n']}")

        base = sub["Value"].max() + step * 0.4
        for x1, x2, label, level in stack_brackets(brackets):
            y = base + level * step
            ax.plot([x1, x1, x2, x2], [y, y + tip, y + tip, y],
                    lw=line_width, c="0.2", clip_on=False)
            ax.text((x1 + x2) / 2, y + tip, label, ha="center", va="bottom",
                    fontsize=fontsize, color="0.2")
            top = max(top, y + tip + 0.03 * span)

    ax.set_ylim(y_lo, top)


def load_df(cfg, out_dir: str) -> pd.DataFrame:
    """Rescan the per-run metric CSVs, or reuse the combined CSV that task31
    already wrote for this exp_name when cfg.from_cached_csv is set."""
    if cfg.get("from_cached_csv", False):
        csv_path = f"{out_dir}/dfboxplot_{cfg.exp_name}.csv"
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded cached {csv_path} ({len(df)} rows).")
        return df
    return load_all_data(cfg)


def plot_annotated_boxplots(df: pd.DataFrame, cfg, palette, out_dir: str):
    """One figure per metric, aggregated over datasets, with significance
    brackets per Model group over (Dataset, Seed) blocks."""
    eval_metrics = list(cfg.eval_metrics) if hasattr(cfg, "eval_metrics") else ["dst_eclad"]
    save_pdf = cfg.get("save_pdf", True)
    outnr = cfg.outnr
    alpha = cfg.get("sig_alpha", 0.05)
    mode = cfg.get("sig_mode", "significant")
    fontsize = cfg.get("sig_fontsize", 13)
    metric_names = [f"{base}_{em}" for em in eval_metrics for base in get_metric_bases(cfg)]

    for m_name in metric_names:
        plot_df = df.loc[df["Metric"] == m_name].dropna(subset=["Value", "Model", "CE Method"])
        if plot_df.empty:
            print(f"[WARN] No data for metric={m_name}. Skipping.")
            continue

        print(f"[INFO] {m_name}: annotating {mode} pairs at Holm alpha={alpha}")
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        plot_single_boxplot(ax, plot_df, palette)
        annotate_pairwise_significance(ax, plot_df, ["Dataset", "Seed"],
                                       alpha=alpha, mode=mode, fontsize=fontsize)
        ax.set_ylabel(m_name)
        ax.legend(ncol=2)
        ax.set_title(cfg.exp_name)

        suffix = {"nonsignificant": "_ns", "all": "_allp"}.get(mode, "")
        path_stem = f"{out_dir}/boxplotHT_{cfg.exp_name}_out{outnr}_{m_name}{suffix}"
        save_figure(fig, path_stem, save_pdf)
        print(f"[OK] {path_stem}.png")


@hydra.main(config_path="../data/config", config_name="CE_plots", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_style()
    ce_base = getattr(cfg, "ce_base_path", "./data/ConceptExtraction")
    out_dir = f"{ce_base}/comparison"
    os.makedirs(out_dir, exist_ok=True)

    k_mode = cfg.get("k_mode", "all")
    if k_mode == "all":
        print("[INFO] k_mode=all: each matched block is a (Dataset, Seed) pair whose Value "
              "is averaged over n_concepts, while the boxes show every k separately. "
              "Use k_mode=best to match the task38 tables.")

    df = load_df(cfg, out_dir)
    if df.empty:
        print("[WARN] No data loaded. Exiting.")
        return

    df = apply_k_mode(df, k_mode)

    ce_methods_sorted = sorted(df["CE Method"].unique())
    df["CE Method"] = pd.Categorical(df["CE Method"], categories=ce_methods_sorted, ordered=True)
    palette = dict(zip(ce_methods_sorted, compute_palette(len(ce_methods_sorted))))

    plot_annotated_boxplots(df, cfg, palette, out_dir)


if __name__ == "__main__":
    main()
