import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import hydra
from omegaconf import DictConfig

from task31_plot_metrics import apply_k_mode, CE_DISPLAY_NAMES

DASH_PLAIN = "—"
PM_PLAIN = "±"


def load_family_df(ce_base_path: str, exp_name: str, family_label: str, k_mode: str) -> pd.DataFrame:
    """Load a task31-cached dfboxplot CSV, apply the same best-k selection used for the
    boxplots (on the full metric set, before any eval_metric filtering), and tag Family."""
    csv_path = f"{ce_base_path}/comparison/dfboxplot_{exp_name}.csv"
    df = pd.read_csv(csv_path)
    df = apply_k_mode(df, k_mode)
    df["Family"] = family_label
    return df


def compute_row_order(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(Family, Dataset) pairs in first-seen order, i.e. family list order then each
    family's original dataset-config order (both preserved by the concatenation in main())."""
    pairs = df[["Family", "Dataset"]].drop_duplicates()
    return list(pairs.itertuples(index=False, name=None))


def compute_stats(df: pd.DataFrame, methods_display: list[str], eval_metric: str,
                   metric_bases: list[str], confidence: float,
                   group_cols: list[str]) -> pd.DataFrame:
    """Mean/Std/N/CIHalf per group_cols + [CE Method, MetricBase], for sRC/sIC @ eval_metric.
    CIHalf uses a 95%-style Student-t interval: t.ppf(1-alpha/2, n-1) * Std/sqrt(N), NaN if N<2."""
    metric_map = {f"{base}_{eval_metric}": base for base in metric_bases}
    sub = df.loc[df["Metric"].isin(metric_map)].copy()
    sub["MetricBase"] = sub["Metric"].map(metric_map)
    sub = sub.loc[sub["CE Method"].isin(methods_display)]

    grouped = sub.groupby(group_cols + ["CE Method", "MetricBase"], as_index=False)["Value"].agg(
        Mean="mean", Std="std", N="count",
    )

    alpha = 1.0 - confidence

    def ci_half(row):
        n, std = row["N"], row["Std"]
        if n < 2 or pd.isna(std):
            return np.nan
        tcrit = scipy_stats.t.ppf(1 - alpha / 2, n - 1)
        return tcrit * std / np.sqrt(n)

    grouped["CIHalf"] = grouped.apply(ci_half, axis=1)
    return grouped


def build_cell_grid(stats_df: pd.DataFrame, row_order: list[tuple[str, str]],
                     methods_display: list[str], metric_bases: list[str]) -> list[dict]:
    """One entry per (Family, Dataset) row, holding a {(method, base): (mean, std, ci_half)}
    dict. stats_df should already be filtered to a single Model for per-model tables."""
    lookup = {}
    for _, r in stats_df.iterrows():
        lookup[(r["Family"], r["Dataset"], r["CE Method"], r["MetricBase"])] = (
            r["Mean"], r["Std"], r["CIHalf"],
        )

    grid = []
    for family, dataset in row_order:
        cells = {}
        for method in methods_display:
            for base in metric_bases:
                cells[(method, base)] = lookup.get(
                    (family, dataset, method, base), (np.nan, np.nan, np.nan)
                )
        grid.append({"Family": family, "Dataset": dataset, "cells": cells})
    return grid


FAMILY_SUMMARY_LABEL = "Mean"


def append_family_summary_rows(grid: list[dict], family_stats: pd.DataFrame,
                                methods_display: list[str], metric_bases: list[str],
                                label: str = FAMILY_SUMMARY_LABEL) -> list[dict]:
    """Append one aggregate row per Family (pooling all its datasets' raw values, not an
    average of the per-dataset means) right after that family's last dataset row."""
    lookup = {}
    for _, r in family_stats.iterrows():
        lookup[(r["Family"], r["CE Method"], r["MetricBase"])] = (r["Mean"], r["Std"], r["CIHalf"])

    families_in_order = list(dict.fromkeys(row["Family"] for row in grid))

    new_grid = []
    for family in families_in_order:
        new_grid.extend(row for row in grid if row["Family"] == family)
        cells = {
            (method, base): lookup.get((family, method, base), (np.nan, np.nan, np.nan))
            for method in methods_display
            for base in metric_bases
        }
        new_grid.append({"Family": family, "Dataset": label, "cells": cells, "is_summary": True})
    return new_grid


def _fmt_mean_std(mean, std, decimals: int, latex: bool) -> str:
    if pd.isna(mean) or pd.isna(std):
        return "--" if latex else DASH_PLAIN
    pm = r"$\pm$" if latex else PM_PLAIN
    return f"{mean:.{decimals}f} {pm} {std:.{decimals}f}"


def _fmt_ci(mean, ci_half, decimals: int, latex: bool) -> str:
    if pd.isna(mean) or pd.isna(ci_half):
        return "--" if latex else DASH_PLAIN
    lo, hi = mean - ci_half, mean + ci_half
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"


def format_csv(grid: list[dict], methods_display: list[str], metric_bases: list[str],
               decimals: int) -> pd.DataFrame:
    """Flat wide table for CSV export: grouping (Family/Dataset) as plain columns,
    one mean+-std and one CI column per (method, metric base)."""
    records = []
    for row in grid:
        rec = {"Family": row["Family"], "Dataset": row["Dataset"]}
        for method in methods_display:
            for base in metric_bases:
                mean, std, ci_half = row["cells"][(method, base)]
                rec[f"{method}__{base}_mean_std"] = _fmt_mean_std(mean, std, decimals, latex=False)
                rec[f"{method}__{base}_ci95"] = _fmt_ci(mean, ci_half, decimals, latex=False)
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


def build_latex_table(grid: list[dict], methods_display: list[str], metric_base: str,
                       confidence: float, col_width_cm: float, data_col_width_cm: float,
                       decimals: int, label_suffix: str = "", caption_suffix: str = "") -> str:
    """Hand-built booktabs table for a single metric base (sRC or sIC): one table per
    call (mirrors this project's table style: \\scriptsize, \\setlength{\\tabcolsep}{3pt},
    @{}-trimmed column spec). All data columns share one fixed centered width so every
    method block is separated equally instead of auto-sizing to content. Method names
    span 2 cols each (mean+-std, CI); the Family (SyntheticL/SyntheticF) rows act as the
    section title for the row-label column, so that column gets no "Dataset" header text."""
    total_data_cols = len(methods_display) * 2
    total_cols = 1 + total_data_cols
    ci_pct = int(round(confidence * 100))
    label = f"tab:distributional_summary_{metric_base}{('_' + label_suffix) if label_suffix else ''}"
    caption = (
        r"Distributional summary (mean $\pm$ std, "
        f"{ci_pct}\\% CI) of {metric_base} across CE methods and datasets"
        f"{caption_suffix}."
    )

    lines = [
        r"% Requires \usepackage{booktabs,graphicx} in the preamble (plain p{} columns",
        r"% only -- intentionally avoids the array package's >{...} column modifiers,",
        r"% which produce garbled output if array isn't loaded).",
        r"\begin{table}[h]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
    ]
    data_col = f"p{{{data_col_width_cm}cm}}"
    col_spec = (
        f"@{{}}p{{{col_width_cm}cm}}|"
        + data_col * total_data_cols
        + "@{}"
    )
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Row 1: method names spanning 2 columns each.
    row1 = [""] + [
        f"\\multicolumn{{2}}{{c}}{{{escape_latex(method)}}}" for method in methods_display
    ]
    lines.append(" & ".join(row1) + r" \\")
    col = 2
    cmid1 = []
    for _ in methods_display:
        cmid1.append(f"\\cmidrule(lr){{{col}-{col + 1}}}")
        col += 2
    lines.append("".join(cmid1))

    # Row 2: "mean +- std" / "CI" sub-column labels. First cell left blank -- the
    # Family rows below act as the row-label column's title, not the word "Dataset".
    row2 = [""]
    for method in methods_display:
        row2.append(r"mean $\pm$ std")
        row2.append(f"{ci_pct}\\% CI")
    lines.append(" & ".join(row2) + r" \\")
    lines.append(r"\midrule")

    current_family = None
    for row in grid:
        if row["Family"] != current_family:
            if current_family is not None:
                lines.append(r"\midrule")
            lines.append(
                f"\\multicolumn{{{total_cols}}}{{@{{}}l}}{{\\textbf{{{escape_latex(row['Family'])}}}}} \\\\"
            )
            current_family = row["Family"]

        if row.get("is_summary"):
            lines.append(r"\midrule")

        # Best (highest-mean) method for this metric base, bolded below -- ties keep
        # the first method encountered in methods_display order.
        candidates = [
            (method, row["cells"][(method, metric_base)][0]) for method in methods_display
        ]
        candidates = [(m, v) for m, v in candidates if not pd.isna(v)]
        best_method = max(candidates, key=lambda mv: mv[1])[0] if candidates else None

        row_label = escape_latex(row["Dataset"])
        if row.get("is_summary"):
            row_label = f"\\textit{{{row_label}}}"
        cells = [row_label]
        for method in methods_display:
            mean, std, ci_half = row["cells"][(method, metric_base)]
            mean_std_str = _fmt_mean_std(mean, std, decimals, latex=True)
            ci_str = _fmt_ci(mean, ci_half, decimals, latex=True)
            if method == best_method:
                mean_std_str = f"\\textbf{{{mean_std_str}}}"
                ci_str = f"\\textbf{{{ci_str}}}"
            cells.append(mean_std_str)
            cells.append(ci_str)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


@hydra.main(config_path="../data/config", config_name="CE_distributional_summary_table", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ce_base_path = cfg.ce_base_path
    out_dir = f"{ce_base_path}/comparison"
    os.makedirs(out_dir, exist_ok=True)

    methods_display = [CE_DISPLAY_NAMES.get(m, m) for m in cfg.methods]
    metric_bases = list(cfg.metric_bases)
    eval_metric = cfg.eval_metric
    confidence = cfg.confidence
    decimals = cfg.decimals
    exp_name = cfg.exp_name

    family_dfs = [
        load_family_df(ce_base_path, family_exp_name, family_label, cfg.k_mode)
        for family_label, family_exp_name in cfg.families.items()
    ]
    df = pd.concat(family_dfs, ignore_index=True)
    row_order = compute_row_order(df)

    # --- Pooled table: mean/std/CI over all (Model, Seed) combinations per cell ---
    pooled_stats = compute_stats(
        df, methods_display, eval_metric, metric_bases, confidence, group_cols=["Family", "Dataset"]
    )
    pooled_grid = build_cell_grid(pooled_stats, row_order, methods_display, metric_bases)
    # Family summary row: pools every dataset's raw (Model, Seed) values within the family,
    # not an average of the per-dataset means.
    pooled_family_stats = compute_stats(
        df, methods_display, eval_metric, metric_bases, confidence, group_cols=["Family"]
    )
    pooled_grid = append_family_summary_rows(pooled_grid, pooled_family_stats, methods_display, metric_bases)

    for base in metric_bases:
        base_csv = format_csv(pooled_grid, methods_display, [base], decimals)
        csv_path = f"{out_dir}/table_{exp_name}_{eval_metric}_{base}.csv"
        base_csv.to_csv(csv_path, index=False)
        tex_path = f"{out_dir}/table_{exp_name}_{eval_metric}_{base}.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(build_latex_table(pooled_grid, methods_display, base, confidence,
                                       cfg.dataset_col_width_cm, cfg.data_col_width_cm, decimals))
        print(f"[OK] Pooled {base} table -> {csv_path} / {tex_path}")
        print(base_csv.to_string(index=False))

    # --- Per-model tables: mean/std/CI over Seed only, one table per Model ---
    if cfg.get("per_model", True):
        model_stats_all = compute_stats(
            df, methods_display, eval_metric, metric_bases, confidence,
            group_cols=["Model", "Family", "Dataset"],
        )
        model_family_stats_all = compute_stats(
            df, methods_display, eval_metric, metric_bases, confidence,
            group_cols=["Model", "Family"],
        )
        for model in sorted(df["Model"].unique()):
            model_stats = model_stats_all.loc[model_stats_all["Model"] == model]
            model_grid = build_cell_grid(model_stats, row_order, methods_display, metric_bases)
            model_family_stats = model_family_stats_all.loc[model_family_stats_all["Model"] == model]
            model_grid = append_family_summary_rows(
                model_grid, model_family_stats, methods_display, metric_bases
            )

            for base in metric_bases:
                base_csv = format_csv(model_grid, methods_display, [base], decimals)
                m_csv_path = f"{out_dir}/table_{exp_name}_{eval_metric}_{base}_{model}.csv"
                base_csv.to_csv(m_csv_path, index=False)
                m_tex_path = f"{out_dir}/table_{exp_name}_{eval_metric}_{base}_{model}.tex"
                with open(m_tex_path, "w", encoding="utf-8") as f:
                    f.write(build_latex_table(model_grid, methods_display, base, confidence,
                                               cfg.dataset_col_width_cm, cfg.data_col_width_cm, decimals,
                                               label_suffix=model, caption_suffix=f" ({model})"))
                print(f"[OK] Per-model table ({model}, {base}) -> {m_csv_path} / {m_tex_path}")


if __name__ == "__main__":
    main()
