import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import glob
import json
import os
import shutil
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from task38_pairwise_significance_tests import holm_correction


CELL_KEYS = ["Dataset", "Family", "Model", "CE Method", "Metric"]

# Table column key -> (LaTeX header, row formatter, field tested for NaN, field averaged in the
# trailing mean row or None when the column does not average). Selected via cfg.table_columns.
COLUMN_SPECS = {
    "S1_J": (r"$S_1^J$", lambda r: f"{r['S1_J']:.3f}", "S1_J", "S1_J"),
    "S1_J_ci": (r"$95\%$ CI", lambda r: f"[{r['S1_J_lo']:.3f}, {r['S1_J_hi']:.3f}]", "S1_J_lo", None),
    "S1_seed": (r"$S_1^{\mathrm{seed}}$", lambda r: f"{r['S1_seed']:.3f}", "S1_seed", "S1_seed"),
    "eta2_J": (r"$\eta^2_J$", lambda r: f"{r['eta2_J']:.3f}", "eta2_J", "eta2_J"),
    "mu_star": (r"$\mu^*$", lambda r: f"{r['mu_star']:.4f}", "mu_star", "mu_star"),
    "sigma_EE": (r"$\sigma$", lambda r: f"{r['sigma_EE']:.4f}", "sigma_EE", "sigma_EE"),
    "mu": (r"$\mu_{EE}$", lambda r: f"{r['mu']:+.4f}", "mu", "mu"),
    "mu_ci": (r"$95\%$ CI", lambda r: f"[{r['mu_lo']:+.4f}, {r['mu_hi']:+.4f}]", "mu_lo", None),
    "range_J": (r"range", lambda r: f"{r['range_J']:.3f}", "range_J", "range_J"),
    "sigma_J": (r"$\sigma_J$", lambda r: f"{r['sigma_J']:.3f}", "sigma_J", "sigma_J"),
    "p_holm": (r"$p$", lambda r: f"{r['p_holm']:.3f}", "p_holm", None),
    "W": (r"$W$", lambda r: f"{r['W']:.3f}", "W", "W"),
    "equiv_seed": (r"eq$_\sigma$", lambda r: "Y" if r["equiv_seed"] > 0.5 else "N", "equiv_seed", None),
    "equiv_fixed": (r"eq$_\delta$", lambda r: "Y" if r["equiv_fixed"] > 0.5 else "N", "equiv_fixed", None),
    "n_seeds": (r"$n$", lambda r: f"{int(r['n_seeds'])}", "n_seeds", None),
    "n_levels": (r"$k$", lambda r: f"{int(r['n_levels'])}", "n_levels", None),
}


def report_metric_labels(cfg) -> dict:
    """cfg.report_metrics -> {metric column: display label}, dropping the eval-metric suffix.

    A list rather than a mapping in the config, so an experiment bundle replaces the default
    outright instead of merging into it.
    """
    return {name: name.replace(f"_{cfg.primary_eval_metric}", "") for name in cfg.report_metrics}


def setup_style():
    """Match the plot aesthetics used in examples/task20_concept_extraction_ts.ipynb.
    Falls back to mathtext if no LaTeX binary is on PATH (e.g. cluster image)."""
    use_tex = shutil.which("latex") is not None
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["computer modern roman"],
        "font.size": 20,
    })


def save_figure(fig, path_stem: str, save_pdf: bool):
    fig.savefig(f"{path_stem}.png", bbox_inches="tight")
    if save_pdf:
        fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_sensitivity_data(cfg) -> pd.DataFrame:
    """
    Load metrics CSVs across (dataset, model, clusterMode, seed, n_clusters), long-format.

    n_clusters is carried by an nc<value> segment appended to CE.save_dir (see the
    sensitivity_nclusters_* experiment configs), not by top-level `suffix` -- suffix also feeds
    save_dir_model (the trained classifier checkpoint path), which has no such segment. The
    auto-selected n_concepts baked into the metrics filename is unknown ahead of time and is
    discovered via glob, the same approach task34_optimal_k.py uses for auto-k CE methods.

    Values stay as read, NaN included: a NaN marks a run whose extractor returned 0 concepts,
    which build_cell_matrices drops explicitly rather than silently.

    Columns: [Dataset, Family, Model, CE Method, Seed, n_clusters, Metric, Value]
    """
    eval_metrics = list(cfg.eval_metrics)
    ce_base = cfg.ce_base_path
    outnr = cfg.outnr

    rows = []
    for dataset in cfg.datasets:
        family = cfg.dataset_family.get(dataset, dataset)
        for model in cfg.models:
            folder_structure = f"{dataset}_{model}_{cfg.transforms.name}_{cfg.loader.name}_{cfg.training.name}"
            for method in cfg.methods:
                for seed in cfg.seeds:
                    for nc in cfg.n_clusters:
                        folder = f"{ce_base}/CENDRE{method}/{folder_structure}__{seed}_/out{outnr}/nc{nc}"
                        if not os.path.isdir(folder):
                            print(f"[WARN] expected folder does not exist: {folder}")
                            continue
                        # 1) K is the concept count the auto-selection settled on, one entry
                        #    per concept in the importances file; 0 concepts is a real value
                        importances = sorted(glob.glob(f"{folder}/importances_*.json"))
                        if importances:
                            with open(importances[0]) as handle:
                                rows.append({
                                    "Dataset": dataset, "Family": family, "Model": model,
                                    "CE Method": method, "Seed": seed, "n_clusters": nc,
                                    "Metric": "K", "Value": float(len(json.load(handle))),
                                })
                        matches = sorted(glob.glob(f"{folder}/metrics_*.csv"))
                        if not matches:
                            print(f"[WARN] folder exists but has no metrics file: {folder}")
                            continue
                        if len(matches) > 1:
                            print(f"[WARN] {len(matches)} metrics files in {folder}, using {matches[0]}")
                        metric_df = pd.read_csv(matches[0])
                        for em in eval_metrics:
                            for base in ["sRC", "sIC", "RC", "IC"]:
                                col = f"{base}_{em}"
                                if col in metric_df.columns:
                                    rows.append({
                                        "Dataset": dataset,
                                        "Family": family,
                                        "Model": model,
                                        "CE Method": method,
                                        "Seed": seed,
                                        "n_clusters": nc,
                                        "Metric": col,
                                        "Value": metric_df[col].iloc[0],
                                    })
    return pd.DataFrame(rows)


def build_cell_matrices(df: pd.DataFrame):
    """Long frame -> {cell: (Y, seeds, levels)} with Y[seed, n_clusters] complete-case.

    The blocked statistics (variance decomposition, Friedman, TOST) need a gap-free seed x
    level matrix, so within each cell only the n_clusters levels that are valid for every
    seed are kept. Failed runs are the 0-concept ones whose metrics were written as NaN;
    their failure rate is reported elsewhere, so here they are only logged. The curves do
    not go through this restriction, see build_point_frame.

    Returns (cells, dropped) where dropped lists one record per discarded level.
    """
    cells, dropped = {}, []
    for key_vals, group in df.groupby(CELL_KEYS, sort=False):
        pivot = group.pivot_table(index="Seed", columns="n_clusters", values="Value")
        n_seeds_total = pivot.shape[0]
        complete = pivot.dropna(axis=1, how="any")
        for level in pivot.columns.difference(complete.columns):
            dropped.append({
                **dict(zip(CELL_KEYS, key_vals)),
                "n_clusters": int(level),
                "n_valid_seeds": int(pivot[level].notna().sum()),
                "n_seeds": int(n_seeds_total),
            })
        if complete.shape[0] < 3 or complete.shape[1] < 3:
            print(f"[WARN] cell {key_vals} has only {complete.shape[0]} seeds x "
                  f"{complete.shape[1]} levels after complete-case filtering, skipped.")
            continue
        cells[key_vals] = (complete.to_numpy(dtype=float),
                           complete.index.to_numpy(dtype=int),
                           complete.columns.to_numpy(dtype=float))
    return cells, pd.DataFrame(dropped)


def log_voronoi_weights(levels: np.ndarray, weighting: str) -> np.ndarray:
    """Level weights of the log-uniform input measure on n_clusters, summing to 1.

    The swept ladder is geometric only above 73 (50 was inserted as the paper's reference
    value), so equal weights would over-represent the crowded low end. Each level takes the
    half-interval to its neighbours in log space, with half-width ends.
    """
    if weighting == "uniform":
        return np.full(len(levels), 1.0 / len(levels))
    log_levels = np.log(levels)
    edges = np.concatenate(([log_levels[0]],
                            0.5 * (log_levels[1:] + log_levels[:-1]),
                            [log_levels[-1]]))
    widths = np.diff(edges)
    return widths / widths.sum()


def variance_decomposition(Y: np.ndarray, weights: np.ndarray) -> dict:
    """First-order variance shares of n_clusters and of the seed, under the measure
    (uniform over seeds) x (weights over levels).

    Y[s, j] is a randomised complete block layout (seed blocks, n_clusters treatments), so
    the components are the functional-ANOVA terms of the product measure:

        V_J    = sum_j w_j (a_j - m)^2          a_j = seed-mean at level j
        V_seed = sum_s p_s (b_s - m)^2          b_s = weighted level-mean of seed s
        V_res  = V_tot - V_J - V_seed

    eta2 = V_x / V_tot is biased upward by residual noise (with 10 levels and 11 seeds the
    bias is of the order of the effects being measured), so omega2 subtracts the expected
    noise contribution and is the number reported as the Sobol first-order index S1.
    """
    n_seeds, n_levels = Y.shape
    # A perfectly constant response (K pinned at one value) has nothing to apportion, so the
    # shares are zero rather than 0/0; range and sigma below report the constancy directly
    if np.ptp(Y) == 0:
        return {"eta2_J": 0.0, "eta2_seed": 0.0, "S1_J": 0.0, "S1_seed": 0.0,
                "sigma_J": 0.0, "range_J": 0.0, "sigma_resid": 0.0}
    p = np.full(n_seeds, 1.0 / n_seeds)
    level_means = Y.mean(axis=0)                       # (n_levels,)
    seed_means = Y @ weights                           # (n_seeds,)
    grand = float(weights @ level_means)

    v_total = float(p @ ((Y - grand) ** 2) @ weights)
    v_j = float(weights @ (level_means - grand) ** 2)
    v_seed = float(p @ (seed_means - grand) ** 2)
    v_res = max(v_total - v_j - v_seed, 0.0)

    sum_w2 = float((weights ** 2).sum())
    sigma2 = v_res / ((1.0 - 1.0 / n_seeds) * (1.0 - sum_w2))
    bias_j = (sigma2 / n_seeds) * (1.0 - sum_w2)
    bias_seed = sigma2 * sum_w2 * (1.0 - 1.0 / n_seeds)
    denom = v_total + sigma2 / (n_seeds * n_levels)

    return {
        "eta2_J": v_j / v_total if v_total > 0 else np.nan,
        "eta2_seed": v_seed / v_total if v_total > 0 else np.nan,
        "S1_J": max(v_j - bias_j, 0.0) / denom if denom > 0 else np.nan,
        "S1_seed": max(v_seed - bias_seed, 0.0) / denom if denom > 0 else np.nan,
        "sigma_J": np.sqrt(v_j),
        "range_J": float(level_means.max() - level_means.min()),
        "sigma_resid": np.sqrt(sigma2),
    }


def elementary_effects(Y: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Morris elementary effects, one per (seed, step), in metric units per doubling of n_clusters.

    The factor is taken as log2(n_clusters) rather than n_clusters: the swept grid is geometric,
    so a step is one fixed fraction of a doubling everywhere, whereas in raw units the low-end
    steps span a hundredth of the high-end ones and dominate the mean.

    Returns an array of shape (n_seeds, n_levels - 1).
    """
    return np.diff(Y, axis=1) / np.diff(np.log2(levels))[None, :]


def bootstrap_index_ci(Y: np.ndarray, weights: np.ndarray, levels: np.ndarray, n_boot: int,
                       rng: np.random.Generator, confidence: float) -> dict:
    """Percentile bootstrap intervals for the variance shares and the mean elementary effect.

    Seeds are the independent unit here: the levels of n_clusters are a fixed enumerated grid,
    not a sample, so only the blocks are resampled.
    """
    n_seeds = Y.shape[0]
    keys = ["S1_J", "S1_seed", "mu"]
    draws = {key: np.empty(n_boot) for key in keys}
    for b in range(n_boot):
        resampled = Y[rng.integers(0, n_seeds, size=n_seeds)]
        components = variance_decomposition(resampled, weights)
        draws["S1_J"][b] = components["S1_J"]
        draws["S1_seed"][b] = components["S1_seed"]
        draws["mu"][b] = elementary_effects(resampled, levels).mean()
    half = (1.0 - confidence) / 2.0
    out = {}
    for key in keys:
        out[f"{key}_lo"] = float(np.nanquantile(draws[key], half))
        out[f"{key}_hi"] = float(np.nanquantile(draws[key], 1.0 - half))
    return out


def friedman_kendall_w(Y: np.ndarray):
    """Friedman omnibus over n_clusters levels (seeds as blocks) plus Kendall's W."""
    n_seeds, n_levels = Y.shape
    chi2, p = scipy_stats.friedmanchisquare(*[Y[:, j] for j in range(n_levels)])
    return chi2, p, chi2 / (n_seeds * (n_levels - 1))


def tost_equivalence(Y: np.ndarray, ref_index: int, margin: float, alpha: float):
    """Two-one-sided-t equivalence of every level against the reference level.

    Paired differences d_s = Y[s, j] - Y[s, ref] are tested against (-margin, +margin); the
    cell is equivalent only if every level passes, which is an intersection-union test and
    therefore needs no multiplicity correction. Returns (equivalent, p_iut, max_abs_diff).
    """
    n_seeds = Y.shape[0]
    p_levels, mean_diffs = [], []
    for j in range(Y.shape[1]):
        if j == ref_index:
            continue
        d = Y[:, j] - Y[:, ref_index]
        mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n_seeds)
        mean_diffs.append(abs(mean))
        if se == 0:
            p_levels.append(0.0 if abs(mean) < margin else 1.0)
            continue
        p_lower = scipy_stats.t.sf((mean + margin) / se, n_seeds - 1)
        p_upper = scipy_stats.t.cdf((mean - margin) / se, n_seeds - 1)
        p_levels.append(max(p_lower, p_upper))
    p_iut = max(p_levels) if p_levels else np.nan
    return p_iut < alpha, p_iut, max(mean_diffs) if mean_diffs else np.nan


def compute_sensitivity_indices(cells, cfg, rng: np.random.Generator) -> pd.DataFrame:
    """One row per cell with the variance shares and their bootstrap intervals, plus the
    omnibus test and equivalence verdicts kept for reference. Friedman p-values are
    Holm-corrected within each metric."""
    rows = []
    for key_vals, (Y, _, levels) in cells.items():
        weights = log_voronoi_weights(levels, cfg.grid_weighting)
        record = {**dict(zip(CELL_KEYS, key_vals)),
                  "n_seeds": Y.shape[0], "n_levels": Y.shape[1]}
        record.update(variance_decomposition(Y, weights))

        # 1) Same decomposition on equal level weights, as a robustness column
        unweighted = variance_decomposition(Y, np.full(len(levels), 1.0 / len(levels)))
        record["S1_J_unweighted"] = unweighted["S1_J"]
        record["eta2_J_unweighted"] = unweighted["eta2_J"]

        # 2) Morris screening aggregates over the (seed, step) elementary effects
        ee = elementary_effects(Y, levels)
        record["mu"] = float(ee.mean())
        record["mu_star"] = float(np.abs(ee).mean())
        record["sigma_EE"] = float(ee.std(ddof=1))

        # 3) Intervals on the reported indices, so each is readable as a bounded quantity
        record.update(bootstrap_index_ci(Y, weights, levels, cfg.bootstrap_n,
                                         rng, cfg.confidence))

        chi2, p_raw, kendall_w = friedman_kendall_w(Y)
        record["chi2"] = chi2
        record["p_raw"] = p_raw
        record["W"] = kendall_w

        # 4) Both margins: the cell's own between-seed SD at the anchor level, and a fixed one.
        #    The anchor falls back to the nearest retained level when the reference was dropped
        ref_index = int(np.argmin(np.abs(np.log(levels) - np.log(cfg.n_clusters_ref))))
        if levels[ref_index] != float(cfg.n_clusters_ref):
            print(f"[INFO] n_clusters={cfg.n_clusters_ref} not retained for {key_vals}, "
                  f"anchoring equivalence at {int(levels[ref_index])}.")
        delta_seed = float(Y[:, ref_index].std(ddof=1))
        eq_seed, p_seed, max_diff = tost_equivalence(Y, ref_index, delta_seed, cfg.equivalence_alpha)
        eq_fixed, p_fixed, _ = tost_equivalence(Y, ref_index, cfg.equivalence_margin,
                                                cfg.equivalence_alpha)
        record.update({"ref_level": int(levels[ref_index]), "delta_seed": delta_seed,
                       "equiv_seed": float(eq_seed), "p_tost_seed": p_seed,
                       "equiv_fixed": float(eq_fixed), "p_tost_fixed": p_fixed,
                       "max_abs_diff": max_diff})
        rows.append(record)

    si = pd.DataFrame(rows)
    si["p_holm"] = np.nan
    for _, group in si.groupby("Metric"):
        si.loc[group.index, "p_holm"] = holm_correction(group["p_raw"].tolist())
    return si


def build_point_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(cell, seed, level) values, tidy, keeping every valid run.

    Unlike the blocked statistics, a curve does not need a gap-free matrix: a level whose runs
    failed for a few seeds still contributes the seeds that succeeded.
    """
    points = df.dropna(subset=["Value"]).copy()
    points["x"] = points["n_clusters"].astype(float)
    return points[CELL_KEYS + ["Seed", "x", "Value"]]


def summarize_curves(points: pd.DataFrame, value_col: str, group_col: str, band: str,
                     cfg) -> pd.DataFrame:
    """Mean and band per (group, CE Method, Metric, x); band is 'sd' or 't'.

    Blocks are (Dataset, Seed) pairs, so a family facet pools the seeds of its datasets. Each x
    is summarised over the blocks that have it, so a partially failed level keeps its surviving
    seeds; points backed by fewer than cfg.min_blocks_per_point blocks are dropped instead,
    since their band would be meaningless.
    """
    confidence = cfg.confidence
    rows = []
    for (group, method, metric), cell in points.groupby([group_col, "CE Method", "Metric"], sort=False):
        wide = cell.pivot_table(index=["Dataset", "Seed"], columns="x", values=value_col)
        for x in wide.columns:
            values = wide[x].dropna().to_numpy()
            n_blocks = len(values)
            if n_blocks < cfg.min_blocks_per_point:
                continue
            mean = float(values.mean())
            if band == "sd":
                half = values.std(ddof=1)
                lo, hi = mean - half, mean + half
            else:
                half = scipy_stats.t.ppf(1 - (1 - confidence) / 2, n_blocks - 1) * \
                    values.std(ddof=1) / np.sqrt(n_blocks)
                lo, hi = mean - half, mean + half
            rows.append({group_col: group, "CE Method": method, "Metric": metric,
                         "x": float(x), "y": mean, "lo": lo, "hi": hi, "n_blocks": n_blocks})
    return pd.DataFrame(rows)


def plot_curve_grid(summary: pd.DataFrame, cfg, facet_col: str, facets, ylabel_prefix: str,
                    path_stem: str, title_fontsize: int, palette: dict, share_y: str = "none"):
    """rows = metric base (sRC, sIC), cols = facets, hue = CE Method, x = n_clusters (log).

    ylabel_prefix is stamped in front of the metric label. share_y is 'all' (every panel on the
    widest range in the figure), 'row' (widest range within each metric) or 'none'.
    """
    metrics = report_metric_labels(cfg)

    # 1) Widest range any panel needs, so excursions are comparable at a glance across facets
    ylims = {}
    if share_y in ("all", "row"):
        plotted = summary[summary["Metric"].isin(metrics)]
        scopes = {"all": plotted} if share_y == "all" else \
            {name: plotted[plotted["Metric"] == name] for name in metrics}
        for key, scope in scopes.items():
            drawn = scope[scope[facet_col].isin(facets)]
            if drawn.empty:
                continue
            low, high = float(drawn["lo"].min()), float(drawn["hi"].max())
            pad = 0.05 * (high - low)
            ylims[key] = (low - pad, high + pad)

    fig, axes = plt.subplots(
        len(metrics), len(facets), figsize=(6 * len(facets), 4.5 * len(metrics)),
        squeeze=False, constrained_layout=True,
    )
    legend_handles, legend_labels = None, None
    for row, (name, label) in enumerate(metrics.items()):
        for col, facet in enumerate(facets):
            ax = axes[row][col]
            cell = summary[(summary["Metric"] == name) & (summary[facet_col] == facet)]
            if cell.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            else:
                for method, curve in cell.groupby("CE Method", sort=False):
                    curve = curve.sort_values("x")
                    color = palette[method]
                    ax.plot(curve["x"], curve["y"], marker="o", color=color, label=method)
                    ax.fill_between(curve["x"], curve["lo"], curve["hi"], color=color, alpha=0.20)
                ax.set_xscale("log")
                ax.axvline(cfg.n_clusters_ref, color="tab:gray", linestyle="--", linewidth=1)
                if legend_handles is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()
            limits = ylims.get("all" if share_y == "all" else name)
            if limits is not None:
                ax.set_ylim(*limits)
            ax.set_title(facet if row == 0 else "", fontsize=title_fontsize)
            ax.set_ylabel(f"{ylabel_prefix}{label}" if col == 0 else "")
            ax.set_xlabel("n\\_clusters" if row == len(metrics) - 1 else "")

    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels, title="CE Method",
                   loc="center left", bbox_to_anchor=(1.0, 0.5))

    save_figure(fig, path_stem, cfg.save_pdf)


def make_sensitivity_table(si: pd.DataFrame, cfg, model: str, out_dir: str):
    """Booktabs LaTeX table per metric base: one row per dataset, cfg.table_columns per
    clusterMode under a method overtitle, plus a trailing mean row across datasets.

    The mean row averages only the columns COLUMN_SPECS marks with a field name; intervals,
    counts, p-values and the boolean equivalence verdicts are left blank there.
    """
    columns = list(cfg.table_columns)
    methods = list(cfg.methods)
    datasets = list(cfg.datasets)

    for metric_name, label in report_metric_labels(cfg).items():
        sub = si[(si["Metric"] == metric_name) & (si["Model"] == model)]

        data_col = f"p{{{cfg.table_data_col_width_cm}cm}}"
        col_spec = (f"@{{}}p{{{cfg.table_col_width_cm}cm}}|"
                    + data_col * (len(columns) * len(methods)) + "@{}")
        lines = [
            r"% Requires \usepackage{booktabs,graphicx} in the preamble (plain p{} columns",
            r"% only -- intentionally avoids the array package's >{...} column modifiers,",
            r"% which produce garbled output if array isn't loaded).",
            r"\begin{table}[!tbp]",
            rf"\caption{{{cfg.table_caption.format(metric=label)}}}",
            rf"\label{{tab:sensitivity{metric_name.replace('_', '')}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\centering",
            r"\resizebox{\linewidth}{!}{%",
            rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
        ]

        header1 = [""] + [rf"\multicolumn{{{len(columns)}}}{{c}}{{{cfg.method_labels[m]}}}"
                          for m in methods]
        lines.append(" & ".join(header1) + r" \\")

        cmids, start = [], 2
        for _ in methods:
            cmids.append(rf"\cmidrule(lr){{{start}-{start + len(columns) - 1}}}")
            start += len(columns)
        lines.append(" ".join(cmids))

        header2 = ["Dataset"] + [COLUMN_SPECS[c][0] for c in columns] * len(methods)
        lines.append(" & ".join(header2) + r" \\")
        lines.append(r"\midrule")

        pooled = {m: {c: [] for c in columns} for m in methods}
        for ds in datasets:
            # 1) Dataset names have no spaces to wrap at, so break them at the configured segments
            label = ds.replace("_", r"\_")
            for marker in cfg.dataset_label_breaks:
                escaped = marker.replace("_", r"\_")
                label = label.replace(escaped, r"\newline " + escaped)
            cells = [label]
            for m in methods:
                match = sub[(sub["Dataset"] == ds) & (sub["CE Method"] == m)]
                if match.empty:
                    cells += ["--"] * len(columns)
                    continue
                row = match.iloc[0]
                for c in columns:
                    _, formatter, check_field, mean_field = COLUMN_SPECS[c]
                    cells.append("--" if pd.isna(row.get(check_field)) else formatter(row))
                    if mean_field and not pd.isna(row[mean_field]):
                        pooled[m][c].append(row[mean_field])
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\midrule")
        mean_cells = ["Mean"]
        for m in methods:
            for c in columns:
                values, mean_field = pooled[m][c], COLUMN_SPECS[c][3]
                mean_cells.append(COLUMN_SPECS[c][1]({mean_field: float(np.mean(values))})
                                  if values else "--")
        lines.append(" & ".join(mean_cells) + r" \\")

        lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]

        path = f"{out_dir}/table_{cfg.exp_name}_{model}_{metric_name}.tex"
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[OK] {path}")


@hydra.main(config_path="../data/config", config_name="CE_sensitivity_plot", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_style()
    out_dir = f"{cfg.ce_base_path}/comparison"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    palette = dict(zip(cfg.methods, sns.color_palette("tab10", len(cfg.methods))))

    ##############################
    # 1) Load and block the sweep
    ##############################
    df = load_sensitivity_data(cfg)
    if df.empty:
        print("[WARN] No data loaded. Exiting.")
        return
    df.to_csv(f"{out_dir}/df_{cfg.exp_name}.csv", index=False)

    cells, dropped = build_cell_matrices(df)
    if not cells:
        print("[WARN] No cell survived complete-case filtering. Exiting.")
        return
    dropped.to_csv(f"{out_dir}/dropped_{cfg.exp_name}.csv", index=False)
    if not dropped.empty:
        print(f"[INFO] dropped {len(dropped)} (cell, n_clusters) levels with failed runs, "
              f"see dropped_{cfg.exp_name}.csv")

    ##############################
    # 2) Indices, tests, equivalence
    ##############################
    si = compute_sensitivity_indices(cells, cfg, rng)
    si.to_csv(f"{out_dir}/SI_{cfg.exp_name}.csv", index=False)
    report_cols = ["Dataset", "CE Method", "n_seeds", "n_levels", "S1_J", "S1_J_lo", "S1_J_hi",
                   "S1_seed", "range_J", "p_holm", "equiv_seed", "equiv_fixed"]
    primary = si[si["Metric"] == f"sRC_{cfg.primary_eval_metric}"]
    print(primary[report_cols].to_string(index=False))
    for model in cfg.models:
        make_sensitivity_table(si, cfg, model, out_dir)

    ##############################
    # 3) Curves
    ##############################
    points = build_point_frame(df)
    em = cfg.primary_eval_metric
    families = sorted(points["Family"].unique())
    datasets = list(cfg.datasets)
    for facet_col, facets, fontsize in [("Family", families, 20), ("Dataset", datasets, 15)]:
        response = summarize_curves(points, "Value", facet_col, cfg.response_band, cfg)
        plot_curve_grid(response, cfg, facet_col, facets, "", palette=palette,
                        path_stem=f"{out_dir}/response_{facet_col.lower()}_{cfg.exp_name}_{em}",
                        title_fontsize=fontsize, share_y=cfg.share_y)


if __name__ == "__main__":
    main()
