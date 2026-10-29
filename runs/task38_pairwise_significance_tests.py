import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os
import itertools
import string
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import seaborn as sns
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

from task31_plot_metrics import apply_k_mode, CE_DISPLAY_NAMES, setup_style


def save_pdf_figure(fig, path_stem: str) -> None:
    """PDF only (no PNG) -- unlike task31's shared save_figure, which always
    writes a PNG and treats PDF as the optional extra."""
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_family_df(ce_base_path: str, exp_name: str, k_mode: str) -> pd.DataFrame:
    csv_path = f"{ce_base_path}/comparison/dfboxplot_{exp_name}.csv"
    df = pd.read_csv(csv_path)
    n_before = len(df)
    df = df.dropna(subset=["Dataset", "Model", "CE Method", "Seed", "Metric", "Value"])
    if len(df) < n_before:
        print(f"[WARN] {csv_path}: dropped {n_before - len(df)} malformed row(s) with missing fields.")
    return apply_k_mode(df, k_mode)


def holm_correction(pvals: list[float]) -> np.ndarray:
    """Holm (1979) step-down FWER correction: sort ascending, compare the i-th
    smallest to alpha/(m-i+1), enforce monotonicity. Returns adjusted p-values
    in the original input order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min((m - rank) * pvals[idx], 1.0))
        adjusted[idx] = running_max
    return adjusted


def rank_biserial(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Matched-pairs rank-biserial correlation r = (W+ - W-) / (n(n+1)/2),
    positive when x tends to exceed y."""
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return np.nan, 0
    ranks = scipy_stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    r = (w_plus - w_minus) / (n * (n + 1) / 2)
    return r, n


def pairwise_tests(wide_df: pd.DataFrame, methods_display: list[str]):
    """Paired Wilcoxon signed-rank test + rank-biserial effect size for every
    method pair, Holm-corrected across the pairs actually tested, plus a
    Friedman omnibus statistic on the fully-paired subset for context."""
    k = len(methods_display)
    pair_info = []
    for i, j in itertools.combinations(range(k), 2):
        mi, mj = methods_display[i], methods_display[j]
        sub = wide_df[[mi, mj]].dropna()
        x, y = sub[mi].to_numpy(), sub[mj].to_numpy()
        n_total = len(x)
        p_raw, r, n_nonzero = np.nan, np.nan, 0
        if n_total >= 3 and not np.all(x == y):
            try:
                _, p_raw = scipy_stats.wilcoxon(x, y)
                r, n_nonzero = rank_biserial(x, y)
            except ValueError:
                p_raw = np.nan
        pair_info.append({
            "i": i, "j": j, "p_raw": p_raw, "r": r,
            "n": n_nonzero if n_nonzero else n_total,
        })

    valid = [info for info in pair_info if not np.isnan(info["p_raw"])]
    if valid:
        adjusted = holm_correction([info["p_raw"] for info in valid])
        for info, p_adj in zip(valid, adjusted):
            info["p_holm"] = p_adj
    for info in pair_info:
        info.setdefault("p_holm", np.nan)

    full = wide_df[methods_display].dropna()
    friedman = None
    if len(full) >= 3 and k >= 3:
        stat, p = scipy_stats.friedmanchisquare(*[full[m].to_numpy() for m in methods_display])
        friedman = {"stat": stat, "p": p, "n": len(full)}

    return pair_info, friedman


def build_cell_lookup(pair_info: list[dict]) -> dict:
    """Mirror the upper-triangular pair results into a full i!=j lookup;
    r flips sign when the pair is read in the opposite order, p/n don't."""
    lookup = {}
    for info in pair_info:
        i, j = info["i"], info["j"]
        lookup[(i, j)] = {"p_holm": info["p_holm"], "r": info["r"], "n": info["n"]}
        r_ji = -info["r"] if not pd.isna(info["r"]) else np.nan
        lookup[(j, i)] = {"p_holm": info["p_holm"], "r": r_ji, "n": info["n"]}
    return lookup


def sig_marker(p_holm: float, alpha: float) -> str:
    return "*" if (not pd.isna(p_holm) and p_holm < alpha) else ""


def format_p_sci(p: float, sig_digits: int = 2) -> str:
    if pd.isna(p):
        return "n/a"
    return f"{p:.{sig_digits}e}"


def format_cell_markdown(p_holm: float, r: float, n: int, alpha: float) -> str:
    if pd.isna(p_holm):
        return "—"
    p_str = f"{format_p_sci(p_holm)}{sig_marker(p_holm, alpha)}"
    r_str = f"{r:+.2f}" if not pd.isna(r) else "n/a"
    return f"{p_str} (r={r_str}, n={n})"


def build_markdown_matrix(methods_display: list[str], cell_lookup: dict, alpha: float) -> str:
    k = len(methods_display)
    header = "| |" + "|".join(methods_display) + "|"
    sep = "|---|" + "|".join(["---"] * k) + "|"
    lines = [header, sep]
    for i, mi in enumerate(methods_display):
        row = [mi]
        for j in range(k):
            if i == j:
                row.append("—")
                continue
            info = cell_lookup[(i, j)]
            row.append(format_cell_markdown(info["p_holm"], info["r"], info["n"], alpha))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_friedman_line(friedman: dict | None) -> str:
    if friedman is None:
        return "_Friedman omnibus: not computed (fewer than 3 fully-paired blocks)._"
    return (
        f"_Friedman omnibus (context only, not gating the pairwise tests below): "
        f"chi2={friedman['stat']:.3f}, p={friedman['p']:.4f}, n={friedman['n']} blocks._"
    )


def heatmap_label(method: str) -> str:
    """CENDRE method labels get their clustering variant set as a math-mode
    subscript for the heatmap axes, e.g. CENDREACsilhouette -> CENDRE_{ACsilhouette}
    (matches the CENDRe$_{\\mathrm{...}}$ convention already used in the
    sensitivity-plot tables). Non-CENDRE methods (ECLAD-ts, MultiVISION) pass through."""
    prefix = "CENDRE"
    if method.startswith(prefix) and len(method) > len(prefix):
        variant = method[len(prefix):]
        return rf"CENDRE$_{{\mathrm{{{variant}}}}}$"
    return method


def method_abbrev(index: int) -> str:
    """A, B, C, ... letter tag for a method's position in methods_display."""
    return f"({string.ascii_uppercase[index]})"


def draw_heatmap_on_ax(ax, methods_display: list[str], cell_lookup: dict, alpha: float,
                        annot_fontsize: float = 9, square: bool = True,
                        show_row_labels: bool = True) -> None:
    """Draw one method x method significance heatmap into an existing axes
    (no figure/colorbar/save -- shared by the standalone and grid plots).
    Standard diverging heatmap of the signed rank-biserial effect size (upper
    triangle only -- the matrix is antisymmetric), annotated with the
    Holm-adjusted p-value in scientific notation and a significance asterisk.
    Row labels carry the full method name + letter tag; column labels are just
    the letter tag (row labels are the legend for it) to keep the axis compact
    instead of repeating every full name twice. show_row_labels=False hides the
    y-axis method names entirely (for grid layouts where only the first column
    of a row needs them)."""
    k = len(methods_display)
    row_labels = [f"{heatmap_label(m)} {method_abbrev(idx)}" for idx, m in enumerate(methods_display)] \
        if show_row_labels else False
    col_labels = [method_abbrev(idx) for idx in range(k)]
    r_mat = np.full((k, k), np.nan)
    annot = np.full((k, k), "", dtype=object)
    for i in range(k):
        for j in range(k):
            if j <= i:
                continue
            info = cell_lookup[(i, j)]
            r_mat[i, j] = info["r"]
            p = info["p_holm"]
            annot[i, j] = "" if pd.isna(p) else f"{format_p_sci(p)}{sig_marker(p, alpha)}"
    mask = np.tril(np.ones((k, k), dtype=bool))

    sns.heatmap(
        r_mat, mask=mask, annot=False, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        xticklabels=col_labels, yticklabels=row_labels,
        linewidths=0.5, linecolor="white", square=square, ax=ax, cbar=False,
    )
    # Flat grid of square cells only -- no tick marks and no leftover axes
    # grid lines poking through the cell-boundary lines.
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(False)
    # Manual annotation with magnitude-based text color (white on dark cells,
    # black on light cells) -- a single global text color is illegible on one
    # end of a diverging colormap or the other.
    for i in range(k):
        for j in range(k):
            if j <= i or not annot[i, j]:
                continue
            text_color = "white" if abs(r_mat[i, j]) >= 0.5 else "black"
            ax.text(j + 0.5, i + 0.5, annot[i, j], ha="center", va="center",
                     color=text_color, fontsize=annot_fontsize)


def plot_heatmap_grid(family_model_data: dict, alpha: float, family_full_names: dict[str, str],
                       path_stem: str) -> None:
    """One combined figure for a single (eval_metric, metric_base): families
    stacked vertically (rows), architectures side by side (columns). Each cell
    of the grid is one full method x method significance heatmap.
    family_model_data: {(family_label, model_label): (methods_display, cell_lookup)}.
    family_full_names: {family_label: full experiment name}, e.g. SyntheticL -> syntheticLocal,
    used only for the row title (small caps)."""
    families = list(dict.fromkeys(fam for fam, _ in family_model_data))
    models = sorted({mod for _, mod in family_model_data})
    n_rows, n_cols = len(families), len(models)
    if n_rows == 0 or n_cols == 0:
        return

    # Families can have different method counts (e.g. SyntheticF only ran the 3 CENDRE
    # variants, SyntheticL ran all 5) -- size each row's height by its own method count
    # and drop the square-cell constraint (square=False) so every heatmap fills its
    # allocated box edge to edge instead of being centered/shrunk to the smaller family's
    # scale, which otherwise leaves large empty margins around the smaller matrices.
    row_ks = []
    for family in families:
        k_here = next(len(family_model_data[(family, m)][0]) for m in models if (family, m) in family_model_data)
        row_ks.append(k_here)
    max_k = max(row_ks)
    cell_unit = 1.15
    # First column is wider to fit the method-name row labels; the others only need
    # room for the heatmap itself since their row labels are suppressed (see below).
    width_ratios = [1.35] + [1.0] * (n_cols - 1)
    fig_width = cell_unit * max_k * (sum(width_ratios) / n_cols) * n_cols + 1.6
    fig_height = cell_unit * sum(row_ks) + 1.3

    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False, constrained_layout=True,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": row_ks, "width_ratios": width_ratios},
    )

    last_mappable = None
    for row_idx, family in enumerate(families):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx][col_idx]
            key = (family, model)
            if key not in family_model_data:
                ax.axis("off")
                continue
            methods_display, cell_lookup = family_model_data[key]
            draw_heatmap_on_ax(ax, methods_display, cell_lookup, alpha, annot_fontsize=8,
                                square=False, show_row_labels=(col_idx == 0))
            last_mappable = ax.collections[0]
            # Architecture name on every column of every row (no fig-level suptitle --
            # the metric label is added in the paper directly).
            ax.set_title(model, fontsize=16)
            if col_idx == 0:
                # Family name once per row, positioned above the method-name column
                # (not pushed all the way out to the figure's left margin), full
                # experiment name in small caps and bolded. Lowercased first: \textsc only
                # shrinks lowercase letters, so the camelCase exp names (syntheticLocal)
                # would otherwise render with the embedded capital at full cap-height,
                # e.g. "SYNTHETIC-L-ocal" instead of a uniform small-caps word.
                full_name = family_full_names.get(family, family).lower()
                ax.annotate(rf"\textsc{{{full_name}}}", xy=(-0.4, 1.08), xycoords="axes fraction",
                            ha="left", va="bottom", fontsize=21, fontweight="bold")

    if last_mappable is not None:
        fig.colorbar(last_mappable, ax=axes, shrink=0.6, label=r"rank-biserial $r$ (row $>$ col)")
    save_pdf_figure(fig, path_stem)


@hydra.main(config_path="../data/config", config_name="CE_pairwise_significance_tests", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_style()
    ce_base_path = cfg.ce_base_path
    out_dir = f"{ce_base_path}/comparison"
    os.makedirs(out_dir, exist_ok=True)

    methods_display = [CE_DISPLAY_NAMES.get(m, m) for m in cfg.methods]
    eval_metrics = list(cfg.eval_metrics)
    metric_bases = list(cfg.metric_bases)
    alpha = cfg.alpha

    # Collected across the family loop below, keyed by (eval_metric, metric_base) ->
    # {(family_label, model_label): (family_methods, cell_lookup, friedman)}, to build one
    # combined families-x-architectures heatmap grid PDF + one matching combined md table
    # per metric, after all families are processed. No per-(family, model) files anymore --
    # only these two combined, per-metric artifacts.
    grid_data: dict[tuple[str, str], dict[tuple[str, str], tuple]] = {}

    for family_label, family_exp_name in cfg.families.items():
        df = load_family_df(ce_base_path, family_exp_name, cfg.k_mode)
        models = sorted(df["Model"].unique())

        # Not every family necessarily ran every configured method (e.g. syntheticFrequency
        # only ran the CENDRE variants) -- restrict to the methods actually present for this
        # family rather than skipping every matrix outright.
        present = set(df["CE Method"].unique())
        family_methods = [m for m in methods_display if m in present]
        skipped = [m for m in methods_display if m not in present]
        if skipped:
            print(f"[INFO] {family_label}: methods not present in this family's data, excluded: {skipped}")
        if len(family_methods) < 2:
            print(f"[WARN] {family_label}: fewer than 2 configured methods present. Skipping family.")
            continue

        for model_label in models:
            for em in eval_metrics:
                for base in metric_bases:
                    m_name = f"{base}_{em}"
                    sub = df.loc[
                        (df["Metric"] == m_name) & (df["CE Method"].isin(family_methods))
                        & (df["Model"] == model_label)
                    ]
                    if sub.empty:
                        print(f"[WARN] No data for {family_label}/{model_label}/{m_name}. Skipping.")
                        continue
                    wide = sub.pivot_table(index=["Dataset", "Seed"], columns="CE Method", values="Value")
                    missing = [m for m in family_methods if m not in wide.columns]
                    if missing:
                        print(f"[WARN] Missing methods {missing} for {family_label}/{model_label}/{m_name}. Skipping.")
                        continue
                    wide = wide[family_methods]

                    pair_info, friedman = pairwise_tests(wide, family_methods)
                    cell_lookup = build_cell_lookup(pair_info)

                    grid_data.setdefault((em, base), {})[(family_label, model_label)] = (
                        family_methods, cell_lookup, friedman,
                    )

    # One combined heatmap grid PDF + one matching combined md table per metric: dataset
    # families stacked vertically, architectures side by side in the PDF; the md table
    # mirrors the same scope (family -> architecture -> matrix), sharing the file stem.
    for (em, base), family_model_data in grid_data.items():
        m_name = f"{base}_{em}"
        grid_stem = f"{out_dir}/pairwise_heatmap_grid_{em}_{base}"

        heatmap_data = {key: value[:2] for key, value in family_model_data.items()}
        plot_heatmap_grid(heatmap_data, alpha, dict(cfg.families), grid_stem)

        families_seen = list(dict.fromkeys(fam for fam, _ in family_model_data))
        md_lines = [f"# {m_name}\n"]
        for family in families_seen:
            md_lines.append(f"## {family}\n")
            for (fam, model_label), (fam_methods, cell_lookup, friedman) in family_model_data.items():
                if fam != family:
                    continue
                md_table = build_markdown_matrix(fam_methods, cell_lookup, alpha)
                md_lines.append(f"### {model_label}\n\n{format_friedman_line(friedman)}\n\n{md_table}\n")
        md_path = f"{grid_stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"[OK] {m_name} -> {grid_stem}.pdf, {md_path}")


if __name__ == "__main__":
    main()
