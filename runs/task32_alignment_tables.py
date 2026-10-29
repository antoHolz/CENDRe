import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import torch
import numpy as np
import json
import glob
import yaml
from omegaconf import DictConfig
from distmap import euclidean_distance_transform
from cel.utils.correctness import RC, METRICS, dst_eclad
from cel.utils.misc import get_x


CENDRE_PREFIXES = ("CENDRE",)
EVAL_METRICS = ["dst_eclad", "rma"]

# Mirrors CENDRE.naming: maps method suffix -> (first_letter, uses_n_clusters, uses_n_concepts)
CENDRE_MODE_MAP = {
    "kMeans":        ("k", False, True),
    "ACnConcepts":   ("A", True,  True),
    "ACelbow":       ("A", True,  False),
    "ACsilhouette":  ("A", True,  False),
    "HDBSCAN":       ("H", True,  False),
    "ACthreshold":   ("A", False, False),  # ac_factor path — handled separately
}


def cendre_ac_id(method_name, k_str, n_clusters):
    """Reconstruct the AC_id suffix CENDRE uses for saved files.
    method_name is e.g. "CENDREACsilhouette"; k_str is the n_concepts from
    the scores filename. Returns None if the mode is not recognized."""
    if not method_name.startswith("CENDRE"):
        return k_str
    mode = method_name[len("CENDRE"):]
    info = CENDRE_MODE_MAP.get(mode)
    if info is None:
        return None
    letter, uses_nc, uses_kc = info
    parts = [letter]
    if uses_nc:
        parts.append(str(n_clusters))
    if uses_kc:
        parts.append(str(k_str))
    return "_".join(parts)


def compute_max_penalties(input_len, eval_metrics, device="cpu"):
    """Compute max_penalty per metric (mirrors task30 logic)."""
    extreme = torch.ones(input_len, device=device)
    extreme[0] = 0
    dst_max_penalty = float(
        torch.sum(euclidean_distance_transform(extreme)) / (max(input_len) ** 2)
    )
    return {
        m: dst_max_penalty if METRICS[m]["fn"] is dst_eclad else 1.0
        for m in eval_metrics
    }


def get_input_len(cfg):
    """Get input length by instantiating one dataset sample."""
    pretransform = hydra.utils.instantiate(cfg.dataset.pretransform)
    valid_transform = hydra.utils.instantiate(cfg.transforms.val)
    valid_dataset = hydra.utils.instantiate(
        cfg.dataset.val, transform=valid_transform,
        pretransform=pretransform, withId=True,
    )
    x = get_x(valid_dataset[0])
    return x.shape[1:]  # drop channel dim


def load_maskcols(dataset_name):
    """Read maskcols from the dataset config YAML."""
    yaml_path = Path(f"./data/config/dataset/{dataset_name}.yaml")
    if not yaml_path.exists():
        return []
    with open(yaml_path) as f:
        ds_cfg = yaml.safe_load(f)
    return ds_cfg.get("maskcols", [])


def load_importances_raw(out_dir, ac_id):
    """Load raw importances from importances_{ac_id}.json in out_dir.
    Returns {} if the exact file does not exist (no glob fallback — a stale
    neighbour with a different concept count would cause a shape mismatch)."""
    if ac_id is None:
        return {}
    specific = Path(out_dir) / f"importances_{ac_id}.json"
    if specific.exists():
        with open(specific) as f:
            return json.load(f)
    return {}


def importances_flat(raw_importances, n_channels):
    """
    Build a flat (n_concepts * n_channels,) importance tensor from raw JSON.
    Replicates the logic of _flat_importances from correctness.py.
    """
    n_concepts = len(raw_importances)
    vals = [raw_importances[str(i)] for i in range(n_concepts)]
    imp = torch.tensor(vals, dtype=torch.float32)
    if imp.ndim == 1:
        imp = imp.unsqueeze(1).expand(-1, n_channels).flatten()
    else:
        imp = imp.flatten()
    return imp


def importances_scalar(raw_importances):
    """Convert raw importances to {int: float} for display (max across channels)."""
    result = {}
    for k, v in raw_importances.items():
        if isinstance(v, list):
            result[int(k)] = float(max(v))
        else:
            result[int(k)] = float(v)
    return result


def compute_ic(alignments, raw_importances):
    """Compute IC from alignments tensor and raw importances dict."""
    n_channels = alignments.shape[2]
    imp = importances_flat(raw_importances, n_channels)

    aligned_mask = alignments.sum(dim=0).bool().flatten()
    denom_aligned = aligned_mask.sum().float().clamp_min(1.0)
    denom_unaligned = (~aligned_mask).sum().float().clamp_min(1.0)

    I_aligned = aligned_mask.float() * imp / denom_aligned
    I_unaligned = (~aligned_mask).float() * imp / denom_unaligned
    max_imp = torch.max(torch.abs(imp)).clamp_min(1e-12)

    ic = float((torch.sum(I_aligned) - torch.sum(I_unaligned)) / max_imp)
    return ic


def build_metric_section(metric_name, scores_path, raw_importances, prim_names,
                         max_penalty, higher_is_better, is_best=False):
    """Load scores, run RC, return (markdown_section, rc_value).
    The number of concepts is taken from raw_importances (authoritative);
    scores is truncated to that count if larger."""
    scores = torch.tensor(np.load(scores_path), dtype=torch.float32)
    n_concepts = min(len(raw_importances), scores.shape[1])
    scores = scores[:, :n_concepts, :]
    raw_importances = {str(i): raw_importances[str(i)] for i in range(n_concepts)}
    n_primitives, n_concepts, n_channels = scores.shape

    rc, alignments, sim = RC(
        scores, mode='rel', max_penalty=max_penalty,
        higher_is_better=higher_is_better,
    )

    ic = compute_ic(alignments, raw_importances)
    imp_display = importances_scalar(raw_importances)

    best_marker = " **[BEST]**" if is_best else ""
    lines = [f"## {metric_name} (RC = {rc:.4f}, IC = {ic:.4f}){best_marker}\n"]

    lines.append("| Primitive | Aligned Concepts (similarity) |")
    lines.append("|-----------|-------------------------------|")

    aligned_concepts = set()

    for p in range(n_primitives):
        pname = prim_names[p] if p < len(prim_names) else f"prim_{p}"
        aligned = []
        for c in range(n_concepts):
            for ch in range(n_channels):
                if alignments[p, c, ch] > 0:
                    aligned_concepts.add(c)
                    ch_str = f"/ch{ch}" if n_channels > 1 else ""
                    aligned.append(f"C{c}{ch_str} ({sim[p, c, ch]:.3f})")
        cell = ", ".join(aligned) if aligned else "\u2014"
        lines.append(f"| {pname} | {cell} |")

    # Unaligned concepts with importances
    unaligned = []
    for c in range(n_concepts):
        if c not in aligned_concepts:
            imp = imp_display.get(c, 0.0)
            unaligned.append(f"C{c} (imp={imp:.3f})")
    if unaligned:
        lines.append(f"\nUnaligned: {', '.join(unaligned)}")

    lines.append("")
    return "\n".join(lines), rc


def process_folder(out_dir, eval_metrics, max_penalties, prim_names, header,
                   method_name=None, n_clusters=None):
    """
    Process one experiment folder: load all score files, build full MD report.
    Returns markdown string or None if no scores found.
    """
    # Collect all (metric, score_path, k_str, metric_info) entries
    entries = []
    for metric_name in eval_metrics:
        metric_info = METRICS.get(metric_name)
        if metric_info is None:
            continue
        pattern = f"{out_dir}/scores_{metric_name}_*.npy"
        score_files = sorted(glob.glob(pattern))
        for score_path in score_files:
            stem = Path(score_path).stem
            k_str = stem.rsplit("_", 1)[-1]
            entries.append((metric_name, score_path, k_str, metric_info))

    if not entries:
        return None

    # Load importances per k value, reconstructing the CENDRE AC_id when needed
    importances_by_k = {}
    for _, _, k_str, _ in entries:
        if k_str in importances_by_k:
            continue
        ac_id = cendre_ac_id(method_name, k_str, n_clusters) if method_name else k_str
        importances_by_k[k_str] = load_importances_raw(out_dir, ac_id)

    # Check at least one k has importances
    if not any(importances_by_k.values()):
        return None

    # Pre-compute RC for each entry to find best k per metric
    rc_values = {}
    for metric_name, score_path, k_str, metric_info in entries:
        if not importances_by_k.get(k_str):
            continue
        scores = torch.tensor(np.load(score_path), dtype=torch.float32)
        rc, _, _ = RC(
            scores, mode='rel', max_penalty=max_penalties[metric_name],
            higher_is_better=metric_info["higher_is_better"],
        )
        rc_values[(metric_name, k_str)] = rc

    # Find best k per metric (highest RC)
    best_k = {}
    for metric_name in eval_metrics:
        metric_entries = [(k, rc) for (m, k), rc in rc_values.items() if m == metric_name]
        if metric_entries:
            best_k[metric_name] = max(metric_entries, key=lambda x: x[1])[0]

    # Build sections with best marking
    sections = [f"# {header}\n"]
    for metric_name, score_path, k_str, metric_info in entries:
        raw_importances = importances_by_k.get(k_str, {})
        if not raw_importances:
            continue
        is_best = best_k.get(metric_name) == k_str
        section, _ = build_metric_section(
            metric_name=f"{metric_name} (k={k_str})",
            scores_path=score_path,
            raw_importances=raw_importances,
            prim_names=prim_names,
            max_penalty=max_penalties[metric_name],
            higher_is_better=metric_info["higher_is_better"],
            is_best=is_best,
        )
        if section is not None:
            sections.append(section)

    # Build nearest-primitive summary table (best k only)
    nearest_section = build_nearest_table(entries, best_k, prim_names, importances_by_k)
    if nearest_section:
        sections.append(nearest_section)

    return "\n".join(sections)


def build_nearest_table(entries, best_k, prim_names, importances_by_k):
    """
    Build a summary table showing, for each concept, the nearest primitive
    (best raw score) per metric. Uses only the best-k entries.
    """
    # Filter to best-k entries only, one per metric (first = most primitives)
    best_entries = []
    seen_metrics = set()
    for metric_name, score_path, k_str, metric_info in entries:
        if best_k.get(metric_name) == k_str and metric_name not in seen_metrics:
            best_entries.append((metric_name, score_path, k_str, metric_info))
            seen_metrics.add(metric_name)
    if not best_entries:
        return None

    # Load scores for each metric, truncated to that metric's importances count
    metric_scores = {}  # metric_name -> (scores_tensor, higher_is_better)
    n_concepts = 0
    n_channels = 0
    for metric_name, score_path, k_str, metric_info in best_entries:
        scores = torch.tensor(np.load(score_path), dtype=torch.float32)
        nc = len(importances_by_k.get(k_str, {}))
        if nc == 0:
            continue
        scores = scores[:, :nc, :]
        metric_scores[metric_name] = (scores, metric_info["higher_is_better"])
        n_concepts = max(n_concepts, scores.shape[1])
        n_channels = max(n_channels, scores.shape[2])

    if not metric_scores:
        return None
    metric_names = list(metric_scores.keys())

    # Build header
    lines = ["## Nearest Primitive per Concept\n"]
    header_cols = ["Concept"] + metric_names
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

    for c in range(n_concepts):
        for ch in range(n_channels):
            concept_label = f"C{c}" + (f"/ch{ch}" if n_channels > 1 else "")
            cells = [concept_label]
            for m_name in metric_names:
                scores, higher = metric_scores[m_name]
                if c >= scores.shape[1] or ch >= scores.shape[2]:
                    cells.append("")
                    continue
                col = scores[:, c, ch]  # (n_primitives,)
                if higher:
                    best_p = int(col.argmax())
                else:
                    best_p = int(col.argmin())
                best_val = float(col[best_p])
                p_name = prim_names[best_p] if best_p < len(prim_names) else f"prim_{best_p}"
                cells.append(f"{best_val:.4f} ({p_name})")
            lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


@hydra.main(config_path="../data/config", config_name="CE_tables", version_base="1.3")
def main(cfg: DictConfig) -> None:
    eval_metrics = EVAL_METRICS
    n_seeds = cfg.seeds
    models = cfg.models
    methods = cfg.methods
    n_concepts = cfg.n_concepts
    outnr = cfg.outnr
    dataset_names = list(cfg.datasets) if hasattr(cfg, "datasets") else [cfg.dataset.name]

    # Compute max_penalties (needs input_len from one dataset sample)
    input_len = get_input_len(cfg)
    max_penalties = compute_max_penalties(input_len, eval_metrics)

    for dataset_name in dataset_names:
        prim_names = load_maskcols(dataset_name)

        for model in models:
            folder_structure = (
                f"{dataset_name}_{model}_{cfg.transforms.name}"
                f"_{cfg.loader.name}_{cfg.training.name}"
            )
            for ce in methods:
                n_concepts_CE = [0] if ce.startswith(CENDRE_PREFIXES) else n_concepts
                for n_concept in n_concepts_CE:
                    for seed in n_seeds:
                        out_dir = (
                            f"./data/ConceptExtraction/{ce}"
                            f"/{folder_structure}__{seed}"
                            f"_{cfg.suffix}/out{outnr}"
                        )
                        if not Path(out_dir).is_dir():
                            continue

                        header = f"{ce} / {dataset_name} / {model} / seed={seed}"
                        md = process_folder(
                            out_dir, eval_metrics, max_penalties,
                            prim_names, header,
                            method_name=ce,
                            n_clusters=cfg.get("n_clusters", None),
                        )
                        if md is None:
                            print(f"[SKIP] No scores in {out_dir}")
                            continue

                        md_path = f"{out_dir}/alignment_table.md"
                        with open(md_path, "w") as f:
                            f.write(md)
                        print(f"[OK] {md_path}")


if __name__ == "__main__":
    main()
