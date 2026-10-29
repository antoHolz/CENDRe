import warnings
import torch
import numpy as np
from torchmetrics import Metric
from distmap import euclidean_distance_transform
from modules.transforms.tstransforms import TSResize
from .misc import infer_shape


class DSTMetric(Metric):
    """
    TorchMetrics-compatible metric for computing pairwise scores between
    primitives (GT masks) and concept masks.

    Accumulates scores and instance counts across batches via update(),
    then normalizes in compute() to produce a tensor of shape
    (n_primitives, n_concepts, score_channels), where score_channels is
    data_channels in "per_channel" mode and 1 otherwise.

    Channel modes:
        - "per_channel": CENDRE / ECLADts. One distance per data channel,
          with both prim and mask zeroed on the other channels.
        - "broadcast":  MultiVISION / ECLAD / ConceptShap on multi-channel
          data. The 1-channel concept mask is duplicated across all data
          channels and compared against the full multi-channel primitive.
          One score per (p, c).
        - "single":     any method on single-channel data. One score per (p, c).

    Parameters:
        n_primitives (int): Number of GT primitives.
        n_concepts (int): Number of concepts to evaluate.
        data_channels (int): Number of channels in the underlying data.
        channel_mode (str): One of {"per_channel", "broadcast", "single"}.
        metric_fn (callable, optional): Low-level metric function. If None, uses dst_eclad.
        max_penalty (float): Maximum penalty for dst_eclad; 1.0 for soft metrics.
    """

    def __init__(self, n_primitives, n_concepts, data_channels, channel_mode,
                 metric_fn=None, max_penalty=1.0, concept_clamps=None,
                 use_clamps: bool = False, **kwargs):
        super().__init__(**kwargs)
        if channel_mode not in {"per_channel", "broadcast", "single"}:
            raise ValueError(f"Unknown channel_mode: {channel_mode}")
        self.use_dst = metric_fn is None or metric_fn is dst_eclad
        self.metric_fn = metric_fn if metric_fn is not None else dst_eclad
        self.max_penalty = max_penalty
        self.data_channels = data_channels
        self.channel_mode = channel_mode
        # use_clamps=False (default): per-instance normalisation (mask / instance max).
        # use_clamps=True: dataset-level clamp normalisation using concept_clamps.
        self.use_clamps = use_clamps
        self.concept_clamps = concept_clamps or {}  # {c_idx: float clamp value}
        score_channels = data_channels if channel_mode == "per_channel" else 1

        self.add_state(
            "scores",
            default=torch.zeros(n_primitives, n_concepts, score_channels),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "datapoints",
            default=torch.zeros(n_primitives, n_concepts, score_channels),
            dist_reduce_fx="sum",
        )

    def _call(self, prim, mask, clamp_val=None):
        if self.use_dst:
            return self.metric_fn(prim, mask, self.max_penalty, prim.device)
        return self.metric_fn(prim, mask, prim.device, clamp_val=clamp_val)

    def update(self, primitive_masks, concept_masks):
        """
        Accumulate scores for one batch.

        Args:
            primitive_masks (dict): {p_idx: tensor} — resized GT masks for this batch.
            concept_masks (dict): {c_idx: tensor} — resized concept masks for this batch.
        """
        C = self.data_channels
        for p, prim in primitive_masks.items():
            if prim.dim() == 2:
                prim = prim.unsqueeze(1)  # (B, 1, L)
            for c, cmask in concept_masks.items():
                if cmask.dim() == 2:
                    cmask = cmask.unsqueeze(1)

                clamp_val = self.concept_clamps.get(c, None) if self.use_clamps else None

                if self.channel_mode == "single":
                    s, inst = self._call(prim, cmask, clamp_val=clamp_val)
                    self.scores[p, c, 0] += s
                    self.datapoints[p, c, 0] += inst

                elif self.channel_mode == "broadcast":
                    if cmask.shape[1] == 1 and C > 1:
                        mask_full = cmask.expand(-1, C, -1)
                    else:
                        mask_full = cmask
                    s, inst = self._call(prim, mask_full, clamp_val=clamp_val)
                    self.scores[p, c, 0] += s
                    self.datapoints[p, c, 0] += inst

                else:  # per_channel
                    for ch in range(C):
                        mask_ch = torch.zeros(
                            cmask.shape[0], C, cmask.shape[-1],
                            device=cmask.device, dtype=cmask.dtype,
                        )
                        if cmask.shape[1] == C:
                            mask_ch[:, ch, :] = cmask[:, ch, :]
                        else:
                            mask_ch[:, ch, :] = cmask[:, 0, :]

                        prim_ch = torch.zeros(
                            prim.shape[0], C, prim.shape[-1],
                            device=prim.device, dtype=prim.dtype,
                        )
                        if prim.shape[1] == C:
                            prim_ch[:, ch, :] = prim[:, ch, :]
                        else:
                            prim_ch[:, ch, :] = prim[:, 0, :]

                        s, inst = self._call(prim_ch, mask_ch, clamp_val=clamp_val)
                        self.scores[p, c, ch] += s
                        self.datapoints[p, c, ch] += inst

    def compute(self):
        return self.scores / torch.clamp(self.datapoints, 1e-12)



def RC(dist, mode='min', max_penalty=1.0, device="cpu", higher_is_better=False, threshold=0.80):
    """
    Representation Correctness (RC) Metric.

    Computes the correctness of a representation by comparing the scores
    between concepts and primitives. The result is always in [0, 1] where
    1 means perfect alignment. For distance metrics (lower=better), scores
    are converted to similarity via 1 - dist/max_penalty before averaging.

    Modes:
    - 'min': Aligns each concept with its single best primitive
      (argmax for similarity, argmin for distance).
    - 'rel': Aligns each primitive with all concepts whose similarity
      exceeds `threshold`.

    Parameters:
        dist (torch.Tensor): Scores tensor of shape (n_primitives, n_concepts, n_channels).
        mode (str): The alignment mode ('min' or 'rel').
        max_penalty (float): Reference value. For distance metrics this is the
            maximum possible distance; for similarity metrics use 1.0.
        device (str): The computation device, default is "cpu".
        higher_is_better (bool): If True, higher scores indicate better alignment
            (for soft metrics). If False, lower scores are better (for dst_eclad).
        threshold (float): Similarity threshold for 'rel' mode. Default 0.80.

    Returns:
        tuple:
            - rc (float): The representation correctness score in [0, 1].
            - alignments (torch.Tensor): Alignment indicator tensor.
    """
    dist = dist.to(device)

    # Convert to similarity in [0, 1]: higher = better
    if higher_is_better:
        sim = dist.clamp(0.0, max_penalty) / max(max_penalty, 1e-12)
    else:
        sim = 1.0 - dist.clamp(0.0, max_penalty) / max(max_penalty, 1e-12)

    alignments = torch.zeros_like(dist)

    if mode == 'min':
        # For each concept (dim=0: primitives), pick the best primitive
        best_idx = sim.argmax(dim=0)
        # Build alignment indicator from argmax indices
        alignments = torch.zeros_like(dist)
        for c in range(dist.shape[1]):
            for ch in range(dist.shape[2]):
                alignments[best_idx[c, ch], c, ch] = 1
        rc = sim[alignments.bool()].sum()

    elif mode == 'rel':
        alignments = torch.where(sim > threshold, 1, 0)
        rc = sim[alignments.bool()].sum()

    total_length = torch.sum(alignments)
    if total_length > 0:
        rc = float(rc) / float(total_length)
    else:
        rc = 0.0
    return float(rc), alignments, sim


def soft_RC(dist, max_penalty=1.0, device="cpu", higher_is_better=False):
    """
    Soft Representation Correctness — threshold-free.

    For each primitive, picks the concept whose total cross-channel similarity
    is highest, and averages those over primitives. Channel sum happens *inside*
    the max so we select one concept per primitive based on its full multichannel
    alignment, not one per (primitive, channel) slot.

        soft_RC = (1 / (n_channels * n_primitives)) * sum_p max_c sum_ch sim(p, c, ch)

    Result is in [0, 1] since per-channel sim is in [0, 1].

    Parameters:
        dist (torch.Tensor): Scores tensor of shape (n_primitives, n_concepts, n_channels).
        max_penalty (float): Reference value for normalisation.
        device (str): Computation device.
        higher_is_better (bool): If True, higher scores = better alignment.

    Returns:
        float: The soft RC score.
    """
    dist = dist.to(device)
    if higher_is_better:
        sim = dist.clamp(0.0, max_penalty) / max(max_penalty, 1e-12)
    else:
        sim = 1.0 - dist.clamp(0.0, max_penalty) / max(max_penalty, 1e-12)

    n_primitives, _, n_channels = sim.shape
    sim_summed = sim.sum(dim=2)              # (n_primitives, n_concepts)
    best_per_prim = sim_summed.amax(dim=1)   # (n_primitives,)
    return float(best_per_prim.sum() / (n_channels * n_primitives))


def soft_RCmax(dist, max_penalty=1.0, device="cpu", higher_is_better=False):
    """
    Soft Representation Correctness — best-channel variant.

    Like soft_RC but takes the best channel per (primitive, concept) pair
    instead of summing across channels. A primitive gets full credit if it
    aligns with some concept in *any single channel*.

        soft_RCmax = (1/n_primitives) * sum_p max_c max_ch sim(p, c, ch)

    Result is in [0, 1].

    Parameters:
        dist (torch.Tensor): Scores tensor of shape (n_primitives, n_concepts, n_channels).
        max_penalty (float): Reference value for normalisation.
        device (str): Computation device.
        higher_is_better (bool): If True, higher scores = better alignment.

    Returns:
        float: The soft RCmax score.
    """
    dist = dist.to(device)
    if higher_is_better:
        sim = dist.clamp(0.0, max_penalty) / max(max_penalty, 1e-12)
    else:
        sim = 1.0 - dist.clamp(0.0, max_penalty) / max(max_penalty, 1e-12)

    best_per_prim = sim.amax(dim=(1, 2))     # (n_primitives,)
    return float(best_per_prim.mean())


def otsu_RC(dist, max_penalty=1.0, device="cpu", higher_is_better=False):
    """
    Representation Correctness with Otsu-adaptive threshold.

    Like the 'rel' mode of RC, but instead of a hard-coded 0.80 threshold the
    split point is found via Otsu's method on the flattened similarity matrix.
    This makes the aligned/unaligned boundary adapt to the actual score
    distribution of each run.

    Parameters:
        dist (torch.Tensor): Scores tensor of shape (n_primitives, n_concepts, n_channels).
        max_penalty (float): Reference value for normalisation.
        device (str): Computation device.
        higher_is_better (bool): If True, higher scores = better alignment.

    Returns:
        tuple:
            - rc (float): The Otsu-thresholded RC score in [0, 1].
            - alignments (torch.Tensor): Binary alignment indicator.
            - sim (torch.Tensor): Similarity matrix.
            - threshold (float): The Otsu-determined threshold.
    """
    dist = dist.to(device)
    if higher_is_better:
        sim = dist.clamp(0.0, max_penalty) / max(max_penalty, 1e-12)
    else:
        sim = 1.0 - dist.clamp(0.0, max_penalty) / max(max_penalty, 1e-12)

    # Otsu on the flattened similarity values
    vals = sim.flatten().cpu().numpy()
    threshold = _otsu_threshold(vals)

    alignments = torch.where(sim > threshold, 1, 0)
    total = torch.sum(alignments).float()
    if total > 0:
        rc = float(sim[alignments.bool()].sum() / total)
    else:
        rc = 0.0
    return rc, alignments, sim, float(threshold)


def _otsu_threshold(vals):
    """Otsu's method on a 1-D array of floats. Returns the optimal threshold."""
    vals = np.sort(vals)
    n = len(vals)
    if n < 2:
        return float(vals[0]) if n == 1 else 0.0

    best_thr = float(vals[0])
    best_var = -1.0
    cumsum = np.cumsum(vals)
    total_sum = cumsum[-1]

    for i in range(1, n):
        if vals[i] == vals[i - 1]:
            continue
        w0 = i
        w1 = n - i
        mean0 = cumsum[i - 1] / w0
        mean1 = (total_sum - cumsum[i - 1]) / w1
        var_between = w0 * w1 * (mean0 - mean1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thr = (vals[i - 1] + vals[i]) / 2.0
    return float(best_thr)


def nmi_alignment(dist, max_penalty=1.0, device="cpu", higher_is_better=False):
    """
    Normalised Mutual Information between concept assignments and primitive
    assignments across all instances.

    Each instance is assigned to its best concept (argmax of the concept masks)
    and to its best primitive (argmax of the primitive masks).  NMI between these
    two label vectors captures how well knowing the concept tells you the
    primitive, without any pairwise threshold.

    Because the raw per-instance masks are not available at this stage (only the
    aggregated scores matrix is), we use the scores matrix as a soft assignment:
    for each (primitive, concept, channel) entry the similarity is the average
    score over instances.  We construct a contingency-like matrix from sim and
    compute NMI on it.

    Parameters:
        dist (torch.Tensor): Scores tensor of shape (n_primitives, n_concepts, n_channels).
        max_penalty (float): Reference value for normalisation.
        device (str): Computation device.
        higher_is_better (bool): If True, higher scores = better alignment.

    Returns:
        float: NMI score in [0, 1].
    """
    dist = dist.to(device)
    if higher_is_better:
        sim = dist.clamp(0.0, max_penalty) / max(max_penalty, 1e-12)
    else:
        sim = 1.0 - dist.clamp(0.0, max_penalty) / max(max_penalty, 1e-12)

    # Average over channels → (n_primitives, n_concepts)
    sim_2d = sim.mean(dim=2).cpu().numpy()

    # Treat sim as a joint distribution (non-negative, sum to 1)
    sim_2d = np.maximum(sim_2d, 0.0)
    total = sim_2d.sum()
    if total < 1e-12:
        return 0.0
    pxy = sim_2d / total  # joint distribution

    px = pxy.sum(axis=1)  # marginal over concepts  (primitives)
    py = pxy.sum(axis=0)  # marginal over primitives (concepts)

    # MI = sum p(x,y) * log( p(x,y) / (p(x)*p(y)) )
    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] > 1e-12 and px[i] > 1e-12 and py[j] > 1e-12:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

    # Entropies for normalisation
    hx = -float(np.sum(px[px > 1e-12] * np.log(px[px > 1e-12])))
    hy = -float(np.sum(py[py > 1e-12] * np.log(py[py > 1e-12])))

    denom = (hx + hy) / 2.0
    if denom < 1e-12:
        return 0.0
    return float(mi / denom)


def _flat_importances(CEobject, n_channels, device="cpu"):
    """Build a flat (n_concepts * n_channels,) importance tensor.

    Scalar importances (ECLAD, MultiVISION) are replicated n_channels times.
    Per-channel importances (CENDRE, ECLADts decorrelated) are flattened.
    """
    raw = CEobject.importances
    imp = torch.tensor(
        [raw[str(i)] for i in range(len(raw))],
        device=device, dtype=torch.float32,
    )
    imp = torch.relu(imp)  # discard confounding (negative) scores
    if imp.ndim == 1:
        imp = imp.unsqueeze(1).expand(-1, n_channels).flatten()
    else:
        imp = imp.flatten()
    return imp


def IC(concept_ids, CEobject, device="cpu"):
    """
    Compute the Importance Correctness (IC) metric from ECLAD.

    Treats each (concept, channel) pair as an independent unit (c * ch total).
    Scalar importances are replicated across channels; per-channel importances
    are flattened.

    Parameters:
        concept_ids (torch.Tensor): Alignment tensor of shape (n_primitives, n_concepts, n_channels).
        CEobject (object): An object containing the concept importances (CEobject.importances).
        device (str, optional): The device used for computation. Default is "cpu".

    Returns:
        tuple: (ic, ic_aligned, ic_unaligned)
    """
    n_channels = concept_ids.shape[2]
    importances = _flat_importances(CEobject, n_channels, device)

    # Flatten (n_concepts, n_channels) -> (n_concepts * n_channels,)
    concept_ids = concept_ids.sum(dim=0).bool().flatten().to(device)

    denom_aligned   = concept_ids.sum().float().clamp_min(1.0)
    denom_unaligned = (~concept_ids).sum().float().clamp_min(1.0)

    I_aligned = concept_ids.float() * importances / denom_aligned
    I_unaligned = (~concept_ids).float() * importances / denom_unaligned
    max_importance = torch.max(torch.abs(importances)).clamp_min(1e-12)

    ic = (torch.sum(I_aligned) - torch.sum(I_unaligned)) / max_importance

    return float(ic), float(torch.sum(I_aligned)/max_importance), float(torch.sum(I_unaligned) / max_importance)


def soft_IC(sim, CEobject, device="cpu"):
    """
    Soft Importance Correctness.

    Each concept's importance is weighted by its best similarity to any
    primitive (no aligned/unaligned split). Well-aligned concepts contribute
    fully; unaligned concepts contribute ~nothing.

        soft_IC = sum(w_{c,ch} * imp_{c,ch}) / sum(|imp_{c,ch}|)

    where w_{c,ch} = max_p sim(p, c, ch) in [0, 1].

    Result is in [0, 1] (importances are non-negative).

    Returns:
        float: The soft IC score.
    """
    n_channels = sim.shape[2]
    importances = _flat_importances(CEobject, n_channels, device)

    # Flatten sim to (n_primitives, n_concepts * n_channels), amax over primitives only
    sim = sim.to(device)
    sim_flat = sim.flatten(1, 2)                          # (n_primitives, n_concepts * n_channels)
    weights = sim_flat.amax(dim=0).clamp(0.0, 1.0)       # (n_concepts * n_channels,)

    denom = torch.sum(torch.abs(importances)).clamp_min(1e-12)
    sic = torch.sum(weights * importances) / denom

    return float(sic)


def soft_IC_perprim(sim, CEobject, primitive_channels, device="cpu"):
    """
    Soft Importance Correctness — per-primitive channel-matched.

    Each primitive contributes only on its own active channel; for each concept,
    the best-aligned primitive determines both the alignment value and the
    channel used to look up the concept's importance. Result: one (c, ch) slot
    per concept (instead of all n_channels), so the denominator no longer
    includes importance from channels where no primitive is anchored.

        align(p, c) = sim(p, c, ch_p)        if p has fixed channel
                    = max_ch sim(p, c, ch)    if p is broadcast (ch_p = -1)
        p*(c)       = argmax_p align(p, c)
        ch*(c)      = channel used by p*(c)

        soft_IC_perprim = sum_c align(p*(c), c) * imp(c, ch*(c))
                        / sum_c |imp(c, ch*(c))|

    Parameters:
        sim (torch.Tensor): Similarity tensor (n_primitives, n_concepts, n_channels).
        CEobject (object): Object with .importances attribute.
        primitive_channels (Iterable[int]): Per-primitive channel index along sim's
            channel axis. Use -1 for broadcast primitives (active on every channel).
        device (str): Computation device.

    Returns:
        float: The per-primitive soft IC score in [0, 1].
    """
    n_primitives, n_concepts, n_channels = sim.shape
    importances = _flat_importances(CEobject, n_channels, device)
    imp_2d = importances.view(n_concepts, n_channels)

    sim = sim.to(device)

    # Gating mask: 1 on each primitive's active channel(s); 0 elsewhere.
    mask = torch.zeros_like(sim)
    for p, chp in enumerate(primitive_channels):
        if 0 <= chp < n_channels:
            mask[p, :, chp] = 1.0
        else:  # broadcast / invalid index → treat as active on all channels
            mask[p, :, :] = 1.0

    # Disable non-active channels with -inf so amax/argmax pick from active ones only.
    NEG = torch.finfo(sim.dtype).min
    masked_sim = torch.where(mask > 0, sim, torch.full_like(sim, NEG))
    align_pc = masked_sim.amax(dim=2)        # (n_p, n_c)
    ch_winner = masked_sim.argmax(dim=2)     # (n_p, n_c)

    # For each concept, pick the best primitive and its winning channel.
    best_p = align_pc.argmax(dim=0)                                  # (n_c,)
    best_align = align_pc.amax(dim=0).clamp(0.0, 1.0)                # (n_c,)
    cidx = torch.arange(n_concepts, device=device)
    ch_for_imp = ch_winner[best_p, cidx]                             # (n_c,)
    imp_per_c = imp_2d[cidx, ch_for_imp]                              # (n_c,)

    denom = torch.sum(torch.abs(imp_per_c)).clamp_min(1e-12)
    return float(torch.sum(best_align * imp_per_c) / denom)


###############################################################################
# Soft metrics for continuous masks (analogous to dst_eclad)
# Signature: metric(primitive, mask, device, penalize_empty) -> (score, instances)
# - primitive (G): binary GT, shape (batch, channels, len)
# - mask (M): continuous concept mask (non-negative), same shape
# - penalize_empty: if True (default), asymmetric cases (one empty, one populated)
#   count as instances with score 0. If False, they are ignored entirely.
# - Returns (accumulated_score: float, instance_count: int)
###############################################################################


def _normalize_mask(mask, clamp_val=None):
    """Normalize mask to [0, 1].

    If clamp_val is provided (> 0), divides by it — producing a dataset-level
    normalisation that preserves inter-instance comparability (used for CENDRE).
    Otherwise falls back to per-instance normalisation (divide by instance max).
    """
    mask = torch.clamp(mask, min=0.0)
    if clamp_val is not None and clamp_val > 0:
        return torch.clamp(mask / clamp_val, 0.0, 1.0)
    spatial_dims = list(range(2, mask.dim()))
    max_vals = mask.amax(dim=spatial_dims, keepdim=True).clamp_min(1e-12)
    return torch.clamp(mask / max_vals, 0.0, 1.0)

def soft_iou(primitive, mask, device="cpu", penalize_empty=True, clamp_val=None):
    """
    Soft Intersection over Union.

    Formula: sum(M * G) / sum(M + G - M * G)

    Naturally extends IoU to continuous masks. When G is binary and M >= 0,
    the denominator is always non-negative.

    Parameters:
        primitive (torch.Tensor): Binary GT mask, shape (batch, channels, len).
        mask (torch.Tensor): Continuous concept mask (non-negative), same shape.
        device (str): Computation device.
        penalize_empty (bool): If True, asymmetric empty cases (one empty, one
            populated) count as instances with score 0. Default True.
        clamp_val (float, optional): Dataset-level clamp for normalisation (CENDRE).

    Returns:
        tuple: (score: float, instances: int)
    """
    primitive = primitive.float().to(device)
    mask = mask.float().to(device)
    if mask.dim() + 1 == primitive.dim():
        mask = mask.unsqueeze(1)
    mask = _normalize_mask(mask, clamp_val=clamp_val)

    spatial_dims = list(range(2, primitive.dim()))
    prim_empty = ~primitive.any(dim=spatial_dims)
    mask_empty = ~(mask > 0).any(dim=spatial_dims)
    both_populated = ~prim_empty & ~mask_empty
    countable = ~(prim_empty & mask_empty) if penalize_empty else both_populated

    intersection = (primitive * mask).sum(dim=spatial_dims)
    union = (primitive + mask - primitive * mask).sum(dim=spatial_dims)
    iou = intersection / union.clamp_min(1e-12)

    score = torch.where(countable, iou, torch.zeros_like(iou)).sum()
    instances = countable.sum()
    return float(score), int(instances)


def soft_dice(primitive, mask, device="cpu", penalize_empty=True, clamp_val=None):
    """
    Soft Dice coefficient (Sorensen-Dice).

    Formula: 2 * sum(M * G) / (sum(M) + sum(G))

    Parameters:
        primitive (torch.Tensor): Binary GT mask, shape (batch, channels, len).
        mask (torch.Tensor): Continuous concept mask (non-negative), same shape.
        device (str): Computation device.
        penalize_empty (bool): If True, asymmetric empty cases count as score 0. Default True.
        clamp_val (float, optional): Dataset-level clamp for normalisation (CENDRE).

    Returns:
        tuple: (score: float, instances: int)
    """
    primitive = primitive.float().to(device)
    mask = mask.float().to(device)
    if mask.dim() + 1 == primitive.dim():
        mask = mask.unsqueeze(1)
    mask = _normalize_mask(mask, clamp_val=clamp_val)

    spatial_dims = list(range(2, primitive.dim()))
    prim_empty = ~primitive.any(dim=spatial_dims)
    mask_empty = ~(mask > 0).any(dim=spatial_dims)
    both_populated = ~prim_empty & ~mask_empty
    countable = ~(prim_empty & mask_empty) if penalize_empty else both_populated

    PG = (primitive * mask).sum(dim=spatial_dims)
    denom = (primitive.sum(dim=spatial_dims) + mask.sum(dim=spatial_dims)).clamp_min(1e-12)
    dice = 2.0 * PG / denom

    score = torch.where(countable, dice, torch.zeros_like(dice)).sum()
    instances = countable.sum()
    return float(score), int(instances)


def cdc(primitive, mask, device="cpu", penalize_empty=True, clamp_val=None):
    """
    Continuous Dice Coefficient (Shamir et al. 2019).

    Formula: cDC = sum(M * G) / (c * sum(G) + sum(M))
    where c = sum(M * G) / sum(M * sign(G))

    Designed specifically for comparing continuous masks against binary GT.
    The correction factor c adjusts for how mask values distribute inside
    vs outside GT. When G is binary, sign(G) = G.

    Parameters:
        primitive (torch.Tensor): Binary GT mask, shape (batch, channels, len).
        mask (torch.Tensor): Continuous concept mask (non-negative), same shape.
        device (str): Computation device.
        penalize_empty (bool): If True, asymmetric empty cases count as score 0. Default True.
        clamp_val (float, optional): Dataset-level clamp for normalisation (CENDRE).

    Returns:
        tuple: (score: float, instances: int)
    """
    primitive = primitive.float().to(device)
    mask = mask.float().to(device)
    if mask.dim() + 1 == primitive.dim():
        mask = mask.unsqueeze(1)
    mask = _normalize_mask(mask, clamp_val=clamp_val)

    spatial_dims = list(range(2, primitive.dim()))
    prim_empty = ~primitive.any(dim=spatial_dims)
    mask_empty = ~(mask > 0).any(dim=spatial_dims)
    both_populated = ~prim_empty & ~mask_empty
    countable = ~(prim_empty & mask_empty) if penalize_empty else both_populated

    PG = (mask * primitive).sum(dim=spatial_dims)
    sum_P = mask.sum(dim=spatial_dims)
    sum_G = primitive.sum(dim=spatial_dims)
    # sign(G) = G when G is binary
    P_sign_G = (mask * primitive).sum(dim=spatial_dims)  # = PG when G is binary

    c = PG / P_sign_G.clamp_min(1e-12)
    denom = (c * sum_G + sum_P).clamp_min(1e-12)
    cdc_val = PG / denom

    score = torch.where(countable, cdc_val, torch.zeros_like(cdc_val)).sum()
    instances = countable.sum()
    return float(score), int(instances)


def epg(primitive, mask, device="cpu", penalize_empty=True, clamp_val=None):
    """
    Energy-Based Pointing Game.

    Formula: EPG = sum(M * (G > 0)) / sum(M)

    Fraction of total mask energy that falls inside the GT region.
    Very forgiving of subset matches: if a method highlights only part of the
    GT region but nothing outside it, EPG will still be high.

    Parameters:
        primitive (torch.Tensor): Binary GT mask, shape (batch, channels, len).
        mask (torch.Tensor): Continuous concept mask (non-negative), same shape.
        device (str): Computation device.
        penalize_empty (bool): If True, asymmetric empty cases count as score 0. Default True.
        clamp_val (float, optional): Dataset-level clamp for normalisation (CENDRE).

    Returns:
        tuple: (score: float, instances: int)
    """
    primitive = primitive.float().to(device)
    mask = mask.float().to(device)
    if mask.dim() + 1 == primitive.dim():
        mask = mask.unsqueeze(1)
    mask = _normalize_mask(mask, clamp_val=clamp_val)

    spatial_dims = list(range(2, primitive.dim()))
    prim_empty = ~primitive.any(dim=spatial_dims)
    mask_empty = ~(mask > 0).any(dim=spatial_dims)
    both_populated = ~prim_empty & ~mask_empty
    countable = ~(prim_empty & mask_empty) if penalize_empty else both_populated

    energy_in_gt = (mask * (primitive > 0).float()).sum(dim=spatial_dims)
    total_energy = mask.sum(dim=spatial_dims).clamp_min(1e-12)
    epg_val = energy_in_gt / total_energy

    score = torch.where(countable, epg_val, torch.zeros_like(epg_val)).sum()
    instances = countable.sum()
    return float(score), int(instances)


def soft_precision(primitive, mask, device="cpu", penalize_empty=True, clamp_val=None):
    """
    Soft Precision.

    Formula: sum(M * G) / sum(M)

    "What fraction of the predicted energy is in the GT region?"
    When M >= 0 and G is binary, equivalent to EPG.
    A correct subset with no false positives scores 1.0.

    When penalize_empty=True:
    - prim empty, mask populated → score 0 (false alarm: activating where no GT exists)
    - prim populated, mask empty → score 0 (precision undefined, treated as failure)

    Parameters:
        primitive (torch.Tensor): Binary GT mask, shape (batch, channels, len).
        mask (torch.Tensor): Continuous concept mask (non-negative), same shape.
        device (str): Computation device.
        penalize_empty (bool): If True, asymmetric empty cases count as score 0. Default True.
        clamp_val (float, optional): Dataset-level clamp for normalisation (CENDRE).

    Returns:
        tuple: (score: float, instances: int)
    """
    primitive = primitive.float().to(device)
    mask = mask.float().to(device)
    if mask.dim() + 1 == primitive.dim():
        mask = mask.unsqueeze(1)
    mask = _normalize_mask(mask, clamp_val=clamp_val)

    spatial_dims = list(range(2, primitive.dim()))
    prim_empty = ~primitive.any(dim=spatial_dims)
    mask_empty = ~(mask > 0).any(dim=spatial_dims)
    both_populated = ~prim_empty & ~mask_empty
    countable = ~(prim_empty & mask_empty) if penalize_empty else both_populated

    PG = (mask * primitive).sum(dim=spatial_dims)
    sum_P = mask.sum(dim=spatial_dims).clamp_min(1e-12)
    prec = PG / sum_P

    score = torch.where(countable, prec, torch.zeros_like(prec)).sum()
    instances = countable.sum()
    return float(score), int(instances)


def soft_recall(primitive, mask, device="cpu", penalize_empty=True, clamp_val=None):
    """
    Soft Recall.

    Formula: sum(M * G) / sum(G)

    "How much of the GT region is covered by the mask?"
    Low recall + high precision = shortcut learning (correct subset only).

    When penalize_empty=True:
    - prim populated, mask empty → score 0 (method missed the GT entirely)
    - prim empty, mask populated → score 0 (recall undefined, treated as failure)

    The shortcut learning interpretation is preserved: high precision + low recall
    still indicates the method finds a correct subset. The penalty only adds
    an additional failure mode (complete miss) that also produces low recall.

    Parameters:
        primitive (torch.Tensor): Binary GT mask, shape (batch, channels, len).
        mask (torch.Tensor): Continuous concept mask (non-negative), same shape.
        device (str): Computation device.
        penalize_empty (bool): If True, asymmetric empty cases count as score 0. Default True.
        clamp_val (float, optional): Dataset-level clamp for normalisation (CENDRE).

    Returns:
        tuple: (score: float, instances: int)
    """
    primitive = primitive.float().to(device)
    mask = mask.float().to(device)
    if mask.dim() + 1 == primitive.dim():
        mask = mask.unsqueeze(1)
    mask = _normalize_mask(mask, clamp_val=clamp_val)

    spatial_dims = list(range(2, primitive.dim()))
    prim_empty = ~primitive.any(dim=spatial_dims)
    mask_empty = ~(mask > 0).any(dim=spatial_dims)
    both_populated = ~prim_empty & ~mask_empty
    countable = ~(prim_empty & mask_empty) if penalize_empty else both_populated

    PG = (mask * primitive).sum(dim=spatial_dims)
    sum_G = primitive.sum(dim=spatial_dims).clamp_min(1e-12)
    rec = PG / sum_G

    score = torch.where(countable, rec, torch.zeros_like(rec)).sum()
    instances = countable.sum()
    return float(score), int(instances)


def rma(primitive, mask, device="cpu", penalize_empty=True, clamp_val=None):
    """
    Relevance Mass Accuracy (Arras et al. 2022).

    Formula: RMA = sum(max(M, 0) * (G > 0)) / sum(max(M, 0))

    Only considers positive attributions. Equivalent to EPG/soft_precision
    when masks are already non-negative (as in CENDRE). Included for
    completeness and for masks from sources that may contain negative values.

    Parameters:
        primitive (torch.Tensor): Binary GT mask, shape (batch, channels, len).
        mask (torch.Tensor): Continuous concept mask, same shape.
        device (str): Computation device.
        penalize_empty (bool): If True, asymmetric empty cases count as score 0. Default True.
        clamp_val (float, optional): Dataset-level clamp for normalisation (CENDRE).

    Returns:
        tuple: (score: float, instances: int)
    """
    primitive = primitive.float().to(device)
    mask = mask.float().to(device)
    if mask.dim() + 1 == primitive.dim():
        mask = mask.unsqueeze(1)
    mask = _normalize_mask(mask, clamp_val=clamp_val)

    spatial_dims = list(range(2, primitive.dim()))
    prim_empty = ~primitive.any(dim=spatial_dims)
    mask_empty = ~(mask > 0).any(dim=spatial_dims)
    both_populated = ~prim_empty & ~mask_empty
    countable = ~(prim_empty & mask_empty) if penalize_empty else both_populated

    pos_in_gt = (mask * (primitive > 0).float()).sum(dim=spatial_dims)
    sum_pos = mask.sum(dim=spatial_dims).clamp_min(1e-12)
    rma_val = pos_in_gt / sum_pos

    score = torch.where(countable, rma_val, torch.zeros_like(rma_val)).sum()
    instances = countable.sum()
    return float(score), int(instances)


def dst_eclad(primitive, mask, max_dist=None, device="cpu"):
    """
    Compute the one-way distance between a primitive mask and a concept mask.

    This function calculates the Euclidean distance between two binary masks,
    representing a "primitive" and a "concept". The distance measures how well
    the primitive aligns with the concept. If one is empty while the other is
    populated, a maximum penalty distance is applied.

    Parameters:
        primitive (torch.Tensor): Binary mask representing the primitive. shape (batchsize, channels, len)
        mask (torch.Tensor): Binary mask representing the concept.
        max_dist (float, optional): Precomputed maximum possible distance. If None,
                                    it is computed within the function.
        device (str, optional): Computation device, default is "cuda".

    Returns:
        tuple:
            - dist (float): The computed one-way distance.
            - instances (int): The count of valid primitive-concept pairs.
    """

    # (1) Ensure the inputs are boolean tensors and move them to the correct device
    primitive = primitive.bool().to(device)
    mask = mask.bool().to(device)

    if mask.dim()+1==primitive.dim(): #if prim has channel, but mask doesnt
        mask = mask.unsqueeze(1)

    # (2) Determine the input length (last dimension of the mask) #TODO adapt here when addtion trafo support
    input_len = mask.shape[-1]

    # (3) Compute max_dist if not provided
    if max_dist is None:
        extreme = torch.ones(input_len, device=device)
        extreme[0] = 0  # Ensures distance computation is meaningful
        max_dist = torch.sum(euclidean_distance_transform(extreme)) / (input_len**2)

    # (4) Identify empty masks for classification
    primitive_empty = ~primitive.any(dim=list(range(2, primitive.dim())))
    mask_empty = ~mask.any(dim=list(range(2, mask.dim())))

    # (5) Initialize distance and instance counters
    dist = 0.0
    instances = 0

    # (6) Case 1: Primitive mask is empty but concept mask is populated → Apply max_dist penalty
    dist += torch.where(primitive_empty & ~mask_empty, max_dist, 0).sum()
    instances += torch.where(primitive_empty & ~mask_empty, 1, 0).sum()

    # (7) Case 2: Concept mask is empty but primitive mask is populated → Apply max_dist penalty
    dist += torch.where(mask_empty & ~primitive_empty, max_dist, 0).sum()
    instances += torch.where(mask_empty & ~primitive_empty, 1, 0).sum()

    # (8) Case 3: Both masks are populated → Compute Euclidean distance transforms
    d1 = torch.sum(primitive * euclidean_distance_transform(~mask,ndim = primitive.dim()-2), dim=list(range(2, primitive.dim())))
    dist += torch.where(
        ~mask_empty & ~primitive_empty, d1, 0).sum()/ (input_len**2)
    d2 = torch.sum(mask * euclidean_distance_transform(~primitive, ndim = primitive.dim()-2), dim=list(range(2, primitive.dim())))
    dist += torch.where(
        ~mask_empty & ~primitive_empty, d2, 0).sum()/ (input_len**2)

    # (9) Count valid primitive-concept alignments
    instances += torch.where(~mask_empty & ~primitive_empty, 1, 0).sum()

    # (10) Return total distance and number of instances
    return dist, instances.sum()


###############################################################################
# Metric registry
###############################################################################

METRICS = {
    "dst_eclad":      {"fn": dst_eclad,      "higher_is_better": False, "maskMode": "binary"},
    "soft_iou":       {"fn": soft_iou,       "higher_is_better": True,  "maskMode": "gradient"},
    "soft_dice":      {"fn": soft_dice,      "higher_is_better": True,  "maskMode": "gradient"},
    "cdc":            {"fn": cdc,            "higher_is_better": True,  "maskMode": "gradient"},
    "epg":            {"fn": epg,            "higher_is_better": True,  "maskMode": "gradient"},
    "soft_precision": {"fn": soft_precision, "higher_is_better": True,  "maskMode": "gradient"},
    "soft_recall":    {"fn": soft_recall,    "higher_is_better": True,  "maskMode": "gradient"},
    "rma":            {"fn": rma,            "higher_is_better": True,  "maskMode": "gradient"},
}

