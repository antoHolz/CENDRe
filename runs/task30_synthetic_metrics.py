import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from distmap import euclidean_distance_transform
from modules.loader.loaders import trainScaler
from modules.transforms.tstransforms import TSResize
from modules.utils.seeds import seed_everything
from cel.utils import DSTMetric, RC, IC, soft_IC, soft_IC_perprim, soft_RC, soft_RCmax, otsu_RC, nmi_alignment
from cel.utils.correctness import METRICS, dst_eclad
from cel.utils.misc import infer_shape
from os.path import isfile
import logging
import numpy as np
import pandas as pd
from modules.utils.hydraqol import run_decorator
from modules.dataset import HFDataset, npymapDataset
from datasets import Dataset



# Registering the config path with Hydra
@hydra.main(config_path="../data/config", config_name="CE_metrics", version_base="1.3")
@run_decorator
def main(cfg: DictConfig) -> None:
    ##############################
    # Preliminaries
    ##############################
    save_dir = Path(cfg.save_dir_model)

    # Set the random seed for reproducibility
    seed_everything(cfg.seed)

    # Select the device for computation (CUDA, MPS, or CPU)
    device = (
        "cuda"
        if (cfg.CE.device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )

    ##############################
    # Object Instantiation
    ##############################
    # Train scaler if necessary
    if cfg.transforms.standardize:
        cfg.dataset.mean, cfg.dataset.std = trainScaler(cfg.dataset, cfg.loader)

    # Instantiate transforms for training and testing
    pretransform = hydra.utils.instantiate(cfg.dataset.pretransform)
    train_transform = hydra.utils.instantiate(cfg.transforms.train)
    valid_transform = hydra.utils.instantiate(cfg.transforms.val)
    test_transform = hydra.utils.instantiate(cfg.transforms.test)

    # Instantiate datasets with transforms
    train_dataset = hydra.utils.instantiate(cfg.dataset.train, transform=train_transform, pretransform=pretransform, withId=True)
    valid_dataset = hydra.utils.instantiate(cfg.dataset.val, transform=valid_transform, pretransform=pretransform, withId=True)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test, transform=test_transform, pretransform=pretransform, withId=True)
    

    # Instantiate data loaders
    train_loader, valid_loader, _ = hydra.utils.instantiate(
        cfg.CE.loader,
        dataset_train=train_dataset,
        dataset_val=valid_dataset,
        dataset_test=test_dataset,
        task=cfg.dataset.task,
    )

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model.object)
    PATH = f"{save_dir}/best.pth"
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()

    # Initialize ECLAD or ECLAD-ts
    CE = hydra.utils.instantiate(cfg.CE.object, model=model)
    fitExists, _ = CE.check_existence(cfg.CE)

    ###############################################
    # CE Training Loop (Fit ECLAD if appropriate) 
    ###############################################
    if not fitExists:
        print("training Explainer")
        CE.fit(train_loader, cfg.CE, device=device)
        CE.save(cfg.CE)
        # Save the configuration
        with open(f"{cfg.CE.save_dir}/config.yaml", "w") as f:
            OmegaConf.save(config=cfg, f=f)
    else:
        CE.load(cfg.CE, device)
        print("loaded Explainer from files")

    ###################################
    # Actual Task: Metric computation
    ###################################

    # --- Pre-guard setup (cheap config lookups) ---
    eval_metrics = list(cfg.eval_metrics) if hasattr(cfg, 'eval_metrics') else ['dst_eclad']
    DEFAULT_ALIGNMENT_METRICS = ["RC", "IC", "sIC", "sICperprim", "sRC", "sRCmax", "oRC", "NMI"]
    alignment_metrics = list(cfg.alignment_metrics) if hasattr(cfg, 'alignment_metrics') else DEFAULT_ALIGNMENT_METRICS
    has_confounders = bool(getattr(cfg.dataset, 'counfoundercols', []))
    suffixes = ['']
    if has_confounders:
        suffixes.append('_conf')

    # Auto-k CE methods (e.g. HDBSCAN at low n_clusters) can collapse to 0 concepts. RC/IC-family
    # metrics only degrade gracefully (try/except -> NaN) for IC/sIC/sICperprim; sRC/sRCmax/oRC/
    # NMI operate on an empty concept dimension and crash. Skip computation entirely and write an
    # all-NaN row instead, mirroring the same "no concepts found" guard CENDRE.fit() already uses.
    if len(CE.importances) == 0:
        print(f"n_concepts=0 for {cfg.CE.save_dir}; writing NaN metrics row and skipping computation.")
        base_names = {
            "RC": ["RC"], "IC": ["IC", "ICA", "ICU"], "sIC": ["sIC"],
            "sICperprim": ["sICperprim"], "sRC": ["sRC"], "sRCmax": ["sRCmax"],
            "oRC": ["oRC", "oRC_thr", "oIC", "oICA", "oICU"], "NMI": ["NMI"],
        }
        nan_metrics = {
            f'{base}_{metric_name}{suffix}': float('nan')
            for metric_name in eval_metrics
            for suffix in suffixes
            for am in alignment_metrics
            for base in base_names.get(am, [am])
        }
        pd.DataFrame(nan_metrics, index=[0]).to_csv(f'{cfg.CE.save_dir}/metrics_{cfg.CE.n_concepts}.csv')
        return

    # Determine which phases to run
    metrics_exist = isfile(f'{cfg.CE.save_dir}/metrics_{str(cfg.CE.n_concepts)}.csv')
    scores_exist = all(
        isfile(f'{cfg.CE.save_dir}/scores_{m}{s}_{str(cfg.CE.n_concepts)}.npy')
        for m in eval_metrics for s in suffixes
    )

    if cfg.mode in ("force", "force-metric"):
        run_phase1 = True   # recompute scores
    elif cfg.mode == "force-correctness":
        run_phase1 = not scores_exist  # load if available, else compute
    elif not metrics_exist and scores_exist:
        run_phase1 = False  # auto: scores exist, just redo Phase 2
    elif not metrics_exist:
        run_phase1 = True   # auto: nothing exists, full run
    else:
        return  # both exist, nothing to do

    # --- Common setup ---
    concept_ids = list(CE.importances.keys())
    n_concepts = len(concept_ids)
    input_shape = infer_shape(valid_loader)
    input_len = input_shape[1:]
    data_channels = input_shape[0]
    if cfg.CE.decorrelated:
        channel_mode = "per_channel"
    elif data_channels > 1:
        channel_mode = "broadcast"
    else:
        channel_mode = "single"
    n_channels = data_channels if channel_mode == "per_channel" else 1

    # max_penalties (needed for RC)
    extreme = torch.ones(input_len, device=device)
    extreme[0] = 0
    dst_max_penalty = float(torch.sum(euclidean_distance_transform(extreme)) / (max(input_len) ** 2))
    max_penalties = {}
    for metric_name in eval_metrics:
        max_penalties[metric_name] = dst_max_penalty if METRICS[metric_name]["fn"] is dst_eclad else 1.0

    # Per-primitive channel mapping — used by soft_IC_perprim to bind each
    # primitive's contribution to its own active channel. Index is the
    # primitive position in maskcols/counfoundercols; value is the channel
    # index in sim's channel axis (or -1 for broadcast primitives without a
    # _<xc> suffix). Reduces to all-zero for single-channel sim.
    xcols_list = list(getattr(cfg.dataset, 'xcols', []))
    def _derive_primitive_channels(maskcol_names):
        if n_channels == 1 or not xcols_list:
            return [0] * len(maskcol_names)
        chans = []
        for pname in maskcol_names:
            ch_idx = -1
            for ci, xc in enumerate(xcols_list):
                if pname.endswith(f'_{xc}'):
                    ch_idx = ci
                    break
            chans.append(ch_idx)
        return chans

    # Negative primitives: for each primitive whose name does NOT contain
    # "background", we also evaluate against 1 - p. Negatives are appended
    # after the originals, preserving original order.
    def _negative_indices(names):
        return [i for i, n in enumerate(names) if "background" not in n.lower()]

    def _augment_with_negatives(names):
        return list(names) + [f"not_{n}" for n in names if "background" not in n.lower()]

    maskcols_aug = _augment_with_negatives(list(cfg.dataset.maskcols))

    primitive_channels_by_suffix = {'': _derive_primitive_channels(maskcols_aug)}
    if has_confounders:
        primitive_channels_by_suffix['_conf'] = _derive_primitive_channels(list(cfg.dataset.counfoundercols))

    # =========================================================
    # Phase 1: Compute or load scores
    # =========================================================
    all_scores = {}

    if not run_phase1:
        # Load pre-computed scores from .npy files (skip expensive CE.eval)
        print('loading pre-computed scores (skipping Phase 1)')
        for metric_name in eval_metrics:
            for suffix in suffixes:
                path = f'{cfg.CE.save_dir}/scores_{metric_name}{suffix}_{str(cfg.CE.n_concepts)}.npy'
                if isfile(path):
                    all_scores[(metric_name, suffix)] = torch.tensor(
                        np.load(path), dtype=torch.float32, device=device
                    )
                else:
                    logging.warning("Missing scores file: %s — skipping", path)
    else:
        # Full computation: load primitives, run CE.eval, accumulate DSTMetric
        primitives = {}
        confounders = {}
        if isinstance(train_dataset, HFDataset):
            ds = Dataset.load_from_disk(cfg.dataset.val.dataset.dataset_path)
            for p, pname in enumerate(cfg.dataset.maskcols):
                primitives[p] = torch.tensor(ds[pname], dtype=torch.float32)
            for p, pname in enumerate(cfg.dataset.counfoundercols):
                confounders[p] = torch.tensor(ds[pname], dtype=torch.float32)
        elif isinstance(train_dataset.dataset, npymapDataset):
            pathlist = sorted(Path(cfg.dataset.eval_path).glob('**/*.npy'))
            for p, path in enumerate(pathlist):
                primitives[p] = torch.tensor(np.load(path), dtype=torch.float32)

        # --- Expand single-channel primitives to multi-channel ---
        # Needed whenever the data has multiple channels, so that broadcast-mode
        # CE methods (ECLAD, ConceptShap, MultiVISION) also get full-channel
        # primitives, not just decorrelated per-channel methods.
        if data_channels > 1:
            xcols = list(getattr(cfg.dataset, 'xcols', []))
            def _expand_masks(masks, maskcol_names):
                for p, pname in enumerate(maskcol_names):
                    ch_idx = -1
                    for ci, xc in enumerate(xcols):
                        if pname.endswith(f'_{xc}'):
                            ch_idx = ci
                            break
                    if ch_idx >= 0:
                        raw = masks[p]
                        if raw.dim() == 2:
                            expanded = torch.zeros(raw.shape[0], data_channels, raw.shape[-1], dtype=raw.dtype)
                            expanded[:, ch_idx, :] = raw
                        elif raw.dim() == 3 and raw.shape[1] == 1:
                            expanded = torch.zeros(raw.shape[0], data_channels, raw.shape[-1], dtype=raw.dtype)
                            expanded[:, ch_idx, :] = raw[:, 0, :]
                        else:
                            continue
                        masks[p] = expanded
            _expand_masks(primitives, list(cfg.dataset.maskcols))
            if confounders:
                _expand_masks(confounders, list(cfg.dataset.counfoundercols))

        # --- Append negative primitives (1 - p), skipping background ---
        # Done after multi-channel expansion so the negative of e.g. `square_X`
        # is `1 - square_mask` on channel X and all-1 on every other channel.
        def _append_negatives(masks, names):
            P_pos = len(masks)
            for offset, idx in enumerate(_negative_indices(names)):
                masks[P_pos + offset] = 1.0 - masks[idx]

        _append_negatives(primitives, list(cfg.dataset.maskcols))

        if len(input_len) == 1:
            resize = TSResize(new_length=torch.tensor(input_len), mode="nearest")
        else:
            raise NotImplementedError

        mask_sets = [('', primitives)]
        if confounders:
            mask_sets.append(('_conf', confounders))

        # --- Build per-concept clamp values for CENDRE gradient masks ---
        # binary_filter[domain]['asclamp'] is a list indexed by concept position.
        # concept_ids are the CE importances keys (strings like '0','1',...) in order.
        gradient_concept_clamps = {}
        if hasattr(CE, 'binary_filter') and CE.binary_filter:
            eval_domain = getattr(cfg.CE, 'domain', [None])[0]
            bf = CE.binary_filter.get(str(eval_domain), {}) if eval_domain is not None else {}
            if not bf:
                # fallback: try first available domain
                bf = next(iter(CE.binary_filter.values()), {})
            asclamp = bf.get('asclamp', None) if isinstance(bf, dict) else None
            if asclamp is not None:
                for c, d in enumerate(concept_ids):
                    idx = int(d)
                    if idx < len(asclamp):
                        cv = float(asclamp[idx])
                        if cv > 0:
                            gradient_concept_clamps[c] = cv

        # --- Create DSTMetric instances ---
        dst_metrics = {}
        for metric_name in eval_metrics:
            metric_info = METRICS[metric_name]
            # Only pass clamps for soft (gradient) metrics — dst_eclad binarizes and doesn't normalise
            clamps = gradient_concept_clamps if metric_info["maskMode"] == "gradient" else {}
            for suffix, masks in mask_sets:
                dst_metrics[(metric_name, suffix)] = DSTMetric(
                    n_primitives=len(masks), n_concepts=n_concepts,
                    data_channels=data_channels, channel_mode=channel_mode,
                    metric_fn=metric_info["fn"],
                    max_penalty=max_penalties[metric_name],
                    concept_clamps=clamps,
                ).to(device)

        maskModes = {METRICS[m]["maskMode"] for m in eval_metrics}

        print('computing metrics')
        # --- Dataloader loop ---
        for i, data in enumerate(valid_loader):
            if isinstance(data, dict):
                x, ids = data["x"], data["id"]
            else:
                x, _, ids = data

            eval_by_maskMode = {}
            for maskMode in maskModes:
                eval_by_maskMode[maskMode], _ = CE.eval(x, device=device, maskMode=maskMode, domain=getattr(cfg.CE, 'domain', [None])[0])

            concept_dicts = {
                mm: {c: resize(eval_masks[int(d)]) for c, d in enumerate(concept_ids)}
                for mm, eval_masks in eval_by_maskMode.items()
            }
            prim_dicts = {
                suffix: {p: resize(masks[p][ids]) for p in range(len(masks))}
                for suffix, masks in mask_sets
            }

            for metric_name in eval_metrics:
                maskMode = METRICS[metric_name]["maskMode"]
                for suffix, _ in mask_sets:
                    dst_metrics[(metric_name, suffix)].update(
                        prim_dicts[suffix], concept_dicts[maskMode]
                    )

            if (i + 1) * cfg.CE.loader.batch_size >= cfg.CE.n_instances:
                break

        # --- Compute and save scores ---
        for metric_name in eval_metrics:
            for suffix, _ in mask_sets:
                scores = dst_metrics[(metric_name, suffix)].compute()
                np.save(
                    f'{cfg.CE.save_dir}/scores_{metric_name}{suffix}_{str(cfg.CE.n_concepts)}.npy',
                    scores.detach().cpu().numpy()
                )
                all_scores[(metric_name, suffix)] = scores

    # =========================================================
    # Phase 2: Compute RC/IC/soft_IC from scores
    # =========================================================
    metrics = {}
    higher_is_better_for_RC = lambda mn: METRICS[mn]["higher_is_better"]
    needs_rc_intermediate = {"RC", "IC", "sIC", "sICperprim"}
    for metric_name in eval_metrics:
        max_penalty = max_penalties[metric_name]
        for suffix in suffixes:
            if (metric_name, suffix) not in all_scores:
                continue
            scores = all_scores[(metric_name, suffix)]

            # RC / IC / sIC / sICperprim share alignments + sim from a single RC('rel') call.
            if any(m in alignment_metrics for m in needs_rc_intermediate):
                rc_val, alignments, sim = RC(
                    scores, 'rel', max_penalty, device=device,
                    higher_is_better=higher_is_better_for_RC(metric_name)
                )
                if "RC" in alignment_metrics:
                    metrics[f'RC_{metric_name}{suffix}'] = rc_val
                if "IC" in alignment_metrics:
                    try:
                        metrics[f'IC_{metric_name}{suffix}'], metrics[f'ICA_{metric_name}{suffix}'], metrics[f'ICU_{metric_name}{suffix}'] = IC(
                            alignments, CE, device=device
                        )
                    except Exception as e:
                        logging.warning("IC failed for %s%s: %s", metric_name, suffix, e)
                        metrics[f'IC_{metric_name}{suffix}'] = float('nan')
                        metrics[f'ICA_{metric_name}{suffix}'] = float('nan')
                        metrics[f'ICU_{metric_name}{suffix}'] = float('nan')
                if "sIC" in alignment_metrics:
                    try:
                        metrics[f'sIC_{metric_name}{suffix}'] = soft_IC(sim, CE, device=device)
                    except Exception as e:
                        logging.warning("soft_IC failed for %s%s: %s", metric_name, suffix, e)
                        metrics[f'sIC_{metric_name}{suffix}'] = float('nan')
                if "sICperprim" in alignment_metrics:
                    try:
                        metrics[f'sICperprim_{metric_name}{suffix}'] = soft_IC_perprim(
                            sim, CE, primitive_channels_by_suffix[suffix], device=device
                        )
                    except Exception as e:
                        logging.warning("soft_IC_perprim failed for %s%s: %s", metric_name, suffix, e)
                        metrics[f'sICperprim_{metric_name}{suffix}'] = float('nan')

            # --- Soft RC (threshold-free) ---
            if "sRC" in alignment_metrics:
                metrics[f'sRC_{metric_name}{suffix}'] = soft_RC(
                    scores, max_penalty, device=device,
                    higher_is_better=higher_is_better_for_RC(metric_name)
                )
            if "sRCmax" in alignment_metrics:
                metrics[f'sRCmax_{metric_name}{suffix}'] = soft_RCmax(
                    scores, max_penalty, device=device,
                    higher_is_better=higher_is_better_for_RC(metric_name)
                )

            # --- Otsu-adaptive RC (also produces oIC/oICA/oICU) ---
            if "oRC" in alignment_metrics:
                otsu_rc, otsu_alignments, _, otsu_thr = otsu_RC(
                    scores, max_penalty, device=device,
                    higher_is_better=higher_is_better_for_RC(metric_name)
                )
                metrics[f'oRC_{metric_name}{suffix}'] = otsu_rc
                metrics[f'oRC_thr_{metric_name}{suffix}'] = otsu_thr
                try:
                    metrics[f'oIC_{metric_name}{suffix}'], metrics[f'oICA_{metric_name}{suffix}'], metrics[f'oICU_{metric_name}{suffix}'] = IC(
                        otsu_alignments, CE, device=device
                    )
                except Exception as e:
                    logging.warning("otsu IC failed for %s%s: %s", metric_name, suffix, e)
                    metrics[f'oIC_{metric_name}{suffix}'] = float('nan')
                    metrics[f'oICA_{metric_name}{suffix}'] = float('nan')
                    metrics[f'oICU_{metric_name}{suffix}'] = float('nan')

            # --- NMI alignment ---
            if "NMI" in alignment_metrics:
                metrics[f'NMI_{metric_name}{suffix}'] = nmi_alignment(
                    scores, max_penalty, device=device,
                    higher_is_better=higher_is_better_for_RC(metric_name)
                )

    # --- Log and save ---
    if not metrics:
        print(f"[WARN] No metrics computed for {cfg.CE.save_dir} — skipping write.")
        return

    logger_version = f"cnr_{cfg.CE.n_concepts}"
    tb_logger = TensorBoardLogger(save_dir=cfg.CE.save_dir, name="metrics", version=logger_version)
    csv_logger = CSVLogger(save_dir=cfg.CE.save_dir, name="metrics", version=logger_version)
    for logger in [tb_logger, csv_logger]:
        logger.log_metrics(metrics, step=0)
        logger.save()
        logger.finalize("success")

    df = pd.DataFrame(metrics, index=[0])
    df.to_csv(f'{cfg.CE.save_dir}/metrics_{str(cfg.CE.n_concepts)}.csv')
    for metric_name in eval_metrics:
        print(f"  {metric_name}: RC={metrics.get(f'RC_{metric_name}', float('nan')):.4f}, IC={metrics.get(f'IC_{metric_name}', float('nan')):.4f}, ICA={metrics.get(f'ICA_{metric_name}', float('nan')):.4f}, ICU={metrics.get(f'ICU_{metric_name}', float('nan')):.4f}, sIC={metrics.get(f'sIC_{metric_name}', float('nan')):.4f}, sICperprim={metrics.get(f'sICperprim_{metric_name}', float('nan')):.4f}, sRC={metrics.get(f'sRC_{metric_name}', float('nan')):.4f}, sRCmax={metrics.get(f'sRCmax_{metric_name}', float('nan')):.4f}, oRC={metrics.get(f'oRC_{metric_name}', float('nan')):.4f}(thr={metrics.get(f'oRC_thr_{metric_name}', float('nan')):.3f}), NMI={metrics.get(f'NMI_{metric_name}', float('nan')):.4f}")

if __name__ == '__main__':
    main()