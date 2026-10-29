# Add the parent directory to the Python path (because we are executing hydra from within runs)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import json, time, shutil, logging, gc, os
from lightning import seed_everything

from modules.training.training import lightning_train_model as train_model
from modules.utils.count_params import count_parameters
from modules.utils.partialload import load_partial_state_dict
from modules.utils.hydraqol import run_decorator
from modules.loader.loaders import trainScaler

from os.path import isfile
from captum.attr import  Saliency, InputXGradient, IntegratedGradients, FeatureAblation, GuidedBackprop, GuidedGradCam, Deconvolution, IntegratedGradients, DeepLift, DeepLiftShap, GradientShap, LayerIntegratedGradients
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# Registering the config path with Hydra
@hydra.main(config_path="../data/config", config_name="train_model", version_base="1.3")
@run_decorator
def main(cfg: DictConfig) -> None:

    ##############################
    # Preliminaries
    ##############################
    logger = logging.getLogger("training")
    save_dir = Path(cfg.save_dir)

    # Set the random seed for reproducibility
    seed_everything(cfg.seed)

    # Select the device for computation (CUDA, MPS, or CPU)
    device = "cuda" if (cfg.training.device=="cuda" and torch.cuda.is_available()) else "cpu"

    # Train scaler if necessary
    if cfg.transforms.standardize:
        cfg.dataset.mean, cfg.dataset.std = trainScaler(cfg.dataset, cfg.loader)
    
    ##############################
    # Object Instantiation
    ##############################

    # Instantiate transforms for training and testing
    pretransform = hydra.utils.instantiate(cfg.dataset.pretransform)
    train_transform = hydra.utils.instantiate(cfg.transforms.train)
    valid_transform = hydra.utils.instantiate(cfg.transforms.val)
    test_transform = hydra.utils.instantiate(cfg.transforms.test)
    # optional, for forecasting:
    post_cfg = OmegaConf.select(cfg.dataset, "posttransform", default=None)
    posttransform = hydra.utils.instantiate(post_cfg)

    # Instantiate datasets with transforms
    train_dataset = hydra.utils.instantiate(cfg.dataset.train, transform=train_transform, pretransform=pretransform, posttransform=posttransform)
    valid_dataset = hydra.utils.instantiate(cfg.dataset.val, transform=valid_transform, pretransform=pretransform, posttransform=posttransform)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test, transform=test_transform, pretransform=pretransform, posttransform=posttransform)

    # Instantiate data loaders
    train_loader, valid_loader, test_loader =  hydra.utils.instantiate(cfg.loader.object, dataset_train=train_dataset, dataset_val=valid_dataset, dataset_test=test_dataset, task=cfg.dataset.task) 

    logger.info(f"Dataset splits, train: {len(train_dataset)}, valid: {len(valid_dataset)}, test: {len(test_dataset)}")

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model.object)
    n_params = count_parameters(model)
    logger.info(f"Training: {cfg.name} (params: {n_params:,})".replace(",", " "))

    # Load Pretrained Checkpoint (if exists)
    if (cfg.pretrain_checkpoint is not None) and Path(cfg.pretrain_checkpoint).exists():
        mismatch_over_union_layers, mismatch_over_union_params = load_partial_state_dict(model, cfg.pretrain_checkpoint)
        logger.info(f"Loaded pre-trained model from {cfg.pretrain_checkpoint}: {1.0 - mismatch_over_union_layers:.2f} -- {1.0 - mismatch_over_union_params:.2f}")
    
    ##############################
    # Actual Task: Training Loop
    ##############################
    # Training loop 
    PATH=f'{save_dir}/best.pth'
    print(PATH)

    if isfile(PATH) and not cfg.mode=='force':
        model.load_state_dict(torch.load(PATH, weights_only=True))
    else: 
        results = train_model(model, cfg, train_loader, valid_loader, test_loader)
        ##############################
        # Saving Results
        ##############################
        if int(os.environ.get("SLURM_PROCID", 0)) == 0:
            # Save test metrics
            with open(save_dir / "training_results.json", "w") as f:
                json.dump(results, f, indent=4)

            # Empty CUDA cache if GPU is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Perform garbage collection
            gc.collect()
        
    model.eval()

    batch =  next(iter(train_loader))
    if isinstance(batch, dict):
        inputs, outputs = batch["x"], batch["y"]
    else:
        inputs, outputs = batch

    device='cpu'
    model.to(device)
    model.eval()
    inputs = inputs.to(device)
    outputs = outputs.to(device)
    y_hat = model(inputs.to(device))

    true_seq = torch.cat([inputs, outputs[:,-cfg.dataset.pred_len:]], dim=1)
    pred_seq = torch.cat([inputs[:,:(cfg.dataset.seq_len-cfg.dataset.label_len)], y_hat], dim=1)

    # Config
    n_analysed = 4
    len_out = outputs.shape[1]
    len_in = inputs.shape[1]
    patch_len = cfg.model.object.patch_len
    n_patches = len_out // patch_len
    save_dir = Path(cfg.save_dir) / "attribution"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Explain methods
    explainers = [
        FeatureAblation(model),
        IntegratedGradients(model),
        Saliency(model),
        Deconvolution(model),
        DeepLift(model),
    ]
    # Methods whose Captum attribute call is "batchwise" in your scripts
    batchwise_methods = {"Feature Ablation"}

    # ----------------------
    # Core: compute all attributions ONCE per explainer
    # Produces attr_full with shape [B, L_in, C, T_out] for uniform downstream use.
    # ----------------------
    def compute_attr_full(explainer, x, T_out: int) -> torch.Tensor:
        """
        Unifies attribution across targets:
        - For non-batchwise methods: stack per-target attributions along last dim.
        - For batchwise ones (e.g., Feature Ablation): replicate across targets.
        Returns:
        attr_full: [B, L_in, C, T_out]
        """
        name = explainer.get_name()
        if name in batchwise_methods:
            # [B, L_in, C]
            attr = explainer.attribute(x)
            # replicate across outputs so we can aggregate by patches, etc.
            attr_full = torch.permute(attr.unsqueeze(-1).view(x.shape[0],-1, x.shape[1], x.shape[2]),(0,2,3,1))
        else:
            per_t = []
            for t in range(T_out):
                per_t.append(explainer.attribute(x, target=int(t)))  # [B, L_in, C]
            attr_full = torch.stack(per_t, dim=-1)  # [B, L_in, C, T_out]
        return attr_full


    # ----------------------
    # Reductions (no model calls)
    # ----------------------
    def reduce_sum_over_targets(attr_full: torch.Tensor) -> torch.Tensor:
        """[B, L_in, C, T] -> [B, L_in, C]"""
        return attr_full.sum(dim=-1)

    def reduce_patch_means(attr_full: torch.Tensor, patch_len: int) -> torch.Tensor:
        """
        Mean attribution per patch (over its target timesteps).
        Input : [B, L_in, C, T_out]
        Output: [B, L_in, C, n_patches]
        """
        B, L_in, C, T = attr_full.shape
        n_patches = T // patch_len
        T_used = n_patches * patch_len
        trimmed = attr_full[..., :T_used]                                     # [B, L_in, C, n_patches*patch_len]
        reshaped = trimmed.view(B, L_in, C, n_patches, patch_len)             # [B, L_in, C, n_patches, patch_len]
        return reshaped.mean(dim=-1)                                          # [B, L_in, C, n_patches]

    # ----------------------
    # Plotting helpers (re-using your visuals)
    # ----------------------
    def _make_rgba_bars(attribution_map: np.ndarray, len_in: int, y_range: tuple) -> LineCollection:
        """Builds the vertical colored bars based on attribution sign and magnitude."""
        time = np.arange(len_in)
        # Normalize to [0,1]
        denom = np.abs(attribution_map).max() + 1e-12
        norm_attr = np.abs(attribution_map) / denom

        base_color_neg = np.array([0, 0.2, 0.9, 1])    # blue
        base_color_pos = base_color_neg

        segments = np.array([[(t, y_range[0]), (t, y_range[1])] for t in time])  # vertical segments
        rgba_colors = np.zeros([len(time), 4])
        rgba_colors[:len_in] = (norm_attr * (attribution_map > 0)).reshape(-1, 1) * np.tile(base_color_pos, (len_in, 1)) + \
                            (norm_attr * (attribution_map < 0)).reshape(-1, 1) * np.tile(base_color_neg, (len_in, 1))
        rgba_colors[:len_in, 3] = norm_attr
        return LineCollection(segments, colors=rgba_colors, linewidths=6)


    def plot_sample_lines(name: str, attr_sum: torch.Tensor):
        """
        Per-sample line overlays using sum-over-targets map.
        attr_sum: [B, L_in, C]
        """
        B = min(n_analysed, attr_sum.shape[0])
        fig, ax = plt.subplots(nrows=B, figsize=(10, 10))
        if B == 1:
            ax = [ax]  # unify indexing

        plt.suptitle(f'{name}')
        for i in range(B):
            input_signal = inputs[i, :, 0].detach().cpu().numpy()
            attribution_map = attr_sum[i, :, 0].detach().cpu().numpy()

            # y-range
            i_max = input_signal.max()
            i_min = input_signal.min()
            pad = (i_max - i_min) * 0.5
            y_range = (i_min - pad, i_max + pad)

            # Colored vertical attributions
            line_collection = _make_rgba_bars(attribution_map, len_in=len_in, y_range=y_range)
            ax[i].add_collection(line_collection)

            # Overlay true & prediction
            t = np.arange(true_seq.shape[1])
            ax[i].plot(t, true_seq[i, :, 0].detach().cpu().numpy(), color="black", linewidth=1.5, zorder=2, label='True')
            ax[i].plot(t, pred_seq[i, :, 0].detach().cpu().numpy(), color="red", linewidth=1.5, zorder=2, label='Pred')

            ax[i].set_xlim(t.min(), t.max())
            ax[i].set_ylim(y_range)
            ax[i].set_title(f'Sample {i+1}')
            ax[i].set_xlabel("t")
            ax[i].set_ylabel("f(t)")
            ax[i].legend()

        plt.tight_layout()
        plt.savefig(save_dir / f"{name}_attribution.png")
        plt.close(fig)


    def plot_patch_panels(name: str, attr_patch_mean: torch.Tensor):
        """
        Patch grid: rows = patches, cols = samples (first n_analysed).
        attr_patch_mean: [B, L_in, C, n_patches]
        """
        B = min(n_analysed, attr_patch_mean.shape[0])
        fig, ax = plt.subplots(nrows=n_patches, ncols=B, figsize=(8*n_analysed, 4*n_patches), squeeze=False)
        plt.suptitle(f'{name}')

        for k in range(B):                 # column = sample
            for p in range(n_patches):     # row = patch
                attribution_map = attr_patch_mean[k, :, 0, p].detach().cpu().numpy()
                input_signal   = inputs[k, :, 0].detach().cpu().numpy()

                i_max = input_signal.max()
                i_min = input_signal.min()
                pad = (i_max - i_min) * 0.5
                y_range = (i_min - pad, i_max + pad)

                line_collection = _make_rgba_bars(attribution_map, len_in=len_in, y_range=y_range)
                ax[p, k].add_collection(line_collection)

                t = np.arange(true_seq.shape[1])
                ax[p, k].plot(t, true_seq[k, :, 0].detach().cpu().numpy(), color="black", linewidth=1.5, zorder=2, label='True')
                ax[p, k].plot(t, pred_seq[k, :, 0].detach().cpu().numpy(), color="red", linewidth=1.5, zorder=2, label='Pred')

                ax[p, k].set_xlim(t.min(), t.max())
                ax[p, k].set_ylim(y_range)
                if p == 0: ax[p, k].set_title(f'Sample {k+1}')
                if p == (B-1): ax[p, k].set_xlabel("t")
                if k == 0: ax[p, k].set_ylabel(f'Patch {p+1}')
                ax[p, k].legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{name}_patchAttribution.png")
        plt.close(fig)


    def plot_patch_position_hist(name: str, attr_sum: torch.Tensor):
        """
        Histogram over positions within a patch (i % patch_len).
        attr_sum: [B, L_in, C] -> sum |.| over batch & channel, then bin along i%patch_len
        """
        # Aggregate |attribution| over batch and channels
        vals = attr_sum.abs().sum(dim=(0, 2))  # [L_in]
        bins = torch.zeros(patch_len, device=vals.device, dtype=vals.dtype)
        idxs = torch.arange(len_in, device=vals.device)
        bins.index_add_(0, idxs % patch_len, vals)

        fig = plt.figure(figsize=(8, 4))
        plt.bar(range(patch_len), bins.detach().cpu().numpy())
        plt.title(f'{name}')
        plt.xlabel("Position in patch (i % patch_len)")
        plt.ylabel("Aggregated |attribution|")
        plt.tight_layout()
        plt.savefig(save_dir / f"{name}_patchidHistogram.png")
        plt.close(fig)


    # ----------------------
    # Run once per explainer and reuse results for all plots
    # ----------------------
    x_batch = inputs[:n_analysed]

    for expl in explainers:
        name = expl.get_name()
        print(name)

        # 1) single compute pass
        attr_full = compute_attr_full(expl, x_batch, T_out=len_out)      # [B, L_in, C, T_out]

        # 2) reductions reused by the three plots
        attr_sum = reduce_sum_over_targets(attr_full)                    # [B, L_in, C]
        attr_patch_mean = reduce_patch_means(attr_full, patch_len)       # [B, L_in, C, n_patches]

        # 3) plots
        plot_sample_lines(name, attr_sum)
        plot_patch_panels(name, attr_patch_mean)
        plot_patch_position_hist(name, attr_sum)



if __name__ == '__main__':
    main()