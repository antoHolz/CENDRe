# Add the parent directory to the Python path (because we are executing hydra from within runs)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import logging
from lightning import seed_everything
import matplotlib.pyplot as plt
import numpy as np

from modules.utils.hydraqol import run_decorator
from modules.loader.loaders import trainScaler

# Registering the config path with Hydra
@hydra.main(config_path="../data/config", config_name="train_model", version_base="1.3")
@run_decorator
def main(cfg: DictConfig) -> None:

    ##############################
    # Preliminaries
    ##############################
    logger = logging.getLogger("sanity")
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

    # Instantiate datasets with transforms
    train_dataset = hydra.utils.instantiate(cfg.dataset.train, transform=train_transform, pretransform=pretransform)
    valid_dataset = hydra.utils.instantiate(cfg.dataset.val, transform=valid_transform, pretransform=pretransform)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test, transform=test_transform, pretransform=pretransform)

    # Instantiate data loaders
    train_loader, valid_loader, test_loader = hydra.utils.instantiate(cfg.loader.object, dataset_train=train_dataset, dataset_val=valid_dataset, dataset_test=test_dataset, task=cfg.dataset.task)

    ##############################
    # Dataset Info
    ##############################
    logger.info(f"Dataset: {cfg.dataset.name}")
    logger.info(f"Task: {cfg.dataset.task}")
    logger.info(f"n_channels: {cfg.dataset.n_channels}, n_classes: {cfg.dataset.n_classes}")
    logger.info(f"Splits — train: {len(train_dataset)}, val: {len(valid_dataset)}, test: {len(test_dataset)}")

    # Grab a batch and log shapes
    batch = next(iter(train_loader))
    x, y = batch['x'], batch['y']
    logger.info(f"Batch x shape: {list(x.shape)}, y shape: {list(y.shape)}")
    assert x.shape[1] == cfg.dataset.n_channels, f"Expected {cfg.dataset.n_channels} channels, got {x.shape[1]}"

    ##############################
    # Plot examples
    ##############################
    n_examples = min(8, x.shape[0])
    n_channels = x.shape[1]
    channel_names = list(cfg.dataset.xcols) if len(cfg.dataset.xcols) == n_channels else [f"ch{i}" for i in range(n_channels)]

    fig, axes = plt.subplots(2, 4, figsize=(20, 6), sharex=True)
    fig.suptitle(f"{cfg.dataset.name} — train examples", fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i >= n_examples:
            ax.axis('off')
            continue
        sample_x = x[i].numpy()  # (C, L)
        sample_y = y[i].numpy()
        for ch in range(n_channels):
            ax.plot(sample_x[ch], linewidth=0.7, label=channel_names[ch])
        label_str = str(sample_y.item()) if sample_y.ndim == 0 else str(sample_y.tolist())
        ax.set_title(f"y={label_str}", fontsize=9)
        ax.tick_params(labelsize=7)

    # Add channel legend if multi-channel
    if n_channels > 1:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=min(n_channels, 12), fontsize=7)
        fig.subplots_adjust(bottom=0.15)

    fig.tight_layout(rect=[0, 0.08 if n_channels > 1 else 0, 1, 0.95])
    fig.savefig(save_dir / "sanity_examples.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved example plot to {save_dir / 'sanity_examples.png'}")

if __name__ == '__main__':
    main()