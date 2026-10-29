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

    # Instantiate datasets with transforms
    train_dataset = hydra.utils.instantiate(cfg.dataset.train, transform=train_transform, pretransform=pretransform)
    valid_dataset = hydra.utils.instantiate(cfg.dataset.val, transform=valid_transform, pretransform=pretransform)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test, transform=test_transform, pretransform=pretransform)

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

    return results["last"]["test_loss"]
        

if __name__ == '__main__':
    main()