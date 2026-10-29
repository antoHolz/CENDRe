import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import torch
from lightning import seed_everything
from modules.loader.loaders import trainScaler
from cel.utils.reporting import sorted_inputs, sorted_inputs_regression
from modules.utils.seeds import seed_everything
from modules.utils.hydraqol import run_decorator


# Registering the config path with Hydra
@hydra.main(config_path="../data/config", config_name="train_CE", version_base="1.3")
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
    maskMode = getattr(cfg.CE, 'maskMode', 'gradient')
    fitExists, reportExists = CE.check_existence(cfg.CE)

    ##############################
    # Actual Task: CE Training Loop (Fit ECLAD if appropriate)
    ##############################
    if cfg.mode == "force" or not fitExists:
        print("training Explainer")
        CE.fit(train_loader, cfg.CE, device=device)
        CE.save(cfg.CE)
        # Save the configuration
        with open(f"{cfg.CE.save_dir}/config.yaml", "w") as f:
            OmegaConf.save(config=cfg, f=f)
    else:
        CE.load(cfg.CE, device=device)
        print("loaded Explainer from files")
    
    ##############################
    # Actual Task: Report generation (if appropriate)
    ##############################
    if CE.n_concepts == 0:
        print(f"No concepts found (n_concepts=0); skipping report for {cfg.CE.save_dir}")
    elif cfg.mode == "force" or cfg.mode == "force-report" or not reportExists:
        if cfg.dataset.task == "classification":
            batch = sorted_inputs(
                CE.model, valid_loader, n_classes=cfg.dataset.n_classes, n_repeats=2
            )
        elif cfg.dataset.task == "regression" and cfg.transforms.standardize:
            batch = sorted_inputs_regression(
                valid_loader, partitions=5,samples_per_partition=1, out_range=[-2,2],int_out=cfg.CE.multilabel_out, astensor=True
            )
        else:
            batch = next(iter(valid_loader))
        if isinstance(batch, dict):
            test_inputs, test_truth = batch["x"][:4], batch["y"][:4]
        else:
            test_inputs, test_truth, _ = batch[:4]
        
        # Sampling rate from dataset config (defaults to 1 if absent).
        # If transforms changed the sample length (resize / crop-pad), scale fs
        # proportionally so frequency axes reflect what the model actually sees.
        ds_fs_nominal = getattr(cfg.dataset, 'fs', None) or 1
        try:
            raw_length = len(valid_dataset.dataset[0][cfg.dataset.xcols[0]])
        except (AttributeError, KeyError, TypeError, IndexError):
            raw_length = test_inputs.shape[-1]
        input_length = test_inputs.shape[-1]
        if raw_length != input_length:
            ds_fs = ds_fs_nominal * input_length / raw_length
            print(
                f"[fs] sample length changed by transforms "
                f"({raw_length} -> {input_length}); "
                f"effective fs: {ds_fs_nominal} -> {ds_fs:.4f}"
            )
        else:
            ds_fs = ds_fs_nominal

        for domain in cfg.CE.domain:
            print(f'Plotting report on {domain} domain')
            test_masks, test_outputs = CE.eval(test_inputs,domain=domain, device=device, maskMode=getattr(cfg.CE, 'maskMode', None))

            CE.report(
                test_masks,
                cfg.CE,
                test_inputs,
                test_outputs,
                domain=domain,
                fs=ds_fs,
                test_truth=test_truth,
                DS_cfg=cfg.dataset,
                font_base=5,
                destandardize=cfg.transforms.standardize,
                visualization_mode=getattr(cfg.CE, 'visualization_mode', None),
            )

        # Concept correspondence plot (CENDRE + fourier only)
        if hasattr(CE, 'concept_correspondence') and 'fourier' in cfg.CE.domain:
            print('Plotting concept correspondence')
            cc = CE.concept_correspondence(valid_loader, num_batches=5, device=device)
            CE.plot_CC(cc, cfg.CE, fs=ds_fs, DS_cfg=cfg.dataset)


if __name__ == "__main__":
    main()
