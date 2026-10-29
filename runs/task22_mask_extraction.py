import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import torch
from lightning import seed_everything
from modules.loader.loaders import trainScaler
from cel.utils.reporting import sorted_inputs, sorted_inputs_regression,plot_concepts_report_from_HF
from cel.utils.maskgen import extract_ConceptMasks
from modules.utils.seeds import seed_everything
from modules.utils.hydraqol import run_decorator
from datasets import Dataset
from os.path import isfile
import json
import pandas as pd
import os


# Registering the config path with Hydra
@hydra.main(config_path="../data/config", config_name="CE_masks", version_base="1.3")
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
    train_loader, _, _ = hydra.utils.instantiate(
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

    ##############################
    # Actual Task: CE Training Loop (Fit ECLAD if appropriate)
    ##############################
    if not fitExists:
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
    # Actual Task: Mask generation (so far only implemented for regression models!!) 
    ##############################
    data = {'dataset':cfg.dataset.name, 'modelname': cfg.name, 
            'out_analysed':cfg.CE.multilabel_out, 'concept_nr': CE.n_concepts,
            'partitions':cfg.partitions, 'samples_per_partition':cfg.samples_per_partition}
    
    outdir = f'{cfg.save_dir_base}'
    outdir_model = f"{cfg.save_dir}" 

    maskExists = all([isfile(f"{outdir_model}/bin{bucket}_masks") for bucket in range(data['partitions'])])
    if cfg.mode == "force" or not maskExists:
        samples_x, samples_y, samples_id = sorted_inputs_regression(train_loader,samples_per_partition=data['samples_per_partition'], out_range=cfg.partition_range,
                                                                        partitions=data['partitions'],int_out=data['out_analysed'],DS_cfg=cfg.dataset, astensor=False)
        extract_ConceptMasks(CE, samples_x, samples_y, samples_id, cfg.dataset, outdir=outdir_model, modelname=data['modelname'])
        
        # For loading outside of CEL
        with open(f"{outdir}/{data['dataset']}/{data['dataset']}.yaml", 'w') as f:
            OmegaConf.save(config=cfg.dataset, f=f)
        with open(f"{outdir_model}/importances_{str(CE.n_concepts)}.json", "w") as fp:
            json.dump(CE.importances, fp)
        output_path = f"{outdir}/master.csv"
        df = pd.DataFrame(data, index=[0])
        df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
        
        

    if cfg.plot: 
        for bucket in range(data['partitions']):
            save_path = f"{outdir_model}/bin{bucket}_masks"
            ds = Dataset.load_from_disk(save_path)
            if ds.num_rows>0:
                plot_concepts_report_from_HF(ds,CE.importances,cfg.dataset, rows=[i for i in range(5)], save_path=save_path)



if __name__ == "__main__":
    main()
