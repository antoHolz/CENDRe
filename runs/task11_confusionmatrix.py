import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix

from modules.loader.loaders import trainScaler
from modules.utils.seeds import seed_everything


# Registering the config path with Hydra
@hydra.main(config_path="../data/config", config_name="train_model", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ##############################
    # Preliminaries
    ##############################
    save_dir = Path(cfg.save_dir_model if "save_dir_model" in cfg else cfg.save_dir)

    # Set the random seed for reproducibility
    seed_everything(cfg.seed)

    # Select the device for computation (CUDA, MPS, or CPU)
    device = "cuda" if (cfg.training.device == "cuda" and torch.cuda.is_available()) else "cpu"

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
    train_dataset = hydra.utils.instantiate(cfg.dataset.train, transform=train_transform, pretransform=pretransform)
    valid_dataset = hydra.utils.instantiate(cfg.dataset.val, transform=valid_transform, pretransform=pretransform)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test, transform=test_transform, pretransform=pretransform)

    # Instantiate data loaders
    _, valid_loader, test_loader = hydra.utils.instantiate(
        cfg.loader.object,
        dataset_train=train_dataset,
        dataset_val=valid_dataset,
        dataset_test=test_dataset,
        task=cfg.dataset.task,
    )

    eval_loader = test_loader
    eval_split = "test"
    if eval_loader is None:
        eval_loader = valid_loader
        eval_split = "validation"
        print("No test loader available (loader.test_size == 0). Using validation loader for confusion matrix.")

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model.object).to(device)
    PATH = f"{save_dir}/best.pth"
    model.load_state_dict(torch.load(PATH, map_location=device, weights_only=True))
    model.eval()

    
    if cfg.dataset.task != "classification":
        raise ValueError("Confusion matrix is only defined for classification tasks.")

    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for batch in eval_loader:
            if isinstance(batch, dict):
                x = batch["x"]
                y = batch["y"]
            else:
                x, y = batch[:2]

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_targets.append(y.detach().cpu())
            all_predictions.append(preds.detach().cpu())

    targets = torch.cat(all_targets).numpy()
    predictions = torch.cat(all_predictions).numpy()
    
    n_classes = int(cfg.dataset.n_classes)
    labels = np.arange(n_classes)
    cm = confusion_matrix(targets, predictions, labels=labels)
    save_matrix_dir = save_dir / "confusion_matrix"
    save_matrix_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_matrix_dir / "confusion_matrix.npy", cm)
    np.savetxt(save_matrix_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    accuracy = float((targets == predictions).mean())
    with open(save_matrix_dir / "confusion_matrix_info.txt", "w") as f:
        f.write(f"num_samples: {len(targets)}\n")
        f.write(f"accuracy: {accuracy:.6f}\n")

    print(f"Saved confusion matrix to: {save_matrix_dir / 'confusion_matrix.npy'}")
    print(f"Saved confusion matrix CSV to: {save_matrix_dir / 'confusion_matrix.csv'}")
    print(f"Test accuracy from confusion-matrix pass: {accuracy:.6f}")
    print(f"Split used for confusion matrix: {eval_split}")

if __name__ == "__main__":
    main()
