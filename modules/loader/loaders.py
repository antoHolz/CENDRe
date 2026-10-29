import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from typing import Any, Callable, Optional, Tuple, Union, List, Dict
from sklearn.preprocessing import StandardScaler
from modules.dataset import HFDataset, npymapDataset

import hydra


def create_dataloaders(
    dataset_train: Optional[Any] = None,
    dataset_val: Optional[Any] = None,
    dataset_test: Optional[Any] = None,
    train_size: float = 0.8,
    test_size: float = 0.0,
    batch_size: int = 32,
    balanced: bool = True,
    task: str = "classification",
    multiplier: int = 1,
    seed: int = 0,
    num_workers: int = 4,
    drop_last=True,
    persistent_workers=False,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates training and validation DataLoaders from the provided datasets, with options for balancing class distribution,
    extending dataset size via a multiplier, and a seed for reproducibility.

    Args:
        dataset_train (Optional[Any]): The dataset to be used for training. If None, it will be split from the combined dataset.
        dataset_val (Optional[Any]): The dataset to be used for validation. If None, it will be split from the combined dataset.
        train_size (float): The proportion of the dataset to include in the training split.
        batch_size (int): Number of samples per batch to load.
        balanced (bool): Whether to balance the DataLoader in terms of class distribution.
        task (str): The type of task, such as 'classification', 'multilabel classification' or 'regression', which may influence how the data is processed.
        multiplier (int): Factor to multiply the dataset size by, useful for augmenting the dataset for extended training.
        seed (int): Seed for random number generators to ensure reproducibility of data splitting and shuffling.
        num_workers (int): Number of subprocesses to use for data loading.
        **kwargs: Additional keyword arguments passed to the DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training and testing DataLoaders.
    """
    # Create ImageFolder with train_transform
    if len(dataset_train) == len(dataset_val) and len(dataset_train) == len(
        dataset_test
    ):  # TODO: sloppy redo
        # Splitting dataset into train and test
        len_train = int(len(dataset_train) * train_size)
        len_test = int(len(dataset_train) * test_size)
        len_val = len(dataset_train) - len_train - len_test
        dataset_train, _, _ = random_split(
            dataset_train,
            [len_train, len_val, len_test],
            generator=torch.Generator().manual_seed(seed),
        )
        _, dataset_val, _ = random_split(
            dataset_val,
            [len_train, len_val, len_test],
            generator=torch.Generator().manual_seed(seed),
        )
        if len_test > 0:
            _, _, dataset_test = random_split(
                dataset_test,
                [len_train, len_val, len_test],
                generator=torch.Generator().manual_seed(seed),
            )
            if batch_size>len(dataset_test): drop_last = False
        else:
            dataset_test = None
    else:  # TODO clean this up, patch solution
        dataset_train, _, _ = random_split(
            dataset_train, [1, 0, 0], generator=torch.Generator().manual_seed(seed)
        )
        _, dataset_val, _ = random_split(
            dataset_val, [0, 1, 0], generator=torch.Generator().manual_seed(seed)
        )
        
        if len_test > 0:
            _, _, dataset_test = random_split(
                dataset_test, [0, 0, 1], generator=torch.Generator().manual_seed(seed)
            )
            if batch_size>len(dataset_test): drop_last = False
        else:
            dataset_test = None

    if batch_size>len(dataset_val) or batch_size>len(dataset_train): drop_last = False
        
    # Creating DataLoaders
    if balanced and task == "classification":
        # Count number of instances per class to balance the dataset
        if isinstance(dataset_train.dataset,HFDataset):
            targets = [sample['y'] for sample in dataset_train]
        elif isinstance(dataset_train.dataset,npymapDataset) and not dataset_train.dataset.withId:
            targets = [label for _, label in dataset_train]
        elif isinstance(dataset_train.dataset,npymapDataset):
            targets = [label for _, label, _ in dataset_train]
        else:
            raise ValueError("Unsupported dataset type for balancing.")

        targets_t = torch.cat([_to_1d_long_cpu(t) for t in targets], dim=0)
        class_count = np.bincount(targets_t)
        class_weights = 1.0 / class_count
        sample_weights = class_weights[targets_t]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=int(len(sample_weights) * multiplier),
            replacement=True,
        )

        loader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            persistent_workers=persistent_workers
        )
    else:
        loader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=drop_last, persistent_workers=persistent_workers
        )

    loader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=drop_last, persistent_workers=persistent_workers
    )
    if dataset_test == None:
        loader_test = None
    else:
        loader_test = DataLoader(
            dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=drop_last, persistent_workers=persistent_workers
        )
    return loader_train, loader_val, loader_test


def trainScaler(cfg_dataset, cfg_dataloader):
    """
    Computes the mean and standard deviation of the training dataset if they are not already provided.

    Args:
        cfg_dataset: A configuration object containing dataset settings. Expected to have `mean`, `std`, and `train` attributes.
        cfg_dataloader: A configuration object containing dataloader settings. Expected to have an `object` attribute with `batch_size` and `num_workers`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - A tensor representing the mean of the dataset.
            - A tensor representing the standard deviation of the dataset.
    """
    if cfg_dataset.mean is None or cfg_dataset.std is None and len(cfg_dataset.mean)==len():
        # Instantiate the training dataset
        pretransform = hydra.utils.instantiate(cfg_dataset.pretransform)
        train_dataset = hydra.utils.instantiate(cfg_dataset.train, pretransform=pretransform)

        # Create a DataLoader to iterate over the dataset
        loader_train = DataLoader(
            train_dataset,
            batch_size=cfg_dataloader.object.batch_size,
            shuffle=False,
            num_workers=cfg_dataloader.object.num_workers,
        )

        n_channels = cfg_dataset.n_channels
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        # Compute mean and standard deviation using a partial fit approach
        for batch in loader_train:
            if isinstance(batch, dict):
                x,y = batch["x"], batch["y"]
                y = y.numpy().reshape(-1, y.shape[1] if y.ndim > 1 else 1)
            else:
                x, _ = batch

            x = x.to("cpu")
            x = np.swapaxes(x, 1, -1)
            x = np.reshape(x, [-1, n_channels])

            scaler_x.partial_fit(x.numpy())
            if isinstance(batch, dict):
                scaler_y.partial_fit(y)

        mean = scaler_x.mean_.tolist()
        std = np.sqrt(scaler_x.var_).tolist()
        if isinstance(batch, dict):
            mean_y = scaler_y.mean_.tolist()
            std_y = np.sqrt(scaler_y.var_).tolist()
            mean = mean + mean_y
            std = std + std_y
        
        if isinstance(batch, dict):
            all_cols = cfg_dataset.xcols + cfg_dataset.ycols
            mean = {name:val for (name, val) in zip(all_cols, mean)}
            std = {name:val for (name, val) in zip(all_cols, std)}
        return mean, std

    return cfg_dataset.mean, cfg_dataset.std



def _to_1d_long_cpu(x):
    # Accept: torch tensor, numpy scalar/array, python int/list, etc.
    if torch.is_tensor(x):
        t = x.detach()
        if t.is_cuda:
            t = t.cpu()
    else:
        t = torch.as_tensor(x)
    return t.reshape(-1).to(torch.long).cpu()