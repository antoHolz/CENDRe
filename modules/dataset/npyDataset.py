import numpy as np
from torch.utils.data import Dataset
from modules.transforms.tstransforms import TSToTensor
import torch


class npymapDataset(Dataset):
    # For classification tasks

    def __init__(
        self, root, transform=TSToTensor(), pretransform=None, posttransform=None, withId=False
    ) -> None:  # split=0.8, seed=1234
        # rng = np.random.default_rng(seed)
        self.X = np.load(root + "/X.npy", mmap_mode="r", allow_pickle=True)
        self.y = np.load(root + "/y.npy", mmap_mode="r", allow_pickle=True)
        assert self.X.shape[0] == self.y.shape[0], "Size mismatch between tensors"
        self.transform = transform
        self.pretransform = pretransform or []
        self.posttransform = posttransform or []
        self.withId = withId

    def __getitem__(self, index):
        X = self.X[index]

        if (self.pretransform is not None) and (callable(self.pretransform)):
            X = self.pretransform(X)

        if (self.transform is not None) and (callable(self.transform)):
            X = self.transform(X)
        
        if (self.posttransform is not None) and (callable(self.posttransform)):
            X = self.posttransform(X)

        # y=torch.nn.functional.one_hot(self.y[index].long(), num_classes=self.classes)
        y = self.y[index].astype('int64')
        if(self.withId):
            return X, y, index
        else:
            return X, y

    def __len__(self):
        return self.X.shape[0]
    

class npymapDatasetR(Dataset):
    # For regression tasks

    def __init__(
        self, root, transform=TSToTensor(), pretransform=None, posttransform=None, withId=False
    ) -> None:  # split=0.8, seed=1234
        # rng = np.random.default_rng(seed)
        self.X = np.load(root + "/X.npy", mmap_mode="r", allow_pickle=True)
        self.y = np.load(root + "/y.npy", mmap_mode="r", allow_pickle=True)
        assert self.X.shape[0] == self.y.shape[0], "Size mismatch between tensors"
        self.transform = transform
        self.pretransform = pretransform or []
        self.posttransform = posttransform or []
        self.withId = withId

    def __getitem__(self, index):
        X = self.X[index]

        if (self.pretransform is not None) and (callable(self.pretransform)):
            X = self.pretransform(X)

        if (self.transform is not None) and (callable(self.transform)):
            X = self.transform(X)
        
        if (self.posttransform is not None) and (callable(self.posttransform)):
            X = self.posttransform(X)

        # y=torch.nn.functional.one_hot(self.y[index].long(), num_classes=self.classes)
        y = self.y[index].astype(np.float32)
        if(self.withId):
            return X, y, index
        else:
            return X, y

    def __len__(self):
        return self.X.shape[0]



class npymapDatasetF(Dataset): 
    # For forecasting tasks

    def __init__(
        self, root, transform=TSToTensor(), pretransform=None, posttransform=None, withId=False
    ) -> None:  # split=0.8, seed=1234
        # rng = np.random.default_rng(seed)
        self.X = np.load(root + "/X.npy", mmap_mode="r", allow_pickle=True)
        self.transform = transform
        self.pretransform = pretransform or []
        self.posttransform = posttransform or []
        self.withId = withId

    def __getitem__(self, index):
        X = self.X[index]

        if (self.pretransform is not None) and (callable(self.pretransform)):
            X = self.pretransform(X)

        if (self.transform is not None) and (callable(self.transform)):
            X = self.transform(X)
        
        if (self.posttransform is not None) and (callable(self.posttransform)):
            X, y = self.posttransform(X)

        if(self.withId):
            return X, y, index
        else:
            return X, y

    def __len__(self):
        return self.X.shape[0]

