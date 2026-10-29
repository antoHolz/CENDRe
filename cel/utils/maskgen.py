import os
from collections.abc import Mapping
import numpy as np
import torch
from datasets import Dataset

def destandardise(DS_cfg, val, channel, typ='in'):
    ds_std = DS_cfg.std if hasattr(DS_cfg, "std") else None
    ds_mean = DS_cfg.mean if hasattr(DS_cfg, "mean") else None
    c_names = DS_cfg.xcols if hasattr(DS_cfg, "xcols") else None
    label_names = DS_cfg.ycols if hasattr(DS_cfg, "ycols") else None
    id2label = c_names if typ=='in' else label_names

    if isinstance(ds_mean, Mapping):
        idx = id2label[int(channel)]
        if not (hasattr(ds_mean, idx) and hasattr(ds_std, idx)):
            return val
    else:
        idx = int(channel)
    return val * ds_std[idx] + ds_mean[idx]

def extract_ConceptMasks(CE, samples_x, samples_y, samples_id, DS_cfg, outdir='D:/Datasets/DS_TS/WZL/SPP/ConceptMasks', modelname='model1', domain='input', device='cpu'):
    os.makedirs(outdir, exist_ok=True)
    for bucket in range(len(samples_x)):
        key  = str(bucket)
        x = samples_x[key]
        if(len(x)>0):
            y = samples_y[key].reshape(samples_y[key].shape[0], -1) # ensure 2D
            idx = samples_id[key]
            masks, y_hat = CE.eval(x,domain=domain, device=device) 

            # Destandardize
            for channel in range(x.shape[1]):
                x[:,channel] = destandardise(DS_cfg, x[:,channel], channel, 'in')
            for channel in range(y.shape[1]):
                y[:,channel] = destandardise(DS_cfg, y[:,channel], channel, 'out')
                y_hat[:,channel] = destandardise(DS_cfg, y_hat[:,channel], channel, 'out')

            masks_col = []
            n_samples = x.shape[0]
            for i in range(n_samples):
                one = {}
                for k in masks.keys():
                    mk = masks[k][i]
                    # accept shapes (T), (1,T) or (C,T) -> collapse over first dim
                    if mk.ndim >= 2:
                        mk = (mk != 0).any(dim=0)
                    else:
                        mk = (mk != 0)
                    one[str(k)] = mk.numpy().astype("bool").tolist()
                masks_col.append(one)
        
            data = {
                "Key": [bucket for i in range(n_samples)],
                "DS_id": [idx[i] for i in range(n_samples)],
                "x": [x[i].numpy() for i in range(n_samples)],  # (Ch, T)
                "y": [y[i].numpy() for i in range(n_samples)],  # (Ch_out,)
                "y_hat": [y_hat[i].cpu().detach().numpy() for i in range(n_samples)],  # (Ch_out,)
                "masks": masks_col,  # dict per row
            }

        else:
            data = {
                "Key": [],
                "DS_id": [],
                "x": [],  # (Ch, T)
                "y": [],  # (Ch_out,)
                "y_hat": [],  # (Ch_out,)
                "masks": [],  # dict per row
            }
        # store
        
        ds = Dataset.from_dict(data)
        ds.save_to_disk(f"{outdir}/bin{bucket}_masks")
