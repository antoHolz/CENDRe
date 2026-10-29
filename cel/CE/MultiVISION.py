from .ConceptExtraction import ConceptExtraction
import torch
from ..utils.layers import getActivations, reshape_activations, getLayers, zero_upsample, compute_RF_numerical
from sklearn.cluster import MiniBatchKMeans
from collections.abc import Mapping
import torch.nn.functional as F
from ..utils.clustering import load_kmeans

import matplotlib.pyplot as plt
from itertools import product 
from omegaconf import DictConfig
import os
from os.path import isfile
import numpy as np 
import json
import warnings
import time
warnings.simplefilter(action='ignore', category=FutureWarning)
from ..utils.misc import infer_shape



class MultiVISION(ConceptExtraction):
    def __init__(
        self, dimension, model, layer_names, vis_mode, n_classes, t_factor=0.99, n_concepts: int = 20, task="classification",**kwargs
    ):
        super().__init__(dimension)
        self.model = model
        self.task = task
        self.n_classes = n_classes
        # Visualization options (necessary when ERF is too big!)
        self.vis_mode= vis_mode
        # Layer hyperparameters
        self.layername = layer_names
        self.test_layers = getLayers(model, layer_names=layer_names)
        # Clustering hyperparameters
        self.n_concepts = n_concepts
        # Threshold hyperparametrs
        self.t_mode = 'quantile' 
        self.t_factor = t_factor

    def fit(self, loader, cfg_CE, device='cpu',**kwargs):
        self.model.eval()
        num_instances = cfg_CE.n_instances
        batch_size = cfg_CE.loader.batch_size
        full_shape = infer_shape(loader) 
        n_channels = full_shape[0]
        self.threshold = self.getThreshold(loader,batch_size, num_instances ,mode=self.t_mode, factor=self.t_factor)

        self.erf = compute_RF_numerical(
            model=self.model,
            layer_name=self.layername,
            in_shape=[1]+list(full_shape)
            )
        
        if n_channels==1:
            kmeans = MiniBatchKMeans(n_clusters=self.n_concepts, random_state=0, n_init="auto")
        else:
            kmeans = MiniBatchKMeans(n_clusters=self.n_concepts, random_state=0, n_init="auto")

        for i, batch in enumerate(loader):
            if isinstance(batch, dict):
                batchx,batchy = batch["x"], batch["y"]
            else:
                batchx,batchy,_ = batch
            if(n_channels<=5): batchx,batchy = getEIS(batchx, batchy, device)
            hans, _,_ = self.getHAN( batchx, device)
            masks = self.getMHAP( batchx, hans, in_shape=full_shape[-1], erf=self.erf)
            
            if masks is not None:
                if n_channels==1:
                    kmeans = kmeans.partial_fit(masks.squeeze().detach().cpu().numpy())
                else:
                    kmeans = kmeans.partial_fit(masks.flatten(start_dim=1).detach().cpu().numpy())
            if (i+1)*batch_size >= num_instances: break
        if n_channels==1:
            self.centroids = kmeans.cluster_centers_
        else:
            self.centroids = kmeans.cluster_centers_

        
        self.rep, self.importances=self.getImportance(loader, kmeans, batch_size, num_instances, device)
        self.sortConcepts(cfg_CE.decorrelated)

        return self.centroids, self.importances
    
    def eval(self, test_inputs, device='cpu',**kwargs):
        test_inputs = test_inputs.to(device)
        in_shape = test_inputs.shape[-1]
        kmeans = load_kmeans(self.centroids, self.n_concepts)

        hans, _, out = self.getHAN(test_inputs, device)
        masks = self.getMHAP(test_inputs, hans, in_shape=in_shape, erf=self.erf)

        if test_inputs.shape[1]==1:
            labels = kmeans.predict(masks.squeeze().detach().cpu().numpy())
        else:
            labels = kmeans.predict(masks.flatten(start_dim=1).detach().cpu().numpy())

        maskdict = {c_nr: masks[labels==c_nr] for c_nr in range(self.n_concepts)}

        upsampled = torch.sum(hans, dim=1) > 0
        upsampled = torch.flatten(zero_upsample(upsampled, in_shape)).to(device)
        true_indices = torch.nonzero(upsampled).squeeze()
        maskdict = {}
        for k in range(self.n_concepts):
            result = upsampled.clone()
            for i, idx in enumerate(true_indices):
                if not labels[i]==k:
                    result[idx] = False 
            result = torch.reshape(result, (-1, 1, in_shape))
            maskdict[k] = result*1.0
            maskdict[k] = F.conv1d(
                    maskdict[k],
                    torch.ones((1, 1, *self.erf), device=device),
                    padding="same",
                )
            maskdict[k] = torch.clamp(maskdict[k], max=1)
        if device == "cuda":
            maskdict = {k: v.to("cpu") for k, v in maskdict.items()}

        return maskdict, out


    def save(self, CE_cfg: DictConfig):
        save_path = CE_cfg.save_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        try:
            np.save(f"{save_path}/centroids_{str(self.n_concepts)}", self.centroids)
            with open(f"{save_path}/importances_{str(self.n_concepts)}.json", "w") as fp:
                json.dump(self.importances, fp)
            print(f"results saved at: {save_path}")
            with open(f"{save_path}/ERF_{CE_cfg.layer_config}.json", "w") as file:
                json.dump(self.erf, file)
            torch.save(self.threshold.detach().to("cpu"),f"{save_path}/threshold_{CE_cfg.layer_config}.pt" )
        except NameError:
            print("Error saving ECLAD concepts. Check if ECLAD has been fit")
        except:
            print("Error saving ECLAD concepts")

    def load(self, CE_cfg: DictConfig, device='cpu'):
        save_path = CE_cfg.save_dir
        try:
            self.centroids = np.load(
                f"{save_path}/centroids_{str(self.n_concepts)}.npy"
            )
            with open(
                f"{save_path}/importances_{str(self.n_concepts)}.json"
            ) as json_file:
                self.importances = json.load(json_file)
            with open(f"{save_path}/ERF_{CE_cfg.layer_config}.json") as json_file:
                self.erf = json.load(json_file)
            self.threshold = torch.load(f"{save_path}/threshold_{CE_cfg.layer_config}.pt").to(device)
        except FileNotFoundError:
            print("Wrong file or file path")


    def check_existence(self, CE_cfg: DictConfig):
        save_path = CE_cfg.save_dir
        fitting = isfile(f"{save_path}/importances_{str(self.n_concepts)}.json") and isfile(f"{save_path}/centroids_{str(self.n_concepts)}.npy") and isfile(f"{save_path}/ERF_{CE_cfg.layer_config}.json")
        report = isfile(f'{save_path}/concepts_{str(self.n_concepts)}.png')
        return fitting, report
    
    def sortConcepts(self, decorrelated):
        if decorrelated:
            temp = {k: max(v) for k, v in self.importances.items()}
        else:
            temp = self.importances
        c_indexes = [
            int(k)
            for k, _ in sorted(temp.items(), key=lambda item: item[1], reverse=True)
        ]
        self.centroids = self.centroids[c_indexes]
        self.importances = {
            str(i): self.importances[str(k)] for i, k in enumerate(c_indexes)
        }

    def report(
            self,
            test_masks,
            CE_cfg: DictConfig,
            test_inputs=None,
            test_outputs=None,
            c_indexes=None,
            test_truth=None,
            n_repeats=None,
            importance_thr: float = 0.70,
            DS_cfg=None,
            font_base=5,
            destandardize=True,
            show: bool = False,
            IS_scientific: bool = False,
            **kwargs
    ):
        """Visualise the learned concepts.
        * Only channels whose importance ≥ `importance_thr` are shown.
        * Row-wise y-limits are computed **per-channel**.
        * Row label shows the **channel name** (CE_cfg.xcols) instead of ch-id.
        * If the target was standardised, predictions/labels are de-standardised
            with CE_cfg.mean / CE_cfg.std before display.
        """
        save_path = f"{CE_cfg.save_dir}"
        os.makedirs(save_path, exist_ok=True)

        importances = self.importances
        batch_size, n_channels, signal_len = test_inputs.shape
        c_indexes = c_indexes or range(len(importances))
        interesting_out = CE_cfg.multilabel_out

        c_names = DS_cfg.xcols if hasattr(DS_cfg, "xcols") else None
        label_names = DS_cfg.ycols if hasattr(DS_cfg, "ycols") else None
        ds_std = DS_cfg.std if hasattr(DS_cfg, "std") else None
        ds_mean = DS_cfg.mean if hasattr(DS_cfg, "mean") else None

        # ------------------------------------------------------------------
        # 1. Decide which channels to plot for every concept
        # ------------------------------------------------------------------
        keep_channels = {}
        if CE_cfg.decorrelated and n_channels > 1:
            for k in c_indexes:
              
                imp_vec = np.asarray(importances[str(k)]) 
                if self.task=='regression':
                    imp_vec =  np.abs(imp_vec)           # shape (n_channels,)
                sel = np.where(imp_vec > importance_thr)[0].tolist()
                if not sel:                                               # always keep one
                    sel = [int(np.argmax(imp_vec))]
                keep_channels[k] = sel
        else:
            keep_channels = {k: np.arange(0,n_channels) for k in c_indexes}

        n_rows = sum(len(v) for v in keep_channels.values())

        # ------------------------------------------------------------------
        # 2. Build the canvas
        # ------------------------------------------------------------------
        fig, axs = plt.subplots(
            n_rows, batch_size,
            figsize=(4 * batch_size, 4 * n_rows),
            squeeze=False
        )
        plt.subplots_adjust(wspace=0.10, hspace=None)

        # colours for masks
        c = 0.4
        colours = (1.0 - c) * plt.get_cmap("viridis")(np.linspace(0, 0.75, len(importances))) \
                + c * np.ones((len(importances), 4))
        
        # line width for mask stripes (scaled with axes width)
        fig.canvas.draw()
        lw = axs[0, 0].get_window_extent().width * 0.75 / test_inputs.shape[-1] * 1.01

        # helper to restore original scale
        
        def destandardise(val, channel, id2label):
            if not destandardize:
                return val
            if isinstance(ds_mean, Mapping):
                idx = id2label[int(channel)]
                if not (hasattr(ds_mean, idx) and hasattr(ds_std, idx)):
                    return val
            else:
                idx = int(channel)
            return val * ds_std[idx] + ds_mean[idx]



        # Fonts
        font_base = font_base or 10  # e.g. override with 10, 12, 14, etc.
        font_title = font_base * 3.8     # sample titles
        font_label = font_base * 3.4     # channel + importance
        font_ticks = font_base * 3.0     # axis ticks

        # ------------------------------------------------------------------
        # 3. Plot rows
        # ------------------------------------------------------------------
        # Importance Score title on the left upper corner
        axs[0, 0].annotate("Importance\nScore",xy=(-0.43,1.05),
                           xycoords="axes fraction", ha="center", va="bottom",
                           fontsize=font_title,fontweight="bold")

        current_row = 0
        for j, k in enumerate(c_indexes):            # concept loop
            j_imp = importances[str(k)]

            for l in keep_channels[k]:               # channel loop

                #  channel-specific y-range 
                y_min = torch.min(test_inputs[:, l, :]) - abs(torch.min(test_inputs[:, l, :]))*0.1
                y_max = torch.max(test_inputs[:, l, :]) + abs(torch.max(test_inputs[:, l, :]))*0.1
                y_range = (destandardise(y_min.item(),l,c_names), destandardise(y_max.item(),l,c_names))

                for i in range(batch_size):          # sample loop
                    ax = axs[current_row, i]

                    # mask stripes
                    on_positions = test_masks[k][i].nonzero()[:, 1]
                    ax.vlines(on_positions, *y_range,
                            colors=colours[j], linewidth=lw)

                    # signal
                    ax.plot(range(signal_len),
                            destandardise(test_inputs[i, l].cpu().numpy(),l,c_names),
                            color="black")

                    # cosmetics
                    ax.set_ylim(y_range)
                    if i:                       # hide all y-ticks except first column
                        ax.set_yticks([])
                    if current_row != n_rows - 1:
                        ax.set_xticks([])
                    ax.tick_params(labelsize=font_ticks)

                    # row label = concept + channel + importance
                    if i == 0:
                        imp_val = j_imp[l] if CE_cfg.decorrelated else j_imp
                        ch_name = c_names[int(l)] if c_names else f"ch{str(l)}"
                        is_fmt = ".2e" if IS_scientific else ".1f"
                        c_part = f"$C_{{{k}}}$" if n_channels == 1 else f"$C_{{{k}}}$ | {ch_name}"
                        ylab = f"{c_part}\n$I = {float(imp_val):{is_fmt}}$"
                        ax.set_ylabel(ylab, fontsize=font_label, labelpad=font_label * 0.3)
                        ax.get_yaxis().set_label_coords(-0.35, 0.5)
                    # first row: prediction / truth
                    if current_row == 0 and (i % (n_repeats or 1) == 0):
                        if self.task == "multilabel classification":
                            truth = test_truth[i, interesting_out].item()
                            pred  = (test_outputs[i] > 0.5)[interesting_out].item()
                            title = rf"$y$: {truth} $\quad\hat y$: {pred}"
                        elif self.task == "regression":
                            test_truth = test_truth.reshape(test_truth.shape[0], -1)
                            truth = destandardise(test_truth[i, interesting_out].item(),
                                                interesting_out, label_names)
                            pred  = destandardise(test_outputs[i, interesting_out].item(),
                                                interesting_out, label_names)
                            title = rf"$y$: {truth:.3f} $\quad\hat y$: {pred:.3f}"
                        else:   # multiclass
                            truth = test_truth[i].item()
                            pred = torch.argmax(test_outputs[i]).item() 
                            title = rf"$y$: {truth:.0f} $\quad\hat y$: {pred:.0f}"
                        axs[0, i].set_title(title, fontsize=font_title, pad=20)

                current_row += 1

        # ------------------------------------------------------------------
        # 4. Save
        # ------------------------------------------------------------------
        out_png = f"{save_path}/concepts_{self.n_concepts}.png"
        out_pdf = f"{save_path}/concepts_{self.n_concepts}.pdf"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
        if show:
            try:
                from IPython.display import display
                display(fig)
            except ImportError:
                pass
        plt.close(fig)
        print(f"report saved at: {out_png}")


    def getImportance(self,loader, kmeans, batch_size, num_instances, device='cpu'):
        in_shape = infer_shape(loader)
        rep= np.zeros((self.n_classes, self.n_concepts))
        IS={}
        class_total= np.zeros((self.n_classes))

        for i, batch in enumerate(loader):
            if isinstance(batch, dict):
                batchx,batchy = batch["x"], batch["y"]
            else:
                batchx,batchy,_ = batch
            batchx = batchx.to(device)
            batchy = batchy.to(device)
            categories, number = torch.unique(batchy, return_counts=True)
            for category, num in zip(categories, number): 
                if (batchy==category).sum()>2:
                    x = batchx[batchy==category]
                    hans, _, _ = self.getHAN(x, device)
                    masks = self.getMHAP(x, hans, in_shape=in_shape[1], erf=self.erf)
                    class_total[int(category)] += num
                    if masks is not None:
                        if in_shape[0]==1:
                            labels = kmeans.predict(torch.squeeze(masks,1).detach().cpu().numpy())
                        else:
                            labels = kmeans.predict(masks.flatten(start_dim=1).detach().cpu().numpy())
                    label, counts = np.unique(labels, return_counts=True)
                    for k in range(self.n_concepts):
                        count = counts[label==k]
                        if(count.size > 0):
                            rep[int(category), k] += count
            if (i+1)*batch_size >= num_instances: break

        rep = rep/np.expand_dims(class_total,1)
        IS = {str(i):v.tolist() for i,v in enumerate(np.max(rep,axis=0)/np.max(rep))}
        return rep, IS

        
    def getThreshold(self,loader, batch_size, num_instances, mode='quantile', factor = 0.99, device='cpu'):
        """
        Compute the threshold values for each layer in the model based on the maximum activations observed 
        during a forward pass through the model. The thresholds are scaled by a given factor.

        Parameters:
            loader (torch.utils.data.DataLoader): The DataLoader object providing batches of data for evaluation.
            factor (float, optional): The scaling factor to adjust the computed thresholds. Default is 0.8.
            n_batches (int, optional): The number of batches to process from the loader. Default is 1.
            device (str, optional): The device to move tensors to ("cpu" or "cuda"). Default is "cuda".

        Returns:
            dict: A dictionary where the keys are the layer names and the values are the computed thresholds 
                for each layer, scaled by the provided factor.
        """

        # Initialize the threshold dictionary with zeros for each layer
        threshold = { layer: torch.tensor(0) for layer in self.test_layers.keys() }

        # Loop through the batches in the data loader
        for i, batch in enumerate(loader):
            if isinstance(batch, dict):
                batchx,batchy = batch["x"], batch["y"]
            else:
                batchx,batchy,_ = batch
            batchx = batchx.to(device)
            batchy = batchy.to(device)

            # Get the activations for the current batch using the getActivations function
            activations, _ = getActivations(self.model, self.test_layers, batchx)

            if mode=='max':
                # For each layer, compute the maximum activation across the batch and spatial dimensions
                for layer, act in activations.items():
                    dims = list(range(batchx.dim())).remove(1)
                    threshold = torch.max(torch.amax(act, dim=dims), threshold)
            elif mode=='quantile':
                # For each layer, compute the quantile (factor) across the batch and spatial dimensions
                for layer, act in activations.items():
                    act = torch.flatten(torch.swapdims(act, 0,1),1)
                    if i==0: threshold = torch.quantile(act,q=factor, dim=1)
                    else: threshold += torch.quantile(act,q=factor, dim=1)
            else:
                raise NotImplementedError(f"Mode {mode} not implemented.")
            if (i+1)*batch_size >= num_instances: break
            
        if mode=='max':
            # Scale the thresholds by the specified factor
            threshold *= factor
        elif mode=='quantile':
            # Get the mean (factor)-quantile
            threshold /= (i+1)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")
        return threshold
    
    def getHAN(self, batchx, device):
        han = {}
        layershape = {}
        self.model.to(device)
        batchx = batchx.to(device)
        self.threshold = self.threshold.to(device)
        # Get the activations for the current batch using the getActivations function
        activations, out = getActivations(self.model, self.test_layers, batchx)
        act = activations[self.layername[0]]

        # Compare activations with threshold
        if batchx.dim() == 3:
            ha = act > self.threshold.view(1, -1, 1)
        if batchx.dim() == 4:
            ha = act > self.threshold.view(1, -1, 1, 1)
        # Get the indices where ha is True
        han = (
            ha  # .nonzero(as_tuple=False) #Nonzero to get actual indexes in he latent space. Comment out for boolean mask
        )
        layershape = act.shape
        return han, layershape, out


    def getMHAP(self, batchx, han, in_shape, erf, mode='original', device = 'cpu'):
        batchx = batchx.to(device)
        han = han.to(device)
        n_channels = batchx.shape[1]
        subsamples = {}
        
        han = torch.sum(han, dim=1) > 0
        # The ERFs of a CNN Network are te same across all channels of the latent space. 
        # Therefore, if a neuron is highly activated for any channel, we want its ERF. This also simplifies computation
        upsampled = zero_upsample(han, in_shape)  # shape (batchsize, len)
        subsamples_l = []
        erf = erf[0]
        if mode != "original":
            if erf is not None:
                if erf > (in_shape * 0.1):
                    erf = int(0.1 * in_shape)
                    # If the ERF is too big (for many NNs it spans over the whole input), we are later clustering over redundant patches. 
                    # The subsample extraction needs to be over a small enough patch to make sense. 
        if erf is not None:
            for b, sample in enumerate(upsampled):
                for i in range(in_shape):
                    if sample[i]:
                        start_idx = max(0, i - erf // 2)
                        end_idx = min(in_shape, i + (erf+1) // 2)
                        subsample = torch.zeros((1, n_channels, erf))
                        subsample[0, :, 0 : end_idx - start_idx] = batchx[
                            b, :, start_idx:end_idx
                        ]
                        subsamples_l.append(subsample)
            subsamples = torch.cat(subsamples_l)
        else:
            subsamples = None

        return subsamples


def getEIS(batchx,batchy, device='cpu'): 
    """
    Expands a batch of input-output pairs by generating all possible combinations of zeroed-out channels in the input,
    and then returning the expanded inputs and their corresponding outputs.

    Parameters:
    - batchx (torch.Tensor): A tensor of input data, with shape `(batch_size, n_channels, height, width)`.
    - batchy (torch.Tensor): A tensor of corresponding outputs, with shape `(batch_size, ...)`.
    - device (str, optional): The device to move the tensors to. Defaults to `'cpu'`. Can be `'cpu'`, `'cuda'`, or any valid device string.

    Returns:
    - expandedx (torch.Tensor): The expanded input tensor with shape `(batch_size * (2^n_channels-1), n_channels, height, width)`.
    - expandedy (torch.Tensor): The expanded output tensor with shape `(batch_size * (2^n_channels-1), ...)`.
    """

    n_channels = batchx.shape[1]
    combis = torch.tensor(list(product(range(2), repeat=n_channels)))[1:] #torch.tensor([k for i, k in enumerate(combis(range(2), repeat=n_channels)) if not (i==0)]) #
    combis = combis.to(device)
    batchx = batchx.to(device)
    batchy = batchy.to(device)

    # Get the number of combinations
    num_combinations = combis.shape[0]
    
    if batchx.dim()==3:
        # Expand batchx to match the number of combinations and pply the corresponding binary masks
        expandedx = batchx.unsqueeze(1).repeat(1, num_combinations, 1, 1)  # Shape: (batch_size, num_combinations, channels, len)
        expandedx = expandedx * combis.view(1, num_combinations, n_channels, 1)
    if batchx.dim()==4:
        # Expand batchx to match the number of combinations and pply the corresponding binary masks
        expandedx = batchx.unsqueeze(1).repeat(1, num_combinations, 1, 1, 1)  # Shape: (batch_size, num_combinations, channels, len)
        expandedx = expandedx * combis.view(1, num_combinations, n_channels, 1, 1)      

    # The output y doesn't change, we just expand it in the same way as expandedx
    expandedy = batchy.unsqueeze(1).repeat(1, num_combinations)  # Shape: (batch_size, num_combinations, ...)

    # Reshape the expanded inputs and outputs to match the desired format:
    expandedx = expandedx.view(-1, *expandedx.shape[2:])  # Shape: (batch_size * num_combinations, channels, height, width)
    expandedy = expandedy.view(-1)  # Shape: (batch_size * num_combinations, ...)
    return expandedx, expandedy