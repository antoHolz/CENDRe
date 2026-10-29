import torch
from sklearn.cluster import MiniBatchKMeans
from .ConceptExtraction import ConceptExtraction
from typing import Any, Dict, Optional, Union
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import os
from omegaconf import DictConfig
from ..utils.layers import getActivations, reshape_activations, getLayers
from ..utils.gradients import get_classGradients, get_gradients
from ..utils.clustering import load_kmeans, predict_concept
from os.path import isfile
from ..utils.misc import infer_shape
from collections.abc import Mapping

import gc


class ECLAD(ConceptExtraction):
    def __init__(
        self, dimension, model, layer_names, n_concepts: int = 20, task="classification"
    ):
        super().__init__(dimension)
        self.model = model
        self.test_layers = getLayers(model, layer_names=layer_names)
        self.n_concepts = n_concepts
        self.task = task

    def fit(
        self,
        dataloader,
        cfg_CE,
        device = 'cpu',
        **kwargs
    ):
        """Full pipeline for Concept extraction an importance score computation introduced in Methods>ECLAD section. See description of each of the functions for more detail."""
        self.model.eval()
        num_instances = cfg_CE.n_instances
        batch_size = cfg_CE.loader.batch_size
        decorrelated = cfg_CE.decorrelated
        multilabel_out = cfg_CE.multilabel_out
        
        x_shape = infer_shape(dataloader)
        n_channels = x_shape[0]
        input_shape = x_shape[1:]
        
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_concepts, random_state=0, n_init="auto"
        )
        self.model.to("cpu")
        
        for i, batch in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            if isinstance(batch, dict):
                inputs = batch["x"]
            else:
                inputs, _, _ = batch
            inputs = inputs.to("cpu")
            test_activations, _ = getActivations(
                self.model, self.test_layers, inputs
            )
            reshaped_activations = reshape_activations(
                test_activations, input_shape, self.dimension
            )
            lads = self.get_lads(reshaped_activations)
            kmeans = kmeans.partial_fit(lads.detach().cpu().numpy())
            if (i+1)*batch_size >= num_instances: break  
        self.centroids = kmeans.cluster_centers_
        self.importances = self.get_scaledImportances(
            dataloader,
            input_shape,
            batch_size,
            num_instances,            
            n_channels,
            decorrelated,
            multilabel_out,
            device,
        )
        self.sortConcepts(decorrelated)
        return self.centroids, self.importances

    def eval(self, test_inputs, device="cpu",**kwargs):
        """Mask generation for individual instances. This can be used only after CE (fit) has been applied."""
        test_inputs = test_inputs.to(device)
        self.model.to(device)
        centroids = torch.from_numpy(self.centroids).to(device)

        n_inputs = test_inputs.shape[0]  # Inputs are to be given in a batch ...
        if self.dimension == ("1D") or self.dimension == (
            "timeseries"
        ):  # careful with shape here
            input_shape = test_inputs.shape[-1]
        elif self.dimension == ("2D") or self.dimension == ("images"):
            input_shape = test_inputs.shape[-2:]
        test_activations, test_outputs = getActivations(
            self.model, self.test_layers, test_inputs
        )
        test_reshaped_activations = reshape_activations(
            test_activations, input_shape, device=device
        )
        test_lads = self.get_lads(test_reshaped_activations)
        test_labels, _ = predict_concept(test_lads, centroids)
        distances = test_lads - centroids[test_labels]
        test_masks = self.getMasks(test_labels, n_inputs, input_shape, distances,device)
        if device == "cuda":
            test_masks = {k: v.to("cpu") for k, v in test_masks.items()}
        return test_masks, test_outputs


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
        save_path = CE_cfg.save_dir
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
              
                imp_vec = np.asarray(importances[str(k)])             # shape (n_channels,)
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
                            colors=colours[j], linewidth=signal_len / 100)

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
                        ax.set_ylabel(ylab,
                                    rotation=0, labelpad=60, fontsize=font_label, ha='right', va='center')
                        ax.yaxis.set_label_position('left')
                        ax.get_yaxis().set_label_coords(-0.4, 0.5)
                    # first row: prediction / truth
                    if current_row == 0 and (i % (n_repeats or 1) == 0):
                        if self.task == "multilabel classification":
                            truth = test_truth[i, interesting_out].item()
                            pred  = (test_outputs[i] > 0.5)[interesting_out].item()
                            title = rf"$y$: {truth} $\quad\hat y$: {pred}"
                        elif self.task == "regression":
                            truth = destandardise(test_truth[i, interesting_out].item(),
                                                interesting_out, label_names)
                            pred  = destandardise(test_outputs[i, interesting_out].item(),
                                                interesting_out, label_names)
                            title = rf"$y$: {truth:.1f} $\quad\hat y$: {pred:.1f}"
                        else:   # multiclass
                            pred = torch.argmax(test_outputs[i]).item()
                            title = f"pred: {pred}"
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


    def save(self, CE_cfg: DictConfig):
        save_path = CE_cfg.save_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        try:
            np.save(f"{save_path}/centroids_{str(self.n_concepts)}", self.centroids)
            with open(f"{save_path}/importances_{str(self.n_concepts)}.json", "w") as fp:
                json.dump(self.importances, fp)
            print(f"results saved at: {save_path}")
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
        except FileNotFoundError:
            print("Wrong file or file path")

    def check_existence(self, CE_cfg: DictConfig):
        save_path = CE_cfg.save_dir
        fitting = isfile(f"{save_path}/importances_{str(self.n_concepts)}.json") and isfile(f"{save_path}/centroids_{str(self.n_concepts)}.npy")
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

    def get_lads(self, aggregated_activations):
        """Lads extraction as described in Encoding of latent space section"""
        lads = aggregated_activations
        if self.dimension == "2D" or self.dimension == "images":
            lads = torch.flatten(lads, -2, -1)
            lads = torch.permute(lads, (0, 2, 1))
        elif self.dimension == "1D" or self.dimension == "timeseries":
            lads = torch.permute(lads, (0, 2, 1))
        else:
            print("LAD extraction error. Check dimensions of evaluated inputs")
        lads = torch.flatten(lads, 0, 1)
        self.lads_shape = lads.shape

        return lads

    # Visualization
    def getMasks(self, labels, batch_size, input_shape, distances, device="cpu"):
        """Mask generation as described in Mining of patterns and visualization. See eq. 1."""
        labels = labels.to(device)
        distances.to(device)
        if type(input_shape) == int:
            labels = torch.reshape(labels, (batch_size, input_shape))
        else:
            labels = torch.reshape(
                labels, (batch_size, *input_shape)
            )  
        labels = labels.unsqueeze(1)
        masks = {}
        masks[self.n_concepts] = labels
        for i in range(self.n_concepts):
            masks[i] = labels == i

        return masks

    # IMPORTANCE SCORE

    def g(self, y, out_nr=0, device="cpu"):
        """Wrapper for y in Eq. 2."""
        if (
            self.task == "regression" or self.task == "multilabel classification"
        ):  # Single output
            if y.ndim==1: return y[:]
            return y[:,out_nr]
        else:
            y = y.unsqueeze(-1).to(device)
            y_T = torch.transpose(y, 1, 2)

            a = torch.bmm(y, torch.ones(y_T.shape, device=device))
            b = torch.bmm(torch.ones(y.shape, device=device), torch.transpose(y, 1, 2))

        return torch.norm(a - b, p=2, dim=(1, 2))

    def relevance_img(self, gradients, masks, nr, n_channels=1, decorrelated=False):
        """Relevance per image as described in Eq. 3"""

        # Dimensions
        # gradients.squeeze().shape = batch_size,n_channels, flattened_input_shape
        # masks[nr] = batch_size, 1, flattened_input_shape *
        # masks[nr].squeeze().shape = batch_size, flattened_input_shape
        if decorrelated and n_channels > 1:
            relev = gradients.squeeze() * masks[nr].repeat((1, n_channels, 1))
        elif n_channels == 1:
            relev = gradients.squeeze() * masks[nr].squeeze()
        else:
            relev = (
                torch.mean(gradients.squeeze() * masks[nr].repeat((1, n_channels, 1)), dim=1)
            )  # batch_size, (nr_channels), flattened_input_shape
        return torch.norm(relev, p=1, dim=list(range(1, relev.ndim)))

    def get_scaledImportances(
        self,
        dataloader,
        shape_in,
        batch_size,
        num_instances,
        n_channels=1,
        decorrelated=False,
        multilabel_out=0,
        device="cpu",
    ):
        """Importance score computation in section Relevance computation. See Eq. 2,3, and 4."""
        model = self.model
        layers = self.test_layers
        centroids = self.centroids
        model.to(device)
        if decorrelated:
            importances = torch.zeros(
                (centroids.shape[0], n_channels), device=device, dtype=float
            )
        else:
            importances = torch.zeros(centroids.shape[0], device=device, dtype=float)
        n_instances = torch.ones(centroids.shape[0], device=device, dtype=float) * 1e-10

        centroids = torch.from_numpy(centroids).to(device)

        for i, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                inputs = batch["x"]
            else:
                inputs, _, _ = batch
            inputs = inputs.to(device)
            inputs.requires_grad = True
            if (
                len(inputs.shape) == 4
            ): 
                shape_in = inputs.shape[-2:]
            else:
                shape_in = inputs.shape[-1]
            activations, outputs = getActivations(model, layers, inputs)
            reshaped_activations = reshape_activations(
                activations, shape_in, input_dimension=self.dimension, device=device
            )
            lads = self.get_lads(reshaped_activations)
            labels, _ = predict_concept(lads, centroids)
            distances = lads - centroids[labels]

            masks = self.getMasks(labels, inputs.shape[0], shape_in, distances, device)
            out_g = self.g(outputs, out_nr=multilabel_out, device=device)
            gradients = get_gradients(out_g, inputs)
            for c in range(centroids.shape[0]):
                current_instances = 0
                for k in range(masks[c].shape[0]):
                    if masks[c][k].any():
                        current_instances += 1  
                if current_instances != 0:
                    importances[c] += torch.sum(
                        self.relevance_img(
                            gradients, masks, c, n_channels, decorrelated
                        ),
                        dim=0,
                    )
                    n_instances[c] += current_instances

            if (i+1)*batch_size >= num_instances: break
            gc.collect()
            torch.cuda.empty_cache()

        if decorrelated:
            n_instances = torch.unsqueeze(n_instances, dim=1)
        importances = importances.cpu().numpy() / n_instances.cpu().numpy()
        max_abs_importance = float(np.max(np.abs(importances)))
        scaled_importances = {
            str(c): (importances[c] / max_abs_importance).tolist()
            for c in range(centroids.shape[0])
        }
        gc.collect()
        torch.cuda.empty_cache()
        return scaled_importances
