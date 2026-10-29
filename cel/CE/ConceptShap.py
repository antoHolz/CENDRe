from .ConceptExtraction import ConceptExtraction
from ..utils.layers import (
    getActivations,
    reshape_activations,
    getLayers,
    compute_RF_numerical,
)
from ..utils.clustering import predict_concept, concept_similarity
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modules.utils import MetricAggregator, EarlyStoppingCustom
import logging
import numpy as np
import json
from os.path import isfile
import matplotlib.pyplot as plt
import os
from collections.abc import Mapping
import torch
import torch.nn.functional as F
from torchvision.ops import MLP
from torch import nn
import torchmetrics

import hydra
from omegaconf import DictConfig
from ..utils.misc import infer_shape, get_x


class ConceptShap(ConceptExtraction):
    def __init__(
        self,
        dimension,
        model,
        n_concepts,
        n_classes,
        task,
        layer_name,
        n_features,
        lambda1,
        lambda2,
        beta,
        vis_mode,
        **kwargs
    ):
        super().__init__(dimension)
        self.model = model
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.task = task
        self.layer_name = layer_name
        self.n_features = n_features
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.beta = beta
        self.vis_mode = vis_mode

        if self.n_features != 0:
            self.g = MLP(self.n_concepts, [500, self.n_features])
            self.C = nn.Linear(self.n_features, self.n_concepts, bias=False)

    def fit(
        self, valid_loader, cfg_CE, device="cuda", n_concepts: int = 0, verbose=True,**kwargs
    ):
        """Full pipeline for concept extraction using ConceptShap, accordnig to the paper 'On Completeness-aware Concept-Based Explanations in Deep Neural Networks' by Yeh et. al.
        test layers must be a LIST w layername 
        """
        cfg_training = cfg_CE.cfg_training
        full_shape = infer_shape(valid_loader)
        # Init models
        test_layers = getLayers(self.model, layer_names=self.layer_name)
        if n_concepts != 0:
            self.n_concepts = n_concepts
        
        if self.n_features == 0:
            test_activations, _ = getActivations(
                self.model.to(device),
                test_layers,
                get_x(valid_loader.dataset[0]).to(device), 
            )
            self.n_features = test_activations[self.layer_name[0]].shape[1]
            self.layerinfo={'n_features': int(self.n_features)}
        self.C = nn.Linear(self.n_features, self.n_concepts, bias=False)
        self.g = MLP(self.n_concepts, [500, self.n_features])

        if verbose:
            print("training surrogate model")
        self.cs_train(valid_loader, cfg_training, "train", verbose, device)
        self.cScore, _, modelAcc, randAcc = self.completeness(
            valid_loader, device=device
        )

        if verbose:
            print("Computing importance scores")
        self.importances = self.computeImportance(
            modelAcc, randAcc, valid_loader, cfg_training, verbose, device
        )

        full_shape = infer_shape(valid_loader)
        if self.vis_mode == "ERF":
            erf = compute_RF_numerical(
                self.model,
                self.layer_name,
                [1]+list(full_shape),
            )
            self.layerinfo['ERF'] = erf

    def eval(self, x, device="cuda",**kwargs):
        if self.dimension == ("1D") or self.dimension == (
            "timeseries"
        ):  # careful with shape here
            input_shape = x.shape[-1]
        elif self.dimension == ("2D") or self.dimension == ("images"):
            input_shape = x.shape[-2:]
        inputs = x.to(device)
        self.model.to(device)
        self.model.eval()
        test_layers = getLayers(self.model, layer_names=self.layer_name)
        C_weights = self.C.weight.data
        self.C.eval()
        self.g.eval()

        if self.vis_mode == "Upsample":
            activations, out = getActivations(self.model, test_layers, inputs)
            reshaped = reshape_activations(activations, input_shape, device=device)
            lads = self.get_lads(reshaped)

            labels, _ = concept_similarity(lads, C_weights, mode="dot")
            masks = self.getMasksUpsample(labels, x.shape[0], input_shape, device)
        elif self.vis_mode == "ERF":
            out, _, _, activations = self.evalModel(test_layers, inputs, device)
            labels, _ = concept_similarity(
                activations[self.layer_name[0]], C_weights, mode="dot"
            )
            labels = torch.unflatten(
                labels, 0, [x.shape[0], int(labels.shape[0] / x.shape[0])]
            )
            masks = self.getMasksERF(labels, input_shape, device)
        if device == "cuda":
            masks = {k: v.to("cpu") for k, v in masks.items()}
        return masks, out

 

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



    def save(self, path_cfg):
        path = Path(path_cfg.save_dir)

        # Check if save_path is a directory or a file path
        path.mkdir(parents=True, exist_ok=True)

        # Save the model  point
        torch.save(self.g.state_dict(), f"{path}/modelG_{self.n_concepts}.pth")
        torch.save(self.C.state_dict(), f"{path}/modelC_{self.n_concepts}.pth")

        with open(f"{path}/importances_{str(self.n_concepts)}.json", "w") as fp:
            json.dump(self.importances, fp)


        with open(f"{path}/layerinfo_{self.layer_name[0]}.json", "w") as file:
            json.dump(self.layerinfo, file)

    def load(self, path_cfg, device='cuda'):
        path = path_cfg.save_dir

        with open(f"{path}/layerinfo_{self.layer_name[0]}.json") as json_file:
            self.layerinfo = json.load(json_file)
        self.n_features = self.layerinfo['n_features']

        self.g = MLP(self.n_concepts, [500, self.n_features]).to(device)
        self.g.load_state_dict(
            torch.load(f"{path}/modelG_{self.n_concepts}.pth", weights_only=True)
        )
        self.g.eval()
        self.C = nn.Linear(self.n_features, self.n_concepts, bias=False).to(device)
        self.C.load_state_dict(
            torch.load(f"{path}/modelC_{self.n_concepts}.pth", weights_only=True)
        )
        self.C.eval()
        with open(
            f"{path}/importances_{str(self.n_concepts)}.json"
        ) as json_file:
            self.importances = json.load(json_file)


    def check_existence(self, CE_cfg: DictConfig):
        save_path = CE_cfg.save_dir
        fitting = (
            isfile(f"{save_path}/modelG_{self.n_concepts}.pth")
            and isfile(f"{save_path}/modelC_{self.n_concepts}.pth")
            and isfile(f"{save_path}/importances_{str(self.n_concepts)}.json")
            and isfile(f"{save_path}/layerinfo_{self.layer_name[0]}.json")
        )
        report = isfile(f"{save_path}/concepts_{str(self.n_concepts)}.png")
        return fitting, report

    def vc(self, phi):
        vc_flat = self.C(phi)
        vc_flat = nn.Threshold(self.beta, 0, inplace=False)(vc_flat)
        vc_flat = F.normalize(vc_flat, p=2.0, dim=1)
        return vc_flat

    def evalModel(self, layers, input, device="cuda"):
        phi_x_store = {}
        vc_store = {}
        g_vc_store = {}
        self.model.to(device)
        input.to(device)

        def nested_hook(
            layer_id: str, g_vc_store: dict, vc_store: dict, phi_x_store: dict
        ):
            def myhook(module, input, output):
                # flatten
                phi_flat = output
                in_shape = phi_flat.shape[2:]
                batch_size = phi_flat.shape[0]
                if self.dimension == "2D" or self.dimension == "images":
                    phi_flat = torch.flatten(phi_flat, -2, -1)
                    phi_flat = torch.permute(phi_flat, (0, 2, 1))
                elif self.dimension == "1D" or self.dimension == "timeseries":
                    phi_flat = torch.permute(phi_flat, (0, 2, 1))
                else:
                    print(
                        "activation reshaping error. Check dimensions of evaluated inputs"
                    )
                phi_flat = torch.flatten(phi_flat, 0, 1)
                vc = self.vc(phi_flat)
                g_vc = self.g(vc)
                vc = torch.unflatten(vc, 0, [batch_size, *in_shape])
                g_vc = torch.unflatten(g_vc, 0, [batch_size, *in_shape])
                if self.dimension == "2D" or self.dimension == "images":
                    vc = torch.permute(vc, (0, 3, 1, 2))
                    g_vc = torch.permute(g_vc, (0, 3, 1, 2))
                elif self.dimension == "1D" or self.dimension == "timeseries":
                    vc = torch.permute(vc, (0, 2, 1))
                    g_vc = torch.permute(g_vc, (0, 2, 1))

                # storing partial evaluations
                phi_x_store[layer_id] = phi_flat
                vc_store[layer_id] = vc
                g_vc_store[layer_id] = g_vc
                return g_vc

            return myhook

        handles = []
        for key, value in layers.items():
            handles.append(
                value.register_forward_hook(
                    nested_hook(key, g_vc_store, vc_store, phi_x_store)
                )
            )
        # Forward pass
        out = self.model(input)
        # Remove hooks
        for handle in handles:
            handle.remove()

        return out, g_vc_store, vc_store, phi_x_store

    def completeness(self, dataloader, modelAcc=None, randAcc=None, device="cuda"):
        # Set device and mode
        self.model.eval()
        self.g.eval()
        self.C.eval()

        self.model.to(device)
        self.g.to(device)
        self.C.to(device)

        if randAcc == None:
            randAcc = 1 / self.n_classes
        if modelAcc == None:
            modelAccMetric = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.n_classes
            )
        hgAccMetric = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )

        test_layers = getLayers(self.model, layer_names=self.layer_name)
        if self.task == "classification":
            y_dtype = torch.int64
        else:
            y_dtype = torch.float
        if self.task == "multilabel classification":
            sig = torch.nn.Sigmoid()

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                x,y = batch["x"], batch["y"]
            else:
                x,y, _ = batch
            x, y = x.to(device), y.to(device, dtype=y_dtype)
            with torch.autocast(device_type=device):
                out, _, _, _ = self.evalModel(test_layers, x, device)
                if self.task == "classification":
                    logprob = F.log_softmax(out, dim=1)
                    y_hat = torch.exp(logprob)
                elif self.task == "multilabel classification":
                    y_hat = sig(out)
                elif self.task == "regression":
                    y_hat = out.squeeze()
            hgAccMetric(y_hat.cpu(), y.cpu())

            if modelAcc == None:
                f_out = self.model(x)
                if self.task == "classification":
                    logprob = F.log_softmax(f_out, dim=1)
                    y_hat = torch.exp(logprob)
                elif self.task == "multilabel classification":
                    y_hat = sig(f_out)
                elif self.task == "regression":
                    y_hat = f_out.squeeze()
                modelAccMetric(y_hat.cpu(), y.cpu())

        hgAcc = hgAccMetric.compute().tolist()
        if modelAcc == None:
            modelAcc = modelAccMetric.compute().tolist()

        score = (hgAcc - randAcc) / (modelAcc - randAcc)
        return score, hgAcc, modelAcc, randAcc

    def computeImportance(
        self, modelAcc, randAcc, valid_loader, cfg_train, verbose=True, device="cuda"
    ):
        checkpoint_g = self.g.state_dict()
        checkpoint_C = self.C.weight.clone()

        S = np.arange(self.n_concepts)
        Completeness_store = {}
        IS = torch.zeros(self.n_concepts)

        for epoch in range(cfg_train.MCSh_epochs):
            np.random.shuffle(S)

            # Reset C and g weights
            self.g.load_state_dict(checkpoint_g)
            with torch.no_grad():
                self.C.weight = nn.Parameter(checkpoint_C.clone())

            if str(np.sort(S)) not in Completeness_store.keys():
                # compute completeness and store in completeness cscore in ordered form
                Completeness_store[str(np.sort(S))], _, _, _ = self.completeness(
                    valid_loader, modelAcc, randAcc, device
                )

            if verbose:
                print(f"Importance epoch {epoch}")
            for id in np.arange(self.n_concepts - 1):
                IS[id] += Completeness_store[str(np.sort(S[id:]))]

                if str(np.sort(S[id + 1 :])) not in Completeness_store.keys():
                    # set modified c weights
                    with torch.no_grad():
                        self.C.weight[S[id], :] = 0

                    # retrain c and g for n epochs
                    self.cs_train(valid_loader, cfg_train, "finetune", verbose, device)

                    # compute completeness and store in completeness cscore in ordered form
                    Completeness_store[str(np.sort(S[id + 1 :]))], _, _, _ = (
                        self.completeness(valid_loader, modelAcc, randAcc)
                    )
                IS[id] -= Completeness_store[str(np.sort(S[id + 1 :]))]

        # Reset to original 
        self.g.load_state_dict(checkpoint_g)
        with torch.no_grad():
            self.C.weight = nn.Parameter(checkpoint_C.clone())

        IS = IS / cfg_train.MCSh_epochs
        IS_dict = {str(c): (IS[c]).tolist() for c in range(self.n_concepts)}

        return IS_dict

    def cs_train(
        self,
        loader: DataLoader,
        cfg: DictConfig,
        mode="train",
        verbose=True,
        device="cuda",
    ) -> torch.nn.Module:
        """
        Trains and evaluates a neural network model.

        Args:
            model (nn.Module): The neural network model to be trained and evaluated.
            train_loader (DataLoader): DataLoader for the training dataset.
            valid_loader (DataLoader): DataLoader for the validation dataset.
            cfg (DictConfig): Configuration object containing training parameters.
            device (str): The device to run the training on ("cpu" or "cuda").

        Returns:
            None: The function does not return any value.
        """

        ##############################
        # Device Setup
        ##############################
        model = self.model
        model.to(device)
        model.eval()
        if mode == "train":
            max_epochs = cfg.max_epochs_train
        else:
            max_epochs = cfg.max_epochs_finetune
        self.C.to(device)
        self.g.to(device)

        test_layers = getLayers(self.model, layer_names=self.layer_name)

        ##############################
        # Object Instantiation
        ##############################
        # Instantiate loss function
        criterion = hydra.utils.instantiate(cfg.loss)

        # Instantiate optimizer
        if mode == "train":
            optimizer = hydra.utils.instantiate(cfg.optimizer,
                [{"params": self.g.parameters()}, {"params": self.C.parameters()}],
            )
        else:
            optimizer = hydra.utils.instantiate(cfg.optimizer,
                [{"params": self.g.parameters()}]
            )
            self.C.weight.requires_grad = False
            self.C.eval()

        # Instantiate scheduler
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer)

        # Initialize loggers
        loggers = [
            hydra.utils.instantiate(logger_cfg) for logger_cfg in cfg.loggers.values()
        ]
        logger = logging.getLogger("training")

        # Initialize MetricAggregator
        num_classes = cfg.num_classes
        metric_aggregator = MetricAggregator(
            num_classes=num_classes,
            metrics=cfg.metrics,
            device=device,
            loggers=loggers,
            task=self.task,
        )

        # Early stopping and checkpointing
        early_stopping = EarlyStoppingCustom(**cfg.early_stopping_config)

        # scaler for automatic mixed precision
        scaler = torch.amp.GradScaler('cuda')

        # variables for regression or classification
        if self.task == "classification":
            y_dtype = torch.int64
        else:
            y_dtype = torch.float

        if self.task == "multilabel classification":
            sig = torch.nn.Sigmoid()

        ##############################
        # Epoch Loop
        ##############################
        for epoch in range(max_epochs):
            ##############################
            # Training Loop
            ##############################
            if mode == "train":
                self.C.train()
            self.g.train()
            lr = torch.tensor(optimizer.param_groups[0]["lr"])
            for batch_idx, batch in enumerate(loader):
                if isinstance(batch, dict):
                    x,y = batch["x"], batch["y"]
                else:
                    x,y, _ = batch
                x, y = x.to(device), y.to(device, dtype=y_dtype)
                with torch.autocast(device_type=device):
                    out, _, _, phi_x = self.evalModel(test_layers, x, device)

                    if self.task == "classification":
                        logprob = F.log_softmax(out, dim=1)
                        y_hat = torch.exp(logprob)
                    elif self.task == "multilabel classification":
                        logprob = out
                        y_hat = sig(out)
                    elif self.task == "regression":
                        logprob = out.to(dtype=torch.float).squeeze()
                        y_hat = out.squeeze()
                    loss = criterion(logprob, y) / cfg.gradient_accumulation_steps

                    C_weights = F.normalize(self.C.weight.data, p=2.0, dim=1)
                    a_normal = F.normalize(phi_x[self.layer_name[0]], p=2.0, dim=1)
                    labels, distances = concept_similarity(
                        a_normal, C_weights, mode="dot"
                    )

                    K = len(y) / (2 * self.n_classes)  # Assuming balanced classes
                    R1 = torch.sum(distances[labels]) / (self.n_concepts * K)
                    R2 = torch.tril(torch.matmul(C_weights, C_weights.T), -1).sum() / (
                        (self.n_concepts - 1) * self.n_concepts
                    )
                    loss -= (
                        self.lambda1 * R1 - self.lambda2 * R2
                    ) / cfg.gradient_accumulation_steps
                    # In ts, R1 gets huge due to nr of activations ...
                # Accumulates scaled gradients.
                scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0 or (
                    batch_idx + 1
                ) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # Update metrics
                with torch.autocast(device_type=device):
                    metric_aggregator.step(
                        y_hat=y_hat,
                        y=y,
                        loss=loss,
                        epoch=torch.tensor(epoch + 1),
                        lr=lr,
                        phase="train",
                    )
                if batch_idx * len(y) > cfg.max_instances_epoch:
                    break
            # Compute and log metrics
            with torch.autocast(device_type=device):
                train_results = metric_aggregator.compute(phase="train")
            if verbose:
                print(
                    f"Epoch {epoch+1} Train: {' '.join([f'{k}:{v:.3E}'.replace('_epoch','').replace('train_','') for k,v in train_results.items() if isinstance(v,float)])}"
                )

            ##############################
            # End of epoch
            ##############################
            metric_aggregator.reset(phase="train")

            # Step the scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss)

            # Check early stopping
            early_stopping(train_results)
            if early_stopping.should_stop and cfg.min_epochs < epoch:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Remove overlapping concepts
        if mode == "train":
            c_norm = F.normalize(self.C.weight, p=2.0, dim=1)
            sim = torch.tril(torch.matmul(c_norm, c_norm.T), -1)
            removed = []
            for c1 in range(self.n_concepts):
                if c1 in removed:
                    continue

                for c2 in range(c1+1, self.n_concepts):
                    if(c2 not in removed and sim[c2,c1]>0.95): 
                        with torch.no_grad():
                            self.C.weight[c2, :] = 0
                        removed.append(c2)
                        print(f"Removing similar concepts: c_{c1:02}-c_{c2:02}={sim[c2,c1]}")





        ##############################
        # Saving Results
        ##############################
        self.C.eval()
        self.g.eval()

    ################ Visualization #########################

    def getMasksUpsample(self, labels, batch_size, input_shape, device="cuda"):
        """Mask generation as described in Mining of patterns and visualization. See eq. 1."""
        labels = torch.tensor(labels).to(device)

        gral_mask = torch.tensor(labels)
        if type(input_shape) == int:
            gral_mask = torch.reshape(gral_mask, (batch_size, input_shape))
        else:
            gral_mask = torch.reshape(
                gral_mask, (batch_size, *input_shape)
            ) 

        gral_mask = gral_mask.unsqueeze(1)

        masks = {}
        masks["all"] = gral_mask
        for i in range(self.n_concepts):
            masks[i] = (
                gral_mask == i
            ).float() 

        return masks

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

    def getMasksERF(self, labels, shape, device="cuda"):
        labels += torch.ones_like(
            labels
        )  # such that 0 is not a concept but a neutral thing (Careful, here so generally)
        gral_mask = zero_upsample(labels, shape)
        gral_mask = gral_mask.unsqueeze(1)

        masks = {}
        masks["all"] = gral_mask  # self.n_concepts
        for i in range(self.n_concepts):
            masks[i] = (gral_mask == i + 1).float()
            if self.dimension == "2D" or self.dimension == "images":
                masks[i] = F.conv2d(
                    masks[i],
                    torch.ones((1, 1, *self.layerinfo['ERF']), device=device),
                    padding="same",
                )
                masks[i] = torch.clamp(masks[i], max=1)
            elif self.dimension == "1D" or self.dimension == "timeseries":
                masks[i] = F.conv1d(
                    masks[i],
                    torch.ones((1, 1, *self.layerinfo['ERF']), device=device),
                    padding="same",
                )
                masks[i] = torch.clamp(masks[i], max=1)


def layer_correspondence(orig_len, target_len):
    divisor = target_len / orig_len
    list1 = np.zeros(orig_len)
    for k in range(target_len):
        list1[int(k / divisor)] = k
    return list1


def zero_upsample(input_tensor, target_size):
    """
    Upsamples an input tensor to a target size (c, d) using zero insertion instead of interpolation.

    Args:
        input_tensor (torch.Tensor): Input tensor containing labels of shape (B, H, W) or (B, W). No channels expected.
        target_size (tuple): Target output size (target_H, target_W).

    Returns:
        torch.Tensor: Upsampled tensor with zeros inserted.
    """
    if input_tensor.dim() == 2:  # If input is (B, W), add batch and channel dims
        B, W = input_tensor.shape

        target_W = target_size

        corr_W = layer_correspondence(W, target_W)

        # Create an upsampled tensor filled with zeros
        upsampled = torch.zeros(
            (B, target_W), dtype=input_tensor.dtype, device=input_tensor.device
        )

        # Fill the upsampled tensor with original values at appropriate locations
        upsampled[:, corr_W] = input_tensor
    elif input_tensor.dim() == 3:
        B, H, W = input_tensor.shape

        target_H, target_W = target_size

        corr_H = layer_correspondence(H, target_H)
        corr_W = layer_correspondence(W, target_W)

        # Create an upsampled tensor filled with zeros
        upsampled = torch.zeros(
            (B, target_H, target_W),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

        # Fill the upsampled tensor with original values at appropriate locations
        upsampled[:, corr_H, corr_W] = input_tensor

    return upsampled.squeeze()
