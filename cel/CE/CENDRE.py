import logging
import torch
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, HDBSCAN
from .ConceptExtraction import ConceptExtraction
import numpy as np
from ..utils.layers import getActivations, reshape_activations, getLayers, getMaxActivationShape
from ..utils.clustering import predict_concept, compute_elbow_lmethod, compute_second_derivative_elbow,distance_combined, explained_variation
from ..utils.gradients import get_gradients, get_classGradients
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from omegaconf import DictConfig
import os
from os.path import isfile
from ..utils.virtualLayers import (
    IFTLayer, STFTLayer, ISTFTLayer, DWTLayer, IDWTLayer, dwt_band_lengths, dwt_bands_to_image,
)
import json
from captum.attr import FeatureAblation
from torch import nn
from ..utils.misc import infer_shape
from collections.abc import Mapping
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# domains needing power_sensitivity()'s complex-to-magnitude gradient correction
COMPLEX_DOMAINS = ("fourier", "stft")


class CENDRE(ConceptExtraction):
    def __init__(
        self,
        dimension,
        model,
        layer_names,
        n_concepts: int = 0,
        ac_factor=0.1,
        n_clusters: int = 100,
        n_classes: int = 2,
        smooth_window=5,
        task="classification",
        binary_rep=True,
        mask_activation="relu",
        weighted_silhouette: bool = False,
        stft_n_fft: int = 64,
        stft_hop_length: int = 16,
        stft_win_length: int = 64,
        wavelet_name: str = "db4",
        wavelet_level: int = 3,
        **kwargs
    ):
        """Implementation of CENDRE method for concept extraction.
        CENDRE stands for Concept Extraction with Natural Domain Representations for Explanations.
        Args:
            dimension (int): Dimension of the input data.
            model (nn.Module): The model to be analyzed.
            layer_names (list): List of layer names to extract concepts from.
            n_concepts (int, optional): Number of concepts to extract. Defaults to 20.
            ac_factor (float, optional): Activation factor for clustering. Defaults to 0.1.
            n_clusters (int, optional): Number of clusters for clustering. Defaults to 100.
            task (str, optional): Task type ('classification' or 'regression'). Defaults to 'classification'.
            mask_activation (str): How to activate gradient masks: 'relu' (default) or 'abs'.
            stft_n_fft (int): FFT size for the 'stft' domain's STFTLayer/ISTFTLayer.
            stft_hop_length (int): Hop length for the 'stft' domain (>=50% overlap for NOLA reconstruction).
            stft_win_length (int): Window length for the 'stft' domain.
            wavelet_name (str): PyWavelets wavelet name for the 'wavelet' domain's DWTLayer/IDWTLayer.
            wavelet_level (int): Decomposition level for the 'wavelet' domain.
        """
        super().__init__(dimension)
        self.model = model
        self.test_layers = getLayers(model, layer_names=layer_names)
        self.task = task
        self.n_classes = n_classes
        # Hyperparameters for clustering
        self.n_concepts = n_concepts
        self.ac_factor = ac_factor
        self.n_clusters = n_clusters
        self.act_shape = None
        self.binary_filter = {}
        self.smooth_window = smooth_window
        self.binary_rep = binary_rep
        self.mask_activation = mask_activation
        self.weighted_silhouette = weighted_silhouette
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.wavelet_name = wavelet_name
        self.wavelet_level = wavelet_level


    def fit(self, dataloader, cfg_CE,  device="cpu",**kwargs):
        """Fit the CENDRE model to the data.
        Args:
            dataloader (DataLoader): DataLoader for the dataset.
            cfg_CE (DictConfig): Configuration for CENDRE.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        self.model.eval()
        
        # Calculate required parameters
        num_batches = np.ceil(cfg_CE.n_instances / cfg_CE.loader.batch_size)
        x_shape = infer_shape(dataloader)

        # Determine activation shapes
        batch = next(iter(dataloader))
        if isinstance(batch, dict):
            test_inputs = batch["x"][:4]
        else:
            test_inputs, _, _ = batch[:4]
        test_activations, _ = getActivations(
            self.model.to(device), self.test_layers, test_inputs.to(device)
        )
        self.act_shape = getMaxActivationShape(test_activations, self.dimension)

        # Concept clustering: Obtain centroids and aggregated centroids
        logger.info("Computing centroids (mode=%s)", cfg_CE.clusterMode)
        self.centroids, self.aggCentroids = self.get_centroids(
            dataloader, num_batches, cfg_CE.clusterMode, device=device
        )

        if not cfg_CE.clusterMode.split('_')[-1] == "IS":
            logger.info("Computing centroid agglomeration")
            n_before = len(self.centroids)
            self.aggCentroids, _ = self.get_aggCentroids(mode = cfg_CE.clusterMode)
            logger.info("Agglomeration: %d centroids -> %d concepts", n_before, self.n_concepts)

            if self.n_concepts == 0:
                logger.warning("No concepts found (all centroids noise). Saving empty state.")
                self.importancesCW = torch.zeros(self.n_classes, 0, x_shape[0])
                self.importances = {}
                return self.centroids, self.importancesCW

        # Calculate and scale importance scores
        logger.info("Computing importances")
        self.importancesCW = self.get_scaledImportances(
            dataloader=dataloader,
            n_channels=x_shape[0],
            num_batches=num_batches,
            scoreMode= cfg_CE.scoreMode,
            device=device,
            grad_noisetunnel= True,
            mask_noisetunnel= False,
            domain = 'input'
        )
        
        if cfg_CE.clusterMode.split('_')[-1] == "IS":
            logger.info("Computing centroid agglomeration (IS mode)")
            n_before = len(self.centroids)
            self.aggCentroids, self.importancesCW = self.get_aggCentroids(mode = cfg_CE.clusterMode)
            logger.info("Agglomeration: %d centroids -> %d concepts", n_before, self.n_concepts)

            if self.n_concepts == 0:
                logger.warning("No concepts found (all centroids noise). Saving empty state.")
                self.importancesCW = torch.zeros(self.n_classes, 0, x_shape[0])
                self.importances = {}
                return self.centroids, self.importancesCW

        # Sort concepts based on computed scores
        self.sortConcepts()
        self.importances = self.CW2single()
        
        return self.centroids, self.importancesCW

    def eval(self, test_inputs, device="cpu", outdevice="cpu", model=None, domain="input", maskMode="gradient", noisetunnel=True, noisetunnel_n=5, skip_model_transform=False, **kwargs):
        """Mask generation for individual instances. This can be used only after CE (fit) has been applied.
        Args:
            test_inputs (torch.Tensor): Input data for evaluation.
            domain (str, optional): Domain to evaluate on ('input', 'fourier', 'wavelet'). Defaults to 'input'.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
            skip_model_transform (bool, optional): If True, skip model and input transformations
                (use when the caller has already wrapped the model and transformed the inputs).
                Defaults to False.
        Returns:
            dict: Dictionary of masks for each concept.
            torch.Tensor: Model outputs for the input data.
        """
        if model is None: model = self.model
        if not skip_model_transform:
            model = self.model_transformation(model, domain, device, length=test_inputs.shape[-1])

        if not noisetunnel: noisetunnel_n = 1
        test_masks={}
        for i in range(noisetunnel_n):
            if noisetunnel: test_inputs_i = add_white_noise(test_inputs, std=0.01)
            else: test_inputs_i = test_inputs
            if not skip_model_transform:
                test_inputs_i = self.input_transformation(test_inputs_i, domain, device)
            else:
                test_inputs_i = test_inputs_i.to(device)
            if test_inputs_i.grad_fn==None: test_inputs_i.requires_grad = True

            test_activations, test_outputs = getActivations(
                model, self.test_layers, test_inputs_i
            )
            if not self.act_shape: self.act_shape = getMaxActivationShape(
                test_activations, self.dimension)
            test_reshaped_activations = reshape_activations(
                test_activations, self.act_shape, device=device
            )
            test_lads = self.get_lads(test_reshaped_activations)
            test_labels, test_distances = self.predict_labels(test_lads, device=device)
            test_masks_i = self.label2GradMask(
                labels=test_labels,
                distance=test_distances,
                domain=domain,
                inputs=test_inputs_i,
                device=device,
            )
            if not test_masks: test_masks = {k: v.clone()/noisetunnel_n for k, v in test_masks_i.items()}
            else: test_masks = {k: v/noisetunnel_n + test_masks[k] for k, v in test_masks_i.items()}

        if outdevice == "cpu":
            test_masks = {k: v.detach().to("cpu") for k, v in test_masks.items()}

        # Otsu thresholding (before normalization, on raw gradient values)
        if maskMode == "binary":
            for k, x in test_masks.items():
                vals = x.detach().cpu().numpy().ravel()
                vals_pos = vals[vals > 0]
                thr = self.otsu_threshold(vals_pos)
                test_masks[k] = (x > thr).to(dtype=x.dtype)
        elif maskMode == "filtered":
            # Otsu-zeroed but values preserved above threshold; useful for plotting.
            for k, x in test_masks.items():
                vals = x.detach().cpu().numpy().ravel()
                vals_pos = vals[vals > 0]
                thr = self.otsu_threshold(vals_pos)
                test_masks[k] = x * (x > thr).to(dtype=x.dtype)

        return test_masks, test_outputs
    
    def naming(self, CE_cfg: DictConfig):
        mode = CE_cfg.clusterMode.split('_')
        if mode[-1] == 'IS': modeSuffix = "IS"
        else: modeSuffix = ""
        if mode[0] in  ["kMeans"]:
            AC_id= f"{mode[0][0]}{modeSuffix}_{CE_cfg.n_concepts}"
        elif mode[0] in  ["ACnConcepts"]:
            AC_id= f"{mode[0][0]}{modeSuffix}_{CE_cfg.n_clusters}_{CE_cfg.n_concepts}"
        elif mode[0] in  ["ACelbow", "ACsilhouette", "HDBSCAN"]:
            AC_id= f"{mode[0][0]}{modeSuffix}_{CE_cfg.n_clusters}"
        elif mode[0] in ["ACthreshold"]:
            AC_id= f"{mode[0][0]}{modeSuffix}_{CE_cfg.ac_factor}"
        # Append variant tags for non-default settings
        if getattr(self, 'mask_activation', 'relu') != 'relu':
            AC_id += f'_ma{self.mask_activation}'
        return AC_id


    def save(self, CE_cfg: DictConfig): 
        """Save the fitted CENDRE model and its parameters.
        Args:
            CE_cfg (DictConfig): Configuration for CENDRE.
        """
        save_path = CE_cfg.save_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        AC_id = self.naming(CE_cfg)
        
        for attr in ("aggCentroids", "centroids", "importancesCW"):
                if not hasattr(self, attr):
                    raise RuntimeError(f"CENDRE not fit: missing attribute {attr}")
        
        try:
            np.save(f"{save_path}/ACcentroids_{AC_id}.npy", self.aggCentroids)
            np.save(f"{save_path}/centroids_{AC_id}.npy", self.centroids)
            with open(
                    f"{save_path}/importancesCW_{AC_id}.json", "w"
                ) as fp:
                    json.dump(self._importancesCW_to_dict(self.importancesCW), fp)
            with open(
                    f"{save_path}/importances_{AC_id}.json", "w"
                ) as fp:
                    json.dump(self.CW2single(), fp)
            with open(
                    f"{save_path}/binary_filter_{AC_id}.json", "w"
                ) as fp:
                    json.dump(self.binary_filter, fp)
            logger.info("Results saved at: %s", save_path)
        except Exception as e:
            logger.error("Error saving CENDRE concepts: %r", e)
            raise

    def load(self, CE_cfg: DictConfig,  device='cpu'):
        """Load the fitted CENDRE model and its parameters.
        Args:
            CE_cfg (DictConfig): Configuration for CENDRE.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cpu'.
        """
        save_path = CE_cfg.save_dir
        AC_id = self.naming(CE_cfg)

        try:
            self.centroids = np.load(
                f"{save_path}/centroids_{AC_id}.npy"
            )

            self.aggCentroids = np.load(
                f"{save_path}/ACcentroids_{AC_id}.npy"
            )
            self.n_concepts = self.aggCentroids.max() +1
            self.n_clusters = len(self.aggCentroids)

            if isfile(f"{save_path}/binary_filter_{AC_id}.json"):
                with open(
                    f"{save_path}/binary_filter_{AC_id}.json"
                ) as json_file:
                    self.binary_filter = json.load(json_file)
            else: 
                self.binary_filter = {domain:np.zeros(self.n_concepts) for domain in CE_cfg.domain}
                logger.warning("For proper binary masks, run self.get_quantile(dataloader)")

            with open(
                f"{save_path}/importances_{AC_id}.json"
            ) as json_file:
                self.importances = json.load(json_file)
            with open(
                f"{save_path}/importancesCW_{AC_id}.json"
            ) as json_file:
                self.importancesCW = self._importancesCW_from_dict(json.load(json_file))

        except FileNotFoundError as e:
            logger.error("Wrong file or file path: %s", e)

        

    def check_existence(self, CE_cfg: DictConfig):
        """Check if the fitted CENDRE model and its parameters exist.
        Args:
            CE_cfg (DictConfig): Configuration for CENDRE.
        Returns:
            bool: True if the model and parameters exist, False otherwise.
         """
        save_path = CE_cfg.save_dir
        AC_id = self.naming(CE_cfg)
        fitting = (
            isfile(f"{save_path}/importancesCW_{AC_id}.json")
            and isfile(f"{save_path}/centroids_{AC_id}.npy")
            and isfile(f"{save_path}/ACcentroids_{AC_id}.npy")
            and isfile(f"{save_path}/binary_filter_{AC_id}.json")
        )
        report = (isfile(f"{save_path}/concepts_{self.n_concepts}_{AC_id}_f.pdf")
                  and isfile(f"{save_path}/concepts_{self.n_concepts}_{AC_id}_ts.pdf"))
        return fitting, report

    def _compute_input_gradients(self, inputs, domain, device):
        """Compute per-class gradients of model output wrt (domain-transformed) input.
        Returns numpy array of shape (n_classes, batch, channels, time)."""
        model = self.model_transformation(self.model, domain=domain, device=device, length=inputs.shape[-1])
        model.eval()

        x = self.input_transformation(inputs.clone(), domain=domain, device=device)
        x.requires_grad_(True)

        outputs = model(x)
        grads = get_classGradients(outputs, x, out_format='astensor')
        # shape: (n_classes, batch, channels, time)
        if domain in COMPLEX_DOMAINS:
            grads = torch.stack([power_sensitivity(grads[c], x) for c in range(grads.shape[0])])

        return grads.detach().cpu().numpy()

    def report(
        self,
        test_masks,
        CE_cfg: DictConfig,
        test_inputs,
        test_outputs,
        domain: str = "input",
        c_indexes=None,
        fs: int = 1,
        DS_cfg=None,
        test_truth=None,
        importance_thr: float = 0.50,
        font_base: int = 5,
        visualization_mode= 'perceptual',
        destandardize=True,
        plot_gradients=False,
        classwise_IS: bool = True,
        use_clamp: bool = True,
        show: bool = False,
        IS_scientific: bool = False,
        **kwargs
    ):
        """Visualization of the concepts in the form of a report.
        Generates a grid where each row is a concept and each column is a sample.
        Gradient masks are overlaid on the signal as colored alpha bands. Because
        human perception compresses transparency non-linearly, the alpha values
        are scaled to make differences visually faithful to the underlying data.
        Args:
            visualization_mode (str, optional): Alpha scaling mode. Default is
                'perceptual': smooths the raw mask with a moving average and renormalises            samples in the same row.
                Other options: 'sigmoid', 'log', 'exp', 'smoothlog', 'smoothed',
                'binary'.
            test_masks (dict): Dictionary of masks for each concept.
            CE_cfg (DictConfig): Configuration for CENDRE.
            test_inputs (torch.Tensor): Input data for evaluation.
            test_outputs (torch.Tensor): Model outputs for the input data.
            c_indexes (list, optional): List of concept indexes to visualize. Defaults to None.
            fs (int, optional): Sampling frequency. Defaults to 1.
            test_truth (torch.Tensor, optional): Ground truth labels for the input data. Defaults to None.
        """
        # ------------------------------------------------------------
        # ----- 0. Domain-specific processing ------------------------
        # ------------------------------------------------------------
        # Optionally compute per-class input gradients
        input_grads = None
        if plot_gradients:
            device = next(self.model.parameters()).device
            input_grads = self._compute_input_gradients(test_inputs, domain, device)

        # x_scale_freq/y_ticks/y_ticklabels are only set for the 2D time-frequency domains
        # (stft, wavelet); see _plot_time_frequency_report.
        x_scale_freq = None
        y_ticks = None
        y_ticklabels = None
        if domain == 'fourier':
            test_inputs = torch.abs(torch.fft.rfft(test_inputs)) ** 2
            x_scale = fs * np.arange(0, test_inputs.shape[-1]) / (2 * test_inputs.shape[-1])
            plot_type = 'f'
            x_scale_grad = np.arange(0, input_grads.shape[-1]) / fs if plot_gradients else None
        elif domain == 'stft':
            stft = STFTLayer(self.stft_n_fft, self.stft_hop_length, self.stft_win_length)
            test_inputs = torch.abs(stft(test_inputs)) ** 2  # (B, C, F, Frames)
            n_freq, n_frames = test_inputs.shape[-2], test_inputs.shape[-1]
            x_scale = self.stft_hop_length * np.arange(0, n_frames) / fs      # frame times [s]
            x_scale_freq = fs * np.arange(0, n_freq) / self.stft_n_fft        # bin freqs [Hz]
            plot_type = 'stft'
            x_scale_grad = None
            if plot_gradients:
                logger.warning("plot_gradients is not supported for domain='stft'; ignoring.")
                plot_gradients = False
        elif domain == 'wavelet':
            length = test_inputs.shape[-1]
            band_lengths = dwt_band_lengths(length, self.wavelet_name, self.wavelet_level)
            dwt = DWTLayer(self.wavelet_name, self.wavelet_level)
            # each of the level+1 dyadic bands upsampled onto a common time axis -> a genuine
            # (band, time) time-frequency image, same shape role as the stft spectrogram
            test_inputs = dwt_bands_to_image(dwt(test_inputs), band_lengths) ** 2
            test_masks = {k: dwt_bands_to_image(v, band_lengths) for k, v in test_masks.items()}
            n_bands, n_frames = test_inputs.shape[-2], test_inputs.shape[-1]
            x_scale = np.arange(0, n_frames) * length / n_frames / fs
            x_scale_freq = np.arange(0, n_bands + 1)  # row-index edges (extent only)
            if fs and fs != 1:
                # dyadic octave-band edges: [0, fs/2^(L+1), fs/2^L, ..., fs/2] for L=wavelet_level
                edges = [0.0] + [fs / (2 ** (self.wavelet_level + 2 - i)) for i in range(1, self.wavelet_level + 2)]
                y_ticks = np.arange(n_bands + 1)
                y_ticklabels = [f"{e:.0f}" for e in edges]
            plot_type = 'wt'
            x_scale_grad = None
            if plot_gradients:
                logger.warning("plot_gradients is not supported for domain='wavelet'; ignoring.")
                plot_gradients = False
        else:
            x_scale = np.arange(0, test_inputs.shape[-1]) / fs
            plot_type = 'ts'
            x_scale_grad = x_scale

        test_inputs = test_inputs.cpu().detach().numpy()
        test_outputs = test_outputs.cpu().detach().numpy()

        # DS-meta
        c_names     = getattr(DS_cfg, "xcols", None)
        label_names = getattr(DS_cfg, "ycols", None)
        ds_mean     = getattr(DS_cfg, "mean", None)
        ds_std      = getattr(DS_cfg, "std",  None)

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

        if visualization_mode is None:
            visualization_mode = CE_cfg.maskMode
        is_fmt = ".2e" if IS_scientific else ".1f"
        # ------------------------------------------------------------
        # ----- 1. PICK CHANNELS TO DISPLAY --------------------------
        # ------------------------------------------------------------
        # Setup variables
        importancesCW = self.importancesCW
        x_shape = test_inputs.shape
        batch_size = x_shape[0]
        n_channels = x_shape[1]
        c_indexes = c_indexes or range(importancesCW.shape[1])

        keep_channels = {}
        if n_channels > 1:
            for k in c_indexes:
                sel = set()
                candidate=None
                if classwise_IS:
                    for cl in range(importancesCW.shape[0]):
                        imp_vec = importancesCW[cl, k].numpy()  # shape (n_channels,)
                        abs_vec = np.abs(imp_vec)
                        sel.update(set(np.where(abs_vec >= importance_thr)[0]))
                else:
                    imp_vec = np.asarray(self.importances[str(k)])
                    abs_vec = np.abs(imp_vec)
                    sel.update(set(np.where(abs_vec >= importance_thr)[0]))
                if not sel:
                    if classwise_IS:
                        running_max = -np.inf
                        candidate = None
                        for cl in range(importancesCW.shape[0]):
                            imp_vec = importancesCW[cl, k].numpy()
                            abs_vec = np.abs(imp_vec)
                            m = abs_vec.max()
                            if m > running_max:
                                running_max = m
                                candidate = int(abs_vec.argmax())
                    else:
                        imp_vec = np.asarray(self.importances[str(k)])
                        candidate = int(np.abs(imp_vec).argmax())
                    if candidate is not None:
                        sel.add(candidate)
                keep_channels[k] = sel
        else:
            keep_channels = {k: np.arange(n_channels) for k in c_indexes}
        n_rows = sum(len(v) for v in keep_channels.values())
        
        # ------------------------------------------------------------
        # ----- 2. FIGURE & COLOURS ---------------------------------
        # ------------------------------------------------------------
        # Prepare save directory
        save_path = CE_cfg.save_dir
        os.makedirs(save_path, exist_ok=True)

        # Per-(concept, channel) y-range for mask line plots
        mask_ylims = {}
        for k in c_indexes:
            for ch in keep_channels[k]:
                vals = test_masks[k][:, ch].detach().cpu().numpy()
                lo, hi = float(vals.min()), float(vals.max())
                margin = 0.05 * max(abs(hi - lo), 1e-8)
                mask_ylims[(k, ch)] = (lo - margin, hi + margin)

        # Determine figure layout
        if domain in ('stft', 'wavelet'):
            rows_per_concept = 1  # mask-only heatmap panel; see _plot_time_frequency_report
            height_ratios = [1] * n_rows
            fig_height = 3 * n_rows
        else:
            rows_per_concept = 3 if plot_gradients else 2
            height_ratios = [3, 1, 1] * n_rows if plot_gradients else [3, 1] * n_rows
            fig_height = 4 * n_rows if plot_gradients else 3 * n_rows
        fig, axs = plt.subplots(
            n_rows * rows_per_concept, batch_size,
            figsize=(4 * batch_size, fig_height),
            squeeze=False,
            gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.05}
        )
        plt.subplots_adjust(wspace=0.05)

        # Color mapping for concepts
        cmap = plt.get_cmap("viridis")(np.linspace(0.0, 0.75, self.n_concepts))
        colors = 0.6 * cmap + 0.4 * np.ones((self.n_concepts, 4))


        # line width for mask stripes (scaled with axes width)
        fig.canvas.draw()
        lw = axs[0, 0].get_window_extent().width * 0.75 / len(x_scale) * 1.01
        # 0.75 points per pixel in the plot, (.01 extra for aesthetics)

        # fonts
        font_base   = font_base or 10
        font_title  = font_base * 3.8
        font_label  = font_base * 3.4
        font_ticks  = font_base * 3.0

        # ------------------------------------------------------------
        # ----- 3. PLOT ------------------------------------------------
        # ------------------------------------------------------------
        # Importance Score header above the importance-score column
        axs[0, 0].annotate("Importance\nScore", xy=(-0.43, 1.05),
                           xycoords="axes fraction", ha="center", va="bottom",
                           fontsize=font_title, fontweight="bold")

        interesting_out = CE_cfg.multilabel_out or 0
        if self.task == "multilabel classification":
            test_truth = test_truth[:, interesting_out]

        if domain in ('stft', 'wavelet'):
            self._plot_time_frequency_report(
                fig, axs, c_indexes, keep_channels, importancesCW, test_masks, test_inputs,
                test_outputs, test_truth, colors, rows_per_concept, n_rows, x_scale, x_scale_freq,
                font_title, font_label, font_ticks, is_fmt, classwise_IS, importance_thr,
                visualization_mode, use_clamp, domain, c_names, label_names, destandardise,
                interesting_out, fs, y_ticks, y_ticklabels,
            )
            AC_id = self.naming(CE_cfg)
            plt.savefig(f"{save_path}/concepts_{self.n_concepts}_{AC_id}_{plot_type}.png")
            plt.savefig(f"{save_path}/concepts_{self.n_concepts}_{AC_id}_{plot_type}.pdf", format="pdf", bbox_inches="tight")
            if show:
                try:
                    from IPython.display import display
                    display(fig)
                except ImportError:
                    pass
            plt.close(fig)
            logger.info("Report saved at: %s/concepts_%s_%s_%s.png/pdf", save_path, self.n_concepts, AC_id, plot_type)
            return

        row = 0
        for j, k in enumerate(c_indexes):
            concept_max = None
            concept_smooth_max = None
            if not use_clamp:
                concept_vals = test_masks[k].detach().cpu().numpy()
                concept_max = float(np.abs(concept_vals).max())
            if visualization_mode == 'perceptual':
                concept_smooth_vals = moving_average(
                    test_masks[k].detach().cpu().numpy(),
                    self.smooth_window,
                    asnp=True,
                )
                concept_smooth_max = float(np.abs(concept_smooth_vals).max())
            for ch in keep_channels[k]:
                if classwise_IS:
                    ch_importances = {cl: importancesCW[cl, k, ch].item() for cl in range(importancesCW.shape[0])}
                    displ_importance = {key:val for key,val in ch_importances.items() if abs(val) > importance_thr}
                    if not displ_importance:
                        max_cl = max(ch_importances,  key=lambda cl: abs(ch_importances[cl]))
                        displ_importance = {max_cl: ch_importances[max_cl]}
                else:
                    displ_importance = {None: float(self.importances[str(k)][ch])}
                # channel-specific y-range
                chan_sig = test_inputs[:, ch, :]
                y_min = chan_sig.min() - abs(chan_sig.min()) * 0.1
                y_max = chan_sig.max() + abs(chan_sig.max()) * 0.1
                y_range = (destandardise(y_min.item(),ch,c_names), destandardise(y_max.item(),ch,c_names))


                for i in range(batch_size):
                    ax = axs[row * rows_per_concept, i]
                    ax_line = axs[row * rows_per_concept + 1, i]

                    if self.task == "multilabel classification":
                        truth = test_truth[i, interesting_out].item()
                        pred  = (test_outputs[i] > 0.5)[interesting_out].item()
                        title = rf"$y$: {truth} $\quad\hat y$: {pred}"
                    elif self.task == "regression":
                        truth = destandardise(test_truth[i, interesting_out].item(),
                                            interesting_out, label_names)
                        pred  = destandardise(test_outputs[i, interesting_out].item(),
                                            interesting_out, label_names)
                        title = rf"$y$: {truth:.2f} $\quad\hat y$: {pred:.2f}"
                    else:   # multiclass
                        truth = test_truth[i].item()
                        pred = np.argmax(test_outputs[i]).item() 
                        title = rf"$y$: {truth:.0f} $\quad\hat y$: {pred:.0f}"
                    axs[0, i].set_title(title, fontsize=font_title, pad=20)  # top row only
                
                    # Overlay mask as vertical lines
                    # ---------- MASK STRIPES ----------
                    mask = np.asarray(test_masks[k][i,ch].detach().numpy())
                    # Always normalize for alpha: use dataset-level clamp to map to [0,1]
                    if use_clamp:
                        bf = self.binary_filter.get(domain, {})
                        asclamp_list = bf.get('asclamp', None) if isinstance(bf, dict) else None
                        if asclamp_list is not None:
                            clamp_val = float(asclamp_list[int(k)]) if int(k) < len(asclamp_list) else 0.0
                            if clamp_val > 0:
                                mask = mask / clamp_val
                    elif concept_max is not None:
                        mask = mask / concept_max
                    mask = np.clip(mask, 0.0, 1.0)

                    # Perceptual scaling for alpha channel
                    if visualization_mode == 'sigmoid':
                        mask = sigmoid_scaling(mask)
                    elif visualization_mode == 'log':
                        mask = log_scaling(mask)
                    elif visualization_mode == 'exp':
                        mask = exp_scaling(mask)
                    elif visualization_mode == 'smoothlog':
                        mask = log_scaling(mask)
                        mask = moving_average(mask, self.smooth_window, asnp=True)
                    elif visualization_mode == 'smoothed':
                        mask = moving_average(mask, self.smooth_window, asnp=True)
                    elif visualization_mode == 'binary':
                        mask = (mask > 0).astype(mask.dtype, copy=False)
                    elif visualization_mode == 'perceptual':
                        mask = np.asarray(test_masks[k][i, ch].detach().cpu().numpy())
                        mask = moving_average(mask, self.smooth_window, asnp=True)
                        if concept_smooth_max and concept_smooth_max > 0:
                            mask = mask / concept_smooth_max
                        

                    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
                    mask = np.clip(mask, 0.0, 1.0)

                    positions = np.flatnonzero(mask)

                    if len(positions) > 0:
                        alphas = np.asarray(mask[positions], dtype=np.float64)
                        alphas = np.nan_to_num(alphas, nan=0.0, posinf=1.0, neginf=0.0)
                        alphas = np.clip(alphas, 0.0, 1.0)

                        # Create line segments [(x1, y1), (x2, y2)] for each position
                        segments = np.array([
                            [(x_scale[p], y_range[0]), (x_scale[p], y_range[1])]
                            for p in positions
                        ])

                        # Set RGBA colors with alpha per line
                        base_color = colors[j]
                        rgba_colors = np.tile(base_color, (len(positions), 1))
                        rgba_colors[:, 3] = alphas

                        # Create and add the LineCollection
                        line_collection = LineCollection(
                            segments,
                            colors=rgba_colors,
                            linewidths=lw,
                        )
                        ax.add_collection(line_collection)

                    # ---------- SIGNAL ----------
                    # Plot input data
                    ax.plot(
                        x_scale,
                        destandardise(test_inputs[i, ch], ch, c_names),
                        color="black",
                        linewidth=1
                    )
                    ax.set_ylim(y_range)
                    ax.tick_params(axis='both', labelsize=font_ticks)
                    ax.tick_params(labelbottom=False)  # x labels on line plot below

                    # Remove redundant ticks
                    if i > 0:
                        ax.set_yticks([])

                    # ---------- LINE PLOT (below) ----------
                    raw_mask = test_masks[k][i, ch].detach().cpu().numpy()
                    ax_line.plot(x_scale, raw_mask, color=colors[j], linewidth=1)
                    ax_line.set_ylim(mask_ylims[(k, ch)])
                    ax_line.set_xlim(ax.get_xlim())
                    ax_line.tick_params(axis='both', labelsize=font_ticks * 0.7)
                    if i > 0:
                        ax_line.set_yticks([])

                    if plot_gradients:
                        ax_line.tick_params(labelbottom=False)  # gradient row is below
                        # ---------- GRADIENT PLOT (bottom) ----------
                        ax_grad = axs[row * rows_per_concept + 2, i]
                        n_outputs = input_grads.shape[0]
                        grad_cmap = plt.get_cmap("tab10")
                        for ci in range(n_outputs):
                            grad_line = input_grads[ci, i, ch, :]
                            lbl = label_names[ci] if label_names and ci < len(label_names) else f"out {ci}"
                            ax_grad.plot(x_scale_grad, grad_line, color=grad_cmap(ci % 10),
                                         linewidth=0.8, label=lbl if (i == 0 and row == 0) else None)
                        ax_grad.set_xlim(ax.get_xlim())
                        ax_grad.tick_params(axis='both', labelsize=font_ticks * 0.7)
                        if i > 0:
                            ax_grad.set_yticks([])
                        if row < n_rows - 1:
                            ax_grad.tick_params(labelbottom=False)
                        if i == 0 and row == 0:
                            ax_grad.legend(fontsize=font_ticks * 0.6, loc='upper left',
                                           ncol=min(n_outputs, 4))
                    else:
                        if row < n_rows - 1:
                            ax_line.tick_params(labelbottom=False)

                    # ---------- LABELS ----------
                    if i == 0:
                        ch_name = c_names[int(ch)] if c_names else f"ch{str(ch)}"
                        ylab = f"$C_{{{k}}}$" if n_channels == 1 else f"$C_{{{k}}}$ | {ch_name}"
                        for lab in (displ_importance):
                            val = float(displ_importance[lab])
                            if classwise_IS:
                                ylab += f"\n$I_{{{lab}}} = {val:{is_fmt}}$"
                            else:
                                ylab += f"\n$I = {val:{is_fmt}}$"
                        ax.set_ylabel(
                            ylab,
                            fontsize=font_label,
                            labelpad=font_label * 0.3
                        )
                        ax.get_yaxis().set_label_coords(-0.35, 0.5)
                row += 1

        # Single centered x-axis label, anchored a fixed distance below the bottom-most subplot
        # (offset in points so it's constant in absolute terms regardless of figure height)
        if fs and fs != 1:
            xlab = "Frequency [Hz]" if domain == 'fourier' else "Time [s]"
            offset_fig = 28 / 72 / fig.get_figheight()  # 28pt below the axes
            y_bottom = axs[-1, 0].get_position().y0
            fig.text(0.5, y_bottom - offset_fig, xlab,
                     ha='center', va='top', fontsize=font_label * 0.8)

        # ------------------------------------------------------------
        # ----- 4. SAVE ------------------------------------------------
        # ------------------------------------------------------------
        AC_id = self.naming(CE_cfg)
        plt.savefig(f"{save_path}/concepts_{self.n_concepts}_{AC_id}_{plot_type}.png")
        plt.savefig(f"{save_path}/concepts_{self.n_concepts}_{AC_id}_{plot_type}.pdf", format="pdf", bbox_inches="tight")
        if show:
            try:
                from IPython.display import display
                display(fig)
            except ImportError:
                pass
        plt.close(fig)

        logger.info("Report saved at: %s/concepts_%s_%s_%s.png/pdf", save_path, self.n_concepts, AC_id, plot_type)
        
        # ------------------------------------------------------------
        # ----- 5. SAVE AS WAV ------------------------------------------------
        # ------------------------------------------------------------
        if domain == 'fourier' and CE_cfg.create_audio_files:

            mask_keys = list(test_masks.keys())

            maxvaluesperkey = {}
            for k in mask_keys:
                maxvaluesperkey[k] = max(float(test_masks[k].max()), 1e-8)

            import soundfile as sf
            audio_dir = f"{save_path}/audio_files_{self.n_concepts}"
            from pathlib import Path
            audio_dir = Path(audio_dir)  # Convert string to Path
            audio_dir.mkdir(parents=True, exist_ok=True)
            sample_rate = CE_cfg.sampling_rate if hasattr(CE_cfg, "sampling_rate") else 16000  # Default to 16 kHz if not specified
            if CE_cfg.sampling_rate is None:
                logger.warning("Sampling rate not specified in CE_cfg. Defaulting to 16000 Hz.")

            #Save audio files
            for i in range(test_inputs.shape[0]):
                # Prepare mono signal from original inputs
                signal = test_inputs[i].squeeze()
                
                # Save original input as wav file (in time domain)
                signal_time = np.fft.irfft(signal)  # Convert back to time domain from frequency domain
                input_path = audio_dir / f"test_input_{i}.wav"
                sf.write(input_path, signal_time, sample_rate)

                masked_time_signals = []
            
                
                for k in mask_keys:
                    mask = test_masks[k][i].squeeze().detach().cpu().numpy()  # Ensure to detach tensor before converting
                    
                    ##rescale mask between 0 and 1
                    mask = mask / maxvaluesperkey[k]
                    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
                    mask = np.clip(mask, 0.0, 1.0)

                    # Apply mask in frequency domain (assuming signal is already transformed)
                    masked_freq_domain_input = signal * mask  # Element-wise multiplication, scaling factor can be adjusted

                    back_to_time = np.fft.irfft(masked_freq_domain_input).real  # Back to time domain after masking
                    masked_time_signals.append(back_to_time)

                    masked_path = audio_dir / f"test_masked_input_{i}_{k}.wav"
                    sf.write(masked_path, back_to_time.astype(np.float32), sample_rate)

                # Concatenate all signals into one file and save it
                all_combined_signal = np.concatenate([signal_time] + masked_time_signals)
                all_path = audio_dir / f"test_all_{i}.wav"
                sf.write(all_path, all_combined_signal.astype(np.float32), sample_rate)

        return

    def _plot_time_frequency_report(
        self, fig, axs, c_indexes, keep_channels, importancesCW, test_masks, test_inputs,
        test_outputs, test_truth, colors, rows_per_concept, n_rows, x_scale_time, x_scale_freq,
        font_title, font_label, font_ticks, is_fmt, classwise_IS, importance_thr,
        visualization_mode, use_clamp, domain, c_names, label_names, destandardise,
        interesting_out, fs, y_ticks, y_ticklabels,
    ):
        """2D time-frequency + mask-overlay report panel for domain in ('stft', 'wavelet').

        Mirrors `report`'s 1D line+alpha-band layout (one concept/channel per row, one sample per
        column), but each panel is a heatmap (STFT spectrogram, or a DWT octave-band image built by
        `dwt_bands_to_image`) with the mask rendered as an alpha-weighted color overlay instead of
        vertical line stripes, since both representations are genuinely 2D (time x frequency).
        `y_ticks`/`y_ticklabels` override the default numeric y-axis for the wavelet domain, whose
        rows are non-uniform (octave) frequency bands rather than linearly spaced STFT bins.
        """
        n_channels = test_inputs.shape[1]
        batch_size = test_inputs.shape[0]
        extent = [x_scale_time[0], x_scale_time[-1], x_scale_freq[0], x_scale_freq[-1]]

        row = 0
        for j, k in enumerate(c_indexes):
            # concept_max/concept_smooth_max only fed the magnitude+overlay panel's alpha -- disabled
            # along with it (see the commented-out block below)
            # concept_max = None
            # concept_smooth_max = None
            # if not use_clamp:
            #     concept_vals = test_masks[k].detach().cpu().numpy()
            #     concept_max = float(np.abs(concept_vals).max())
            # if visualization_mode == 'perceptual':
            #     concept_smooth_vals = moving_average(
            #         test_masks[k].detach().cpu().numpy(), self.smooth_window, asnp=True,
            #     )
            #     concept_smooth_max = float(np.abs(concept_smooth_vals).max())

            for ch in keep_channels[k]:
                if classwise_IS:
                    ch_importances = {cl: importancesCW[cl, k, ch].item() for cl in range(importancesCW.shape[0])}
                    displ_importance = {key: val for key, val in ch_importances.items() if abs(val) > importance_thr}
                    if not displ_importance:
                        max_cl = max(ch_importances, key=lambda cl: abs(ch_importances[cl]))
                        displ_importance = {max_cl: ch_importances[max_cl]}
                else:
                    displ_importance = {None: float(self.importances[str(k)][ch])}

                for i in range(batch_size):
                    ax_mask = axs[row * rows_per_concept, i]
                    # ax = axs[row * rows_per_concept, i]  # magnitude+overlay panel -- disabled, see below

                    if self.task == "multilabel classification":
                        truth = test_truth[i, interesting_out].item()
                        pred = (test_outputs[i] > 0.5)[interesting_out].item()
                        title = rf"$y$: {truth} $\quad\hat y$: {pred}"
                    elif self.task == "regression":
                        truth = destandardise(test_truth[i, interesting_out].item(), interesting_out, label_names)
                        pred = destandardise(test_outputs[i, interesting_out].item(), interesting_out, label_names)
                        title = rf"$y$: {truth:.2f} $\quad\hat y$: {pred:.2f}"
                    else:
                        truth = test_truth[i].item()
                        pred = np.argmax(test_outputs[i]).item()
                        title = rf"$y$: {truth:.0f} $\quad\hat y$: {pred:.0f}"
                    axs[0, i].set_title(title, fontsize=font_title, pad=20)

                    # ---------- MASK ALPHA (same perceptual semantics as the 1D path, 2D shaped) ----------
                    # only feeds the magnitude+overlay panel below -- disabled along with it
                    # mask = np.asarray(test_masks[k][i, ch].detach().cpu().numpy())  # (F, Frames)
                    # if use_clamp:
                    #     bf = self.binary_filter.get(domain, {})
                    #     asclamp_list = bf.get('asclamp', None) if isinstance(bf, dict) else None
                    #     if asclamp_list is not None:
                    #         clamp_val = float(asclamp_list[int(k)]) if int(k) < len(asclamp_list) else 0.0
                    #         if clamp_val > 0:
                    #             mask = mask / clamp_val
                    # elif concept_max is not None:
                    #     mask = mask / concept_max
                    # mask = np.clip(mask, 0.0, 1.0)
                    #
                    # if visualization_mode == 'sigmoid':
                    #     mask = sigmoid_scaling(mask)
                    # elif visualization_mode == 'log':
                    #     mask = log_scaling(mask)
                    # elif visualization_mode == 'exp':
                    #     mask = exp_scaling(mask)
                    # elif visualization_mode == 'smoothlog':
                    #     mask = log_scaling(mask)
                    #     mask = moving_average(mask, self.smooth_window, asnp=True)
                    # elif visualization_mode == 'smoothed':
                    #     mask = moving_average(mask, self.smooth_window, asnp=True)
                    # elif visualization_mode == 'binary':
                    #     mask = (mask > 0).astype(mask.dtype, copy=False)
                    # elif visualization_mode == 'perceptual':
                    #     mask = np.asarray(test_masks[k][i, ch].detach().cpu().numpy())
                    #     mask = moving_average(mask, self.smooth_window, asnp=True)
                    #     if concept_smooth_max and concept_smooth_max > 0:
                    #         mask = mask / concept_smooth_max
                    #
                    # mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
                    # mask = np.clip(mask, 0.0, 1.0)

                    # ---------- SPECTROGRAM (background) + MASK (color overlay) ----------
                    # spec = test_inputs[i, ch]  # (F, Frames), already |STFT|^2
                    # ax.imshow(spec, aspect='auto', origin='lower', extent=extent, cmap='gray_r')
                    # rgba = np.tile(colors[j], (*mask.shape, 1))
                    # rgba[..., 3] = mask
                    # ax.imshow(rgba, aspect='auto', origin='lower', extent=extent, interpolation='nearest')
                    # ax.tick_params(axis='both', labelsize=font_ticks)
                    # ax.tick_params(labelbottom=False)
                    # if y_ticks is not None:
                    #     ax.set_yticks(y_ticks, y_ticklabels if i == 0 else [])
                    # elif i > 0:
                    #     ax.set_yticks([])

                    # ---------- RAW MASK ----------
                    raw_mask = test_masks[k][i, ch].detach().cpu().numpy()
                    ax_mask.imshow(raw_mask, aspect='auto', origin='lower', extent=extent, cmap='viridis')
                    ax_mask.tick_params(axis='both', labelsize=font_ticks * 0.7)
                    if y_ticks is not None:
                        ax_mask.set_yticks(y_ticks, y_ticklabels if i == 0 else [])
                    elif i > 0:
                        ax_mask.set_yticks([])
                    if row < n_rows - 1:
                        ax_mask.tick_params(labelbottom=False)

                    # ---------- LABELS ----------
                    if i == 0:
                        ch_name = c_names[int(ch)] if c_names else f"ch{str(ch)}"
                        ylab = f"$C_{{{k}}}$" if n_channels == 1 else f"$C_{{{k}}}$ | {ch_name}"
                        for lab in displ_importance:
                            val = float(displ_importance[lab])
                            if classwise_IS:
                                ylab += f"\n$I_{{{lab}}} = {val:{is_fmt}}$"
                            else:
                                ylab += f"\n$I = {val:{is_fmt}}$"
                        ax_mask.set_ylabel(f"Freq [Hz]\n{ylab}", fontsize=font_label, labelpad=font_label * 0.3)
                        ax_mask.get_yaxis().set_label_coords(-0.35, 0.5)
                row += 1

        if fs and fs != 1:
            offset_fig = 28 / 72 / fig.get_figheight()
            y_bottom = axs[-1, 0].get_position().y0
            fig.text(0.5, y_bottom - offset_fig, "Time [s]",
                     ha='center', va='top', fontsize=font_label * 0.8)

    def plot_CC(self, concept_correspondence, CE_cfg: DictConfig, fs=1, show: bool = False, xlabel: str = None, horizontal: bool = False, DS_cfg: DictConfig = None):
        """
        Plot concept correspondence for each concept and channel.

        Args:
            concept_correspondence (torch.Tensor): Tensor of shape [n_concepts, n_channels, timesteps].
            CE_cfg (DictConfig): Configuration for CENDRE, including save directory and naming settings.
            fs (int, optional): Sampling frequency. Defaults to 1.
            horizontal (bool, optional): If False (default), grid is concepts (rows) x channels (cols).
                If True, the grid is transposed: channels (rows) x concepts (cols), producing a wider
                figure. Saved with `_h` suffix to avoid overwriting the default variant.
            DS_cfg (DictConfig, optional): Dataset config; if provided, ``DS_cfg.xcols`` is used as
                the channel names (matching the convention used by ``report``). Falls back to
                ``Channel {ch}`` when unavailable.
        """
        # Prepare save directory
        save_path = CE_cfg.save_dir
        os.makedirs(save_path, exist_ok=True)

        # Extract shape
        n_concepts, n_channels, n_timesteps = concept_correspondence.shape

        # Channel names from dataset config (same convention as report())
        channel_names = getattr(DS_cfg, "xcols", None) if DS_cfg is not None else None
        def _ch_label(ch):
            if channel_names is not None and ch < len(channel_names):
                return str(channel_names[ch])
            return f"Channel {ch}"
        x_scale = fs * np.arange(0, n_timesteps) / (2 * n_timesteps)

        # Y-axis range with margin
        y_min = concept_correspondence.min().item()
        y_max = concept_correspondence.max().item()
        y_margin = 0.1 * abs(y_max - y_min)
        y_range = (y_min - y_margin, y_max + y_margin)

        # Color mapping (same as report) — cap at 0.75 to avoid the yellow end of viridis
        cmap = plt.get_cmap("viridis")(np.linspace(0.0, 0.75, n_concepts))
        colors = 0.6 * cmap + 0.4 * np.ones((n_concepts, 4))

        # Default x-axis label: CC is computed in the fourier domain, so when fs is configured
        # the x-axis represents real frequency in Hz.
        if xlabel is None and fs and fs != 1:
            xlabel = "Frequency [Hz]"

        # Grid layout
        if horizontal:
            nrows, ncols = n_channels, n_concepts
            figsize = (3 * n_concepts, 2.5 * n_channels)
        else:
            nrows, ncols = n_concepts, n_channels
            figsize = (3 * n_channels, 2.5 * n_concepts)

        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        for k in range(n_concepts):
            for ch in range(n_channels):
                row, col = (ch, k) if horizontal else (k, ch)
                ax = axs[row, col]
                ax.plot(
                    x_scale,
                    concept_correspondence[k, ch].cpu().detach().numpy(),
                    color=colors[k],
                    linewidth=1
                )
                ax.set_ylim(y_range)

                # Axis labeling — leftmost column gets ylabel, top row gets title
                if horizontal:
                    if col == 0:
                        ax.set_ylabel(_ch_label(ch), fontsize=15, labelpad=15)
                    if row == 0:
                        ax.set_title(f"C{k}", fontsize=15)
                else:
                    if col == 0:
                        ax.set_ylabel(f"C{k}", fontsize=15, labelpad=15)
                    if row == 0:
                        ax.set_title(_ch_label(ch), fontsize=15)

                # Simplify tick display
                if col > 0:
                    ax.set_yticks([])
                if row < nrows - 1:
                    ax.set_xticks([])
                ax.tick_params(axis='both', labelsize=8)

        plt.subplots_adjust(wspace=0.1, hspace=0.3)

        # Single centered x-axis label, anchored a fixed distance below the bottom-most subplot
        # (offset in points so it's constant in absolute terms regardless of figure height)
        if xlabel is not None:
            offset_fig = 28 / 72 / fig.get_figheight()  # 28pt below the axes
            y_bottom = axs[-1, 0].get_position().y0
            fig.text(0.5, y_bottom - offset_fig, xlabel,
                     ha='center', va='top', fontsize=14)

        AC_id = self.naming(CE_cfg)
        suffix = "_h" if horizontal else ""
        filename_base = f"{save_path}/conceptcorrespondence_{self.n_concepts}_{AC_id}_f{suffix}"
        plt.savefig(filename_base + ".png")
        plt.savefig(filename_base + ".pdf", format="pdf", bbox_inches="tight")
        if show:
            try:
                from IPython.display import display
                display(fig)
            except ImportError:
                pass
        plt.close(fig)

        logger.info("Concept correspondence plot saved at: %s.png/pdf", filename_base)
        return


    # Utilities

    def get_lads(self, aggregated_activations):
        """Lads extraction as described in Encoding of latent space section.
        Args:   
            aggregated_activations (torch.Tensor): Aggregated activations from the model.
        Returns:    
            torch.Tensor: Latent activation descriptors (LADs)."""
        lads = aggregated_activations
        if self.dimension == "2D" or self.dimension == "images":
            lads = torch.flatten(lads, -2, -1)
            lads = torch.permute(lads, (0, 2, 1))
        elif self.dimension == "1D" or self.dimension == "timeseries":
            lads = torch.permute(lads, (0, 2, 1))
        else:
            logger.error("LAD extraction error. Unknown dimension '%s'. Expected '1D', '2D', 'timeseries', or 'images'.", self.dimension)
        lads = torch.flatten(lads, 0, 1)
        self.lads_shape = lads.shape
        return lads

    def get_centroids(self, dataloader, num_batches, mode="kMeans", device="cpu"):
        mode = mode.split('_')
        if mode[0]=='kMeans': n_clusters=self.n_concepts
        else: n_clusters=self.n_clusters

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=0, batch_size=64, n_init="auto"
        )
        self.model.to(device)
        for i, batch in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            if i >= num_batches:
                break
            if isinstance(batch, dict):
                inputs = batch["x"]
            else:
                inputs, _, _ = batch
            inputs = inputs.to(device)
            test_activations, _ = getActivations(
                self.model, self.test_layers, inputs
            )
            reshaped_activations = reshape_activations(
                test_activations, self.act_shape, self.dimension, device=device
            )
            lads = self.get_lads(reshaped_activations)

            kmeans = kmeans.partial_fit(lads.detach().cpu().numpy())
        centroids = kmeans.cluster_centers_
        self.micro_counts = np.bincount(kmeans.labels_, minlength=n_clusters)
        logger.debug("Centroids extracted: shape=%s, n_clusters=%d", centroids.shape, n_clusters)

        labels = np.arange(n_clusters)
        return centroids, labels
    

    def get_aggCentroids(self, mode="kMeans"):
        mode = mode.split('_')
        logger.info("Selected agglomeration mode: %s", mode[0])
        centroids = self.centroids
        if mode[0]=='kMeans': 
            self.binary_filter = {'input' : [0.0] * int(self.aggCentroids.max() + 1)}
            return self.aggCentroids, getattr(self, "importancesCW", None)

        # Score for metric (standard or custom)
        if mode[-1] == "IS":
            conceptwise_IS = self.importancesCW.permute(1, 0, 2)  # [n_concepts, n_classes, n_channels]
            conceptwise_IS = conceptwise_IS.mean(dim=tuple(range(1, conceptwise_IS.ndim)))
            score = distance_combined(centroids, conceptwise_IS) 
            metric = "precomputed"
            linkage = "average"
        else:
            score = centroids
            metric = "cosine"
            linkage = "average"

        # Agglomerative clustering
        if mode[0] == "ACthreshold":
            ACthreshold = self.maxselfDist(centroids, metric=metric) * self.ac_factor
            logger.debug("ACthreshold: distance_threshold=%.4f (ac_factor=%.4f)", ACthreshold, self.ac_factor)
            agglomerator = AgglomerativeClustering(
                n_clusters=None, distance_threshold=ACthreshold, metric=metric, linkage = linkage
            )
            agglomerator.fit(centroids)
            self.n_concepts = agglomerator.n_clusters_
            labels = agglomerator.labels_

        elif mode[0] == "ACnConcepts":
            agglomerator = AgglomerativeClustering(n_clusters=self.n_concepts, metric=metric, linkage = linkage)
            agglomerator.fit(score)
            self.n_concepts = agglomerator.n_clusters_
            labels = agglomerator.labels_

        elif mode[0] == "ACelbow":
            n_candidates = np.arange(2, max(self.n_concepts,11), 1)
            labels_candidates = []
            total_dist = np.zeros(len(n_candidates))
            for i, n in enumerate(n_candidates):
                agglomerator = AgglomerativeClustering(
                    n_clusters=n, metric=metric, linkage=linkage
                )
                agglomerator.fit(score)
                labels_candidates.append(agglomerator.labels_)
                total_dist[i] = explained_variation(centroids, labels_candidates[i])
            chosen_n = compute_elbow_lmethod(n_candidates, total_dist)
            self.n_concepts = n_candidates[chosen_n]
            logger.info("Optimal concept number (elbow): %d (explained_var=%.4f)", n_candidates[chosen_n], total_dist[chosen_n])
            labels = labels_candidates[chosen_n]

        elif mode[0] == "ACsilhouette":
            n_candidates = np.arange(2, max(self.n_concepts,11), 1)
            labels_candidates = []
            total_dist = np.zeros(len(n_candidates))
            if self.weighted_silhouette:
                weights = getattr(self, 'micro_counts', None)
                if weights is None:
                    logger.warning("micro_counts not available; falling back to unweighted silhouette")
            else:
                weights = None
            for i, n in enumerate(n_candidates):
                agglomerator = AgglomerativeClustering(
                    n_clusters=n, metric=metric, linkage=linkage
                )
                agglomerator.fit(score)
                labels_candidates.append(agglomerator.labels_)
                if weights is not None:
                    per_sample = silhouette_samples(score, agglomerator.labels_, metric='cosine')
                    total_dist[i] = np.average(per_sample, weights=weights)
                else:
                    total_dist[i] = silhouette_score(score, agglomerator.labels_, metric='cosine')
            chosen_n = np.argmax(total_dist)
            self.n_concepts = n_candidates[chosen_n]
            logger.info("Optimal concept number (silhouette, weighted=%s): %d (score=%.4f)",
                        self.weighted_silhouette, n_candidates[chosen_n], total_dist[chosen_n])
            labels = labels_candidates[chosen_n]

        elif mode[0] == "HDBSCAN":
            agglomerator = HDBSCAN(min_cluster_size=5, metric=metric)
            agglomerator.fit(score)
            labels = agglomerator.labels_
            self.n_concepts = max(labels) + 1
            n_valid = int((labels >= 0).sum())
            n_noise = int((labels < 0).sum())
            logger.info("HDBSCAN: %d valid, %d noise out of %d centroids -> %d concepts", n_valid, n_noise, len(labels), self.n_concepts)
            if self.n_concepts == 0:
                logger.warning("HDBSCAN assigned all centroids to noise. No concepts found.")

        elif mode[0] == "interactive":
            raise NotImplementedError # TODO implement interactive clustering
        
        else:
            raise ValueError("Invalid mode. Choose'kMeans', 'ACthreshold', 'ACnClusters', 'ACelbow', 'ACsilhouette' or 'HDBSCAN'. " \
            "2-phase modes (all except kMeans) can additionally have'_IS' as suffix.")

        n_filter = int(labels.max() + 1) if labels.max() >= 0 else 0
        self.binary_filter = {'input' : [0.0] * n_filter}
        
        if mode[-1] != "IS":
            return labels, None
        
        # Adapt importance score to agglomeration
        importancesCW = self.importancesCW  # [n_classes, n_clusters, n_channels]
        n_classes, n_old, n_ch = importancesCW.shape
        n_concepts = self.n_concepts

        updated = torch.zeros(n_classes, n_concepts, n_ch)
        valid_mask = labels >= 0
        labels_count = torch.from_numpy(
            np.bincount(labels[valid_mask], minlength=n_concepts) if valid_mask.any() else np.ones(n_concepts)
        ).float()

        for cluster in range(n_old):
            if labels[cluster] < 0:
                continue  # skip HDBSCAN noise centroids
            updated[:, labels[cluster], :] += importancesCW[:, cluster, :] / labels_count[labels[cluster]]

        # rescale importances per class
        max_abs = updated.abs().amax(dim=(1, 2), keepdim=True)
        max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
        updated = updated / max_abs
        return labels, updated

    def predict_labels(self, lads, device="cpu"):
        """Predict labels for the given LADS using the trained model.
        Args:
            lads (torch.Tensor): Latent activation descriptors (LADs).
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
        Returns:
            tuple: Predicted labels and distances.
        """
        # Domain-specific processing
        centroids = torch.from_numpy(self.centroids).to(device)
        aggCentroids = torch.from_numpy(self.aggCentroids).to(device)
        lads = lads.to(device)

        labels, distances = predict_concept(
            lads, centroids
        )  # [n_lads], [n_lads,n_clusters]
        ### Determine concept labels ###
        concept_labels = aggCentroids[labels.tolist()]
        concept_labels[labels < 0] = -1

        ### Compute concept distances ###
        # For each instance: d_ik - mean_weight_to_other_concepts
        #   d_ik           = raw distance to assigned micro-centroid
        #   other concepts = decay(|d|) summed over centroids NOT in the same concept
        N, n_clusters = distances.shape
        n_concepts = self.n_concepts
        concept_dist = distances.new_full((N,), float("nan"))

        if n_concepts == 0:
            logger.warning("predict_labels: n_concepts=0. Returning NaN distances.")
            return concept_labels, concept_dist

        # --- Filter out HDBSCAN noise (label = -1) ---
        valid_centroids = aggCentroids >= 0                          # [n_clusters]
        valid_inst = (labels >= 0) & (labels < n_clusters)
        if not valid_inst.any() or not valid_centroids.any():
            logger.warning("predict_labels: no valid instances. Returning NaN distances.")
            return concept_labels, concept_dist

        # Work only with valid subset
        labels_v   = labels[valid_inst]                              # [Nv]
        concept_v  = aggCentroids[labels_v]                          # [Nv] concept id per instance
        dist_v_all = distances[valid_inst]                           # [Nv, n_clusters]

        # Secondary filter: exclude instances whose micro-centroid maps to noise macro-cluster
        valid_concept = concept_v >= 0
        if not valid_concept.all():
            idx_in_valid = torch.where(valid_inst)[0]
            noise_idx = idx_in_valid[~valid_concept]
            valid_inst[noise_idx] = False
            if not valid_inst.any():
                logger.warning("predict_labels: all instances map to noise macro-clusters. Returning NaN distances.")
                return concept_labels, concept_dist
            labels_v   = labels[valid_inst]
            concept_v  = aggCentroids[labels_v]
            dist_v_all = distances[valid_inst]

        # Weight to assigned micro-centroid: w(z, mu) = 1 / (1 + d)
        d_micro = dist_v_all.gather(1, labels_v.unsqueeze(1)).squeeze(1)  # [Nv]
        w_micro = 1.0 / (1.0 + d_micro)                                  # [Nv]

        # Compute weights per concept using decay function
        dist_valid   = dist_v_all[:, valid_centroids]                # [Nv, Cv]
        concept_ids  = aggCentroids[valid_centroids]                 # [Cv]
        n_valid      = int(valid_centroids.sum().item())
        Nv           = dist_valid.shape[0]

        weights = 1.0 / (1.0 + dist_valid)                              # [Nv, Cv]

        # Accumulate weights per concept
        concept_sums = distances.new_zeros((Nv, n_concepts))
        concept_sums.scatter_add_(
            1,
            concept_ids.unsqueeze(0).expand(Nv, -1),
            weights,
        )
        # How many centroids belong to each concept
        centroids_per_concept = torch.bincount(concept_ids, minlength=n_concepts)  # [M]

        # Per-instance: weight sum to own concept, count of other centroids
        own_sum   = concept_sums.gather(1, concept_v.unsqueeze(1)).squeeze(1)      # [Nv]
        own_count = centroids_per_concept[concept_v]                                # [Nv]
        other_count = (n_valid - own_count).clamp_min(1).to(distances.dtype)        # [Nv]

        # concept_dist = w_micro - mean_weight_to_other_concepts
        mean_other_w = (weights.sum(dim=1) - own_sum) / other_count
        concept_dist[valid_inst] = w_micro - mean_other_w

        return concept_labels, concept_dist

    def label2GradMask(self, labels, distance, inputs, domain='input', device="cpu"):
        """Mask generation, gradient based. See eq. 1.
        Args:   
            labels (torch.Tensor): Predicted labels for the input data.
            distance (torch.Tensor): Distances between LADS and centroids.
            inputs (torch.Tensor): Input data for evaluation.
            domain (str): Domain to evaluate on ('input', 'fourier', 'wavelet').
            device (str): Device to use ('cuda' or 'cpu').
        Returns:
            dict: Dictionary of masks for each concept.
            """
        labels = labels.to(device)
        distance = distance.to(device)
        inputs = inputs.to(device)
        masks = {}
        n_concepts = self.aggCentroids.max() + 1
        for k in range(n_concepts):
            self.cn = k
            binary_mask = labels == k
            instances = binary_mask.sum()
            if instances == 0: instances = 1
            distK = distance * binary_mask.detach()
            maskK = get_gradients(distK.sum()/instances, inputs)

            # complex to real magnitude representation
            if domain in COMPLEX_DOMAINS:
                maskK = power_sensitivity(maskK, inputs, positive=(self.mask_activation != 'abs'))
            if self.mask_activation == 'abs':
                masks[k] = maskK.abs()
            else:
                masks[k] = maskK * (maskK > 0)

        return masks

    def get_scaledImportances(
        self,
        dataloader,
        n_channels=1,
        num_batches=50,
        scoreMode="classwise",
        device="cpu",
        acc_dtype = torch.float32, 
        domain="input",
        mask_noisetunnel: bool = False,
        grad_noisetunnel: bool = False,
        grad_noisetunnel_n: int = 5,
        grad_noisetunnel_std: float = 0.1,
        importance_mode: str = "gradient", #"alt: gradient, inputxgradient"
    ):
        """Compute scaled importance scores for each concept. 
        Importance score computation in section Relevance computation. See Eq. 2,3, and 4. 
        Args:
            dataloader (DataLoader): DataLoader for the dataset.
            n_channels (int, optional): Number of channels in the input data. Defaults to 1.
            num_batches (int, optional): Number of batches to process. Defaults to 50.
            out_nr (int, optional): Output number for multilabel classification. Defaults to None.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
        Returns:
            dict: Dictionary of scaled importance scores for each concept.
        """
        # Initialize tensors for importance accumulation and instance counts
        if(scoreMode == "classwise" or self.task=='regression'):
            n_classes = self.n_classes
        else:
            n_classes = 1
        
        n_concepts = self.aggCentroids.max() + 1
        importancesCW = torch.zeros((n_classes, n_concepts, n_channels), device=device, dtype=acc_dtype)
        n_instances = torch.ones((n_classes, n_concepts, n_channels), device=device, dtype=acc_dtype)* 1e-8

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            # get the inputs; data is a list of [inputs, labels, indexes]
            if isinstance(batch, dict):
                inputs = batch["x"]
            else:
                inputs, _, _ = batch
            # rebuilt per batch: the inverse layer needs each batch's pre-transform signal length
            model = self.model_transformation(self.model, domain=domain, device=device, length=inputs.shape[-1])
            inputs=self.input_transformation(inputs, domain=domain, device=device)
            inputs = inputs.detach().requires_grad_(True)
            masks, _ = self.eval(inputs, model=model, domain=domain, maskMode="gradient", device=device, outdevice=device, noisetunnel=mask_noisetunnel, skip_model_transform=True)
            n_channels = inputs.shape[1]

            nt_n = grad_noisetunnel_n if grad_noisetunnel else 1
            grad_acc = None

            for _ in range(nt_n):
                # IMPORTANT: new leaf for gradient each time
                inputs_nt = add_white_noise(inputs, std=grad_noisetunnel_std) if grad_noisetunnel else inputs
                inputs_nt = inputs_nt.detach().requires_grad_(True)

                # forward on noisy input
                outputs_nt = model(inputs_nt)

                if scoreMode == "classwise":
                    g = get_classGradients(outputs_nt, inputs_nt, out_format="astensor")
                    if domain in COMPLEX_DOMAINS:
                        # sensitivity w.r.t. |X|^2 in the complex domain
                        g = power_sensitivity(g, inputs_nt)
                else:
                    g = get_gradients(outputs_nt, inputs_nt).unsqueeze(0)

                if importance_mode == "inputxgradient":
                    if domain in COMPLEX_DOMAINS:
                        g = g* abs(inputs_nt)**2
                    else: g = g * abs(inputs_nt)
                # accumulate
                grad_acc = g if grad_acc is None else (grad_acc + g)

                # avoid graph accumulation
                model.zero_grad(set_to_none=True)

            gradients = grad_acc / float(nt_n)

            spatial_dims = tuple(range(3, gradients.ndim))
            for c,m in masks.items():
                m = m.to(device)
                present = m.flatten(start_dim=2).any(dim=-1)            # (B, C)
                present = present.to(n_instances.dtype).sum(dim=0).unsqueeze(0)   # (1, C)
                n_instances[:, c, :] += present
                img_importance = (gradients * m.unsqueeze(0).expand(n_classes, *m.shape)).sum(dim=spatial_dims)
                importancesCW[:,c,:] += torch.sum(img_importance, dim=1)

        imp = (importancesCW / n_instances).detach().cpu()
        logger.debug("Importance tensor shape: %s", imp.shape)
        if imp.shape[1] == 0:  # no concepts
            logger.warning("No concepts found (n_concepts=0). Returning empty importance structure.")
            return torch.zeros(n_classes, 1, n_channels)  # Safeguard for HDBSCAN...
        max_abs = imp.abs().amax(dim=(0,1, 2))  # (n_classes,)
        max_abs_safe = torch.where(max_abs != 0, max_abs, torch.ones_like(max_abs))
        imp_scaled = imp / max_abs_safe.view(-1, 1, 1)
        return imp_scaled
    
    @staticmethod
    def otsu_threshold(vals):
        """Otsu's method: find the threshold that maximizes inter-class variance.
        Args:
            vals (np.ndarray): 1-D array of positive values.
        Returns:
            float: Optimal threshold separating background from signal.
        """
        if len(vals) < 2:
            return 0.0
        counts, bin_edges = np.histogram(vals, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total = counts.sum()
        best_var, best_thr = -1, bin_centers[0]
        w0, sum0, sum_total = 0, 0, (counts * bin_centers).sum()
        for c, b in zip(counts, bin_centers):
            w0 += c
            if w0 == 0 or w0 == total:
                continue
            sum0 += c * b
            mean0 = sum0 / w0
            mean1 = (sum_total - sum0) / (total - w0)
            var_between = w0 * (total - w0) * (mean0 - mean1) ** 2
            if var_between > best_var:
                best_var = var_between
                best_thr = b
        return float(best_thr)

    def get_clamp(self, dataloader, num_batches=1, device="cpu", domain="input", quantile=0.98):
        """Compute a dataset-level clamp (high percentile) per concept from gradient masks.
        Args:
            dataloader (DataLoader): DataLoader for the dataset.
            num_batches (int, optional): Number of batches to process. Defaults to 1.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cpu'.
            domain (str, optional): Domain to evaluate on ('input', 'fourier'). Defaults to 'input'.
            quantile (float, optional): Quantile value for the clamp. Defaults to 0.995.
        Returns:
            list[float]: Clamp value for each concept.
        """
        n_concepts = self.aggCentroids.max() + 1
        clamp = torch.zeros(n_concepts, dtype=float, device=device)
        inst_counter = torch.ones(n_concepts, dtype=float, device=device) * 1e-8

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            if isinstance(batch, dict):
                inputs = batch["x"]
            else:
                inputs, _, _ = batch

            test_masks, _ = self.eval(
                test_inputs=inputs,
                device=device,
                outdevice="cpu",
                domain=domain,
                maskMode="gradient",
                noisetunnel=False,
            )

            for k in range(n_concepts):
                kk = str(k) if str(k) in test_masks else k
                maskK = test_masks[kk]
                positive_gradients = maskK[maskK > 0]

                if positive_gradients.numel() > 0:
                    inst_counter[k] += 1
                    clamp[k] += float(torch.quantile(positive_gradients, quantile))

        return (clamp / inst_counter).cpu().tolist()

    def maxselfDist(self, y, metric='cosine'):
        """Maximum pairwise distance between centroids.
        Args:
            y (np.ndarray): Centroids, shape (N, D).
            metric (str): Distance metric (must match AgglomerativeClustering).
        Returns:
            float: Maximum pairwise distance."""
        return pdist(y, metric=metric).max()

    def sortConcepts(self):
        """Sort concepts based on their importance scores."""
        single_imp = self.CW2single()  # dict: concept -> [n_channels]

        # Ensure concepts are in numeric order: "0","1",...
        keys = sorted(single_imp.keys(), key=int)

        single_tensor = torch.tensor([single_imp[k] for k in keys])  # [n_concepts, n_channels]
        c_scores = single_tensor.abs().max(dim=1).values            # [n_concepts]

        perm = torch.argsort(c_scores, descending=True)               # perm[new] = old

        # inverse: inv[old] = new
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)

        # Remap macro concept ids of microcentroids: old -> new
        self.aggCentroids = np.array(inv[self.aggCentroids].cpu().tolist()).astype(np.int64) 

        # Reorder importancesCW to match new concept indexing
        self.importancesCW = self.importancesCW[:, perm, :]
        
    def CW2max(self):
        importances = self.importancesCW.permute(1, 0, 2)  # [n_concepts, n_classes, n_channels]
        idx = torch.abs(importances).argmax(dim=1, keepdim=True)
        importances = importances.gather(dim=1, index=idx).squeeze(1)
        importances = torch.max(importances, dim=1)[0]
        return {str(i): row for i, row in enumerate(importances.tolist())}

    def CW2single(self):
        importances = self.importancesCW.permute(1, 0, 2)  # [n_concepts, n_classes, n_channels]

        idx = importances.abs().argmax(dim=1, keepdim=True)
        top = importances.gather(dim=1, index=idx).squeeze(1)  # [n_concepts, n_channels_flat]

        if self.n_classes > 1:
            mean_others = (importances.sum(dim=1) - top) / (self.n_classes - 1)
            importances = abs((top - mean_others)/2)
        else:
            importances = abs(top)  # no “others” exist

        return {str(i): row for i, row in enumerate(importances.tolist())}

    def model_transformation(self, model, domain="input", device='cpu', length=None):
        """Transform the model to the desired domain.
        Args:
            model (nn.Module): The model to be transformed.
            domain (str): Domain to transform the model to ('input', 'fourier', 'stft', 'wavelet').
            device (str): Device to use ('cuda' or 'cpu').
            length (int, optional): Original (pre-transform) signal length. Required for 'stft' and
                'wavelet', whose inverse layers must reconstruct exactly that length.
        Returns:
            nn.Module: Transformed model.
        """
        if domain == "fourier":
            return nn.Sequential(IFTLayer(), nn.Identity(), model).to(device)
        elif domain == "stft":
            if length is None:
                raise ValueError("model_transformation requires `length` for domain='stft'.")
            inverse = ISTFTLayer(self.stft_n_fft, self.stft_hop_length, self.stft_win_length, length)
            return nn.Sequential(inverse, model).to(device)
        elif domain == "wavelet":
            if length is None:
                raise ValueError("model_transformation requires `length` for domain='wavelet'.")
            inverse = IDWTLayer(self.wavelet_name, self.wavelet_level, length)
            return nn.Sequential(inverse, model).to(device)
        elif domain == "input":
            return nn.Sequential(nn.Identity(), model).to(device)

    def input_transformation(self, inputs, domain="input", device='cpu'):
        """Transform the inputs to the desired domain.
        Args:
            inputs (torch.Tensor): Input data.
            domain (str): Domain to transform the inputs to ('input', 'fourier', 'stft', 'wavelet').
            device (str): Device to use ('cuda' or 'cpu').
        Returns:
            torch.Tensor: Transformed inputs.
        Note: the inputs are detached such that they represent new 'leaf' nodes in the computation graph.
        """
        if domain == "fourier":
            inputs=torch.fft.rfft(inputs).to(device)
        elif domain == "stft":
            stft = STFTLayer(self.stft_n_fft, self.stft_hop_length, self.stft_win_length)
            inputs = stft(inputs).to(device)
        elif domain == "wavelet":
            dwt = DWTLayer(self.wavelet_name, self.wavelet_level)
            inputs = dwt(inputs).to(device)
        elif domain == "input":
            inputs = inputs.to(device)
        return inputs
    
    def concept_correspondence(
        self, dataloader, num_batches=20, device="cpu"
    ):
        """ Global correspondece of concepts.
        Mean is taken only over instances where the concept was actually assigned
        (i.e. its mask has any non-zero entry). Instances with no LAD assignment
        to the concept produce an all-zero mask and are excluded from the average.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.
            num_batches (int, optional): Number of batches to process. Defaults to 20.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
        Returns:
            torch.Tensor: Concept gradients for each concept.
        """
        n_channels, in_len = infer_shape(dataloader) #this does NOT support images
        concept_gradients = torch.zeros(
            (self.n_concepts, n_channels, int(in_len / 2 + 1)), device=device)
        counts = torch.zeros(self.n_concepts, device=device)

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            if isinstance(batch, dict):
                inputs = batch["x"]
            else:
                inputs, _, _ = batch
            masks,_= self.eval(inputs, domain="fourier", device=device, maskMode="gradient", noisetunnel=False)
            for c in range(self.n_concepts):
                m = masks[c].to(device)                       # [B, C, F]
                assigned = m.abs().sum(dim=(1, 2)) > 0        # [B]
                if assigned.any():
                    concept_gradients[c] += m[assigned].sum(dim=0)
                    counts[c] += assigned.sum()

        concept_gradients = concept_gradients / counts.clamp(min=1).view(-1, 1, 1)
        concept_gradients = concept_gradients / (concept_gradients.sum(dim=[1,2], keepdim=True)  +1e-8)
        return concept_gradients
    
    @staticmethod
    def _importancesCW_to_dict(tensor):
        """Convert [n_classes, n_concepts, n_channels] tensor to nested dict for JSON."""
        n_classes, n_concepts, _ = tensor.shape
        return {
            str(cl): {str(c): tensor[cl, c].tolist() for c in range(n_concepts)}
            for cl in range(n_classes)
        }

    @staticmethod
    def _importancesCW_from_dict(d):
        """Convert nested dict from JSON to [n_classes, n_concepts, n_channels] tensor."""
        n_classes = len(d)
        n_concepts = len(d['0'])
        return torch.tensor([
            [d[str(cl)][str(c)] for c in range(n_concepts)]
            for cl in range(n_classes)
        ])


    def cluster_quality(self, dataloader, metric='silhouette', device='cpu'):

        for i, batch in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            if i > 0:
                break
            if isinstance(batch, dict):
                inputs = batch["x"]
            else:
                inputs, _, _ = batch
            inputs = inputs.to(device)
            test_activations, _ = getActivations(
                self.model.to(device), self.test_layers, inputs
            )
            reshaped_activations = reshape_activations(
                test_activations, self.act_shape, self.dimension
            )
            lads = self.get_lads(reshaped_activations)
            labels, _ = self.predict_labels(lads)

            lads = lads.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            if metric == 'silhouette':
                score = silhouette_score(lads, labels, metric='cosine')
            elif metric == 'calinski_harabasz':
                score = calinski_harabasz_score(lads, labels)
            elif metric == 'davies_bouldin':
                score = davies_bouldin_score(lads, labels)
        return score


def exp_scaling(x):
    x = x*np.log(2)
    x = (np.exp(x)-1)
    return x

def log_scaling(x):
    return np.log1p(x) / np.log(2)

def sigmoid_scaling(x):
    y = 12*(x-0.5)
    return 1/(1 + np.exp(-y))

def quad_scaling(x):
    return x**2

def moving_average(x, window_size, asnp=False):
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    is_torch = isinstance(x, torch.Tensor)
    x = np.atleast_1d(np.asarray(x)) 
    orig_shape = x.shape

    x2 = x.reshape(-1, orig_shape[-1])                 # (B*C*..., L)
    w = (np.ones(window_size, dtype=x.dtype) / window_size)  # (1, K)
    y2 = np.zeros_like(x2)
    for i in range(x2.shape[0]):
        y2[i] = np.convolve(x2[i], w, mode="same")
    y2 = y2.reshape(orig_shape)
    if is_torch and not asnp:
        y2 = torch.from_numpy(y2)
    return y2


def power_sensitivity(grad_X: torch.Tensor, X: torch.Tensor, positive=False) -> torch.Tensor:
    # grad_X, X: complex tensors with same shape
    # returns real tensor: sensitivity w.r.t. |X|^2 (up to a constant factor)
    s = 2.0 * torch.real(torch.conj(X) * grad_X)
    if positive: s = F.relu(s)
    return s  

def add_white_noise(x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    if std <= 0:
        return x
    if torch.is_complex(x):
        # complex Gaussian noise with per-component std so total variance ~ std^2
        real = torch.randn_like(x.real) * (std)
        imag = torch.randn_like(x.imag) * (std)
        return x + torch.complex(real, imag)
    else:
        return x + torch.randn_like(x) * std
