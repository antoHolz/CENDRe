from itertools import product
import torch
from .misc import infer_shape
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from .maskgen import destandardise
from typing import Dict, List, Optional, Sequence, Union
from collections.abc import Mapping

def inputs_perclass(dataloader, n_classes, n_repeats=2, device='cpu'):
    input_shape=dataloader.dataset[0][0].shape
    test_inputs=torch.zeros((n_classes, *input_shape), device=device)
    test_truth=torch.zeros((n_classes), device=device)
    test_id=torch.zeros((n_classes), device=device)
    filled=torch.zeros((n_classes), dtype=torch.bool)
    combinations = torch.Tensor(list(range(n_classes))).repeat_interleave(n_repeats, dim=0)

    for (x,y, id) in dataloader:
        x, y = x.to(device), y.to(device)
        for xi, yi, idi in zip(x,y, id):
            for j,k in enumerate(combinations):
                if filled[j]!=True and yi == k:
                    test_inputs[j,:]=xi.detach()
                    test_truth[j]=yi.detach()
                    test_id[j]=idi.detach()
                    filled[j]=True

                if(filled==True).all():
                    return test_inputs, test_truth, test_id
    return test_inputs[filled], test_truth[filled], test_id[filled]


def sorted_inputs(model, dataloader, n_classes, n_repeats=2, device='cpu', only_correct=False):
    input_shape = infer_shape(dataloader)
    # only_correct: keep just the (y, y_hat) diagonal instead of the full grid
    combinations = [(c, c) for c in range(n_classes)] if only_correct \
                   else list(product(range(n_classes),repeat=2))
    n_samples = len(combinations)*n_repeats
    test_inputs=torch.zeros((n_samples, *input_shape), device=device)
    test_truth=torch.zeros((n_samples), device=device)
    test_id=torch.zeros((n_samples), device=device)
    filled=torch.zeros((n_samples), dtype=torch.bool)
    combinations = torch.Tensor(combinations, device=device).repeat_interleave(n_repeats, dim=0)
    combinations = order_combinations(combinations.type(torch.int64))
    model.to(device)
    model.eval()
    for batch in dataloader:
        if isinstance(batch, dict):
            x,y, idx = batch["x"], batch["y"], batch["id"]
        else:
            x,y, idx = batch
        x, y = x.to(device), y.to(device)
        pred = model(x)
        current_combi = torch.stack((y,torch.argmax(pred, dim=1)), dim=1)
        for i,l in enumerate(current_combi):
            for j,k in enumerate(combinations):
                if filled[j]!=True and (l.eq(k)).all():
                    test_inputs[j,:]=x[i,:].detach()
                    test_truth[j]=y[i].detach()
                    test_id[j]=idx[i].detach()
                    filled[j]=True
                    break
        if(filled==True).all():
            return test_inputs, test_truth, test_id[filled]

    return test_inputs[filled], test_truth[filled], test_id[filled]



def sorted_inputs_regression(
    dataloader,
    samples_per_partition=10,
    out_range=[-2, 2],
    partitions=5,
    int_out=0,
    destandardized=True,
    astensor=False,
    DS_cfg=None,                 # pass DS_cfg (mean/std/ycols)
    out_range_is_standardized=False,  #  if out_range is in std units (e.g. [-2,2])
):
    if not int_out: int_out = 0

    # stats for destandardization (mirrors your report()) 
    label_names = DS_cfg.ycols if (DS_cfg is not None and hasattr(DS_cfg, "ycols")) else None
    ds_std = DS_cfg.std if (DS_cfg is not None and hasattr(DS_cfg, "std")) else None
    ds_mean = DS_cfg.mean if (DS_cfg is not None and hasattr(DS_cfg, "mean")) else None

    def destandardise(val, channel, id2label):
        if ds_mean is None or ds_std is None:
            return val
        if isinstance(ds_mean, Mapping):
            idx = id2label[int(channel)]
            if not (hasattr(ds_mean, idx) and hasattr(ds_std, idx)):
                return val
        else:
            idx = int(channel)
        return val * ds_std[idx] + ds_mean[idx]

    bnd = torch.linspace(out_range[0], out_range[1], partitions + 1)[1:-1]

    samples_x = {str(k): [] for k in range(partitions)}
    samples_y = {str(k): [] for k in range(partitions)}
    samples_id = {str(k): [] for k in range(partitions)}
    done = torch.zeros(partitions)

    for i, batch in enumerate(dataloader):
        if (done == 1).all():
            break
        if isinstance(batch, dict):
            inputs, outputs, idx = batch["x"], batch["y"], batch["id"]
        else:
            inputs, outputs, idx = batch

        if outputs.ndim <= 1:
            y_scalar = outputs
        else:
            y_scalar = outputs[:, int_out]

        y_scalar_for_bins = destandardise(y_scalar, int_out, label_names) if destandardized else y_scalar
        bin_idx = torch.bucketize(y_scalar_for_bins, bnd)

        for j in range(partitions):
            if done[j]: continue

            key = str(j)
            need = samples_per_partition - len(samples_x[key])

            # Indices in this batch that belong to bin j
            sel = (bin_idx == j).nonzero(as_tuple=False).flatten()
            if sel.numel() == 0: continue
            take = sel[:need].tolist()

            # Keep original objects/slices (simple and efficient)
            samples_x[key].extend([inputs[t] for t in take])
            samples_y[key].extend([outputs[t] for t in take])
            samples_id[key].extend([idx[t] for t in take])

            if len(samples_x[key]) >= samples_per_partition:
                done[j] = True

    # Stack the lists 
    for sampledict in [samples_x, samples_y, samples_id]:
        for k in range(partitions):
            key = str(k)
            if len(sampledict[key]) > 0 and torch.is_tensor(sampledict[key][0]):
                sampledict[key] = torch.stack(sampledict[key], dim=0)

    if astensor:
        samples_x = torch.cat([samples_x[str(key)] for key in range(partitions) if len(samples_x[str(key)]) > 0])
        samples_y = torch.cat([samples_y[str(key)] for key in range(partitions) if len(samples_y[str(key)]) > 0])
        samples_id = torch.cat([samples_id[str(key)] for key in range(partitions) if len(samples_id[str(key)]) > 0])

    return samples_x, samples_y, samples_id



def order_combinations(combinations):
    # Convert to a list of lists for easier manipulation
    combinations = combinations.tolist()
    
    # Create a dictionary to hold lists of combinations grouped by the first element
    grouped_combinations = {}
    for comb in combinations:
        if comb[0] not in grouped_combinations:
            grouped_combinations[comb[0]] = {'same': [], 'diff': []}
        if comb[0] == comb[1]:
            grouped_combinations[comb[0]]['same'].append(comb)
        else:
            grouped_combinations[comb[0]]['diff'].append(comb)
    
    # Sort each group
    for key in grouped_combinations:
        grouped_combinations[key]['same'].sort()
        grouped_combinations[key]['diff'].sort(key=lambda x: x[1])
    
    # Concatenate the sorted groups in order of the first element
    sorted_combinations = []
    for key in sorted(grouped_combinations.keys()):
        sorted_combinations.extend(grouped_combinations[key]['same'])
        sorted_combinations.extend(grouped_combinations[key]['diff'])
    
    # Convert back to a tensor
    return torch.tensor(sorted_combinations)





# General plotting

def plot_concepts_report_from_DS(
    inputs: torch.Tensor,                     # (N, Ch, T)  denormalized
    outputs: torch.Tensor,                    # (N, Ch_out)
    targets: torch.Tensor,                    # (N, Ch_out)
    masks: Dict[Union[int, str], torch.Tensor],       # {k: (N, T) or (N, 1, T)} binary
    importances: Dict[Union[int, str], Sequence[float]],  # {k: list/array length Ch}
    cfg_DG: Optional[object] = None,          # expects attrs: xcols, ycols (optional)
    importance_thr: float = 0.5,
    out_index: int = 0,
    task: str = "regression",
    save_path: str = r"D:/Datasets/DS_TS/WZL/SPP/ConceptMasks",
    font_base: int = 5,
    n_repeats: Optional[int] = None,          # title repeats every n columns (default: every column)
):
    # ------------------------------
    # Validate & unpack shapes
    # ------------------------------
    assert inputs.ndim == 3, f"inputs must be (N, Ch, T), got {tuple(inputs.shape)}"
    N, Ch, T = inputs.shape
    assert outputs.ndim == 2 and targets.ndim == 2, "outputs/targets must be (N, Ch_out)"
    assert outputs.shape == targets.shape, "outputs and targets must have same shape"
    Ch_out = outputs.shape[1]
    if not (0 <= out_index < Ch_out):
        raise IndexError(f"out_index {out_index} out of range for outputs with Ch_out={Ch_out}")

    # Names (optional)
    channel_names = getattr(cfg_DG, "xcols", None) if cfg_DG is not None else None
    label_names = getattr(cfg_DG, "ycols", None) if cfg_DG is not None else None

    # Normalize concept keys to int and take intersection of masks/importances
    def _as_int(k):
        try:
            return int(k)
        except Exception:
            raise ValueError(f"Concept key {k!r} must be convertible to int")

    imp_keys = {_as_int(k) for k in importances.keys()}
    mask_keys = {_as_int(k) for k in masks.keys()}
    concept_ids = sorted(imp_keys & mask_keys)
    if not concept_ids:
        raise ValueError("No overlapping concept IDs between masks and importances.")

    # ------------------------------
    # Channel selection per concept
    # ------------------------------
    keep_channels: Dict[int, List[int]] = {}
    for k in concept_ids:
        imp_vec = np.asarray(importances[str(k)], dtype=float)
        if imp_vec.shape[0] != Ch:
            raise ValueError(f"importances[{k}] length {imp_vec.shape[0]} != number of channels {Ch}")
        imp_use = np.abs(imp_vec) if task == "regression" else imp_vec
        sel = np.where(imp_use > importance_thr)[0].tolist()
        if not sel:
            sel = [int(np.argmax(imp_use))]  # always keep one
        keep_channels[k] = sel

    n_rows = sum(len(v) for v in keep_channels.values())
    if n_rows == 0:
        raise RuntimeError("No channels selected for plotting.")

    # ------------------------------
    # Build canvas
    # ------------------------------
    fig, axs = plt.subplots(n_rows, N, figsize=(4 * N, 4 * n_rows), squeeze=False)
    plt.subplots_adjust(wspace=0.10, hspace=None)

    # Colors for masks (blend with white, like your original)
    c = 0.4
    colours = (1.0 - c) * plt.get_cmap("viridis")(np.linspace(0, 1, len(concept_ids))) \
              + c * np.ones((len(concept_ids), 4))

    # Line width for mask stripes (scaled by axes width)
    fig.canvas.draw()
    lw = axs[0, 0].get_window_extent().width * 0.75 / T * 1.01

    # Fonts
    font_base = font_base or 10
    font_title = font_base * 3.8
    font_label = font_base * 3.4
    font_ticks = font_base * 3.0

    # "IS" label in top-left, as in your report
    axs[0, 0].annotate("IS", xy=(-0.5, 1.01), xycoords="axes fraction",
                       ha="left", va="bottom", fontsize=font_title, fontweight="bold")

    # ------------------------------
    # Helpers
    # ------------------------------
    def _to_np(t: torch.Tensor):
        return t.detach().cpu().numpy()

    def _mask_positions(mask_k: torch.Tensor, i: int):
        """Return 1D numpy array of time indices where mask is on for sample i."""
        if not isinstance(mask_k, torch.Tensor):
            mask_k = torch.as_tensor(mask_k)
        mk = mask_k[i]
        if mk.ndim == 1:             # (T,)
            pos = (mk > 0).nonzero(as_tuple=False).flatten()
        elif mk.ndim == 2:           # (1, T) or (C, T) -> take time dim
            pos = (mk > 0).nonzero(as_tuple=False)
            if pos.numel() == 0:
                return np.empty((0,), dtype=int)
            pos = pos[:, -1]
        else:
            raise ValueError(f"Unsupported mask dims for masks[k][i]: {tuple(mk.shape)}")
        return pos.detach().cpu().numpy()

    # ------------------------------
    # Plot rows
    # ------------------------------
    current_row = 0
    every = n_repeats or 1  # how often to show titles across columns

    for j, k in enumerate(concept_ids):       # concept loop
        imp_vec = np.asarray(importances[str(k)], dtype=float)
        mask_k = masks[k]                     # (N, T) or (N, 1, T)

        for ch in keep_channels[k]:           # channel loop
            # y-range per channel (10% headroom)
            ch_data = inputs[:, ch, :]        # (N, T)
            y_min = ch_data.min().item()
            y_max = ch_data.max().item()
            pad_min = abs(y_min) * 0.1
            pad_max = abs(y_max) * 0.1
            y_range = (y_min - pad_min, y_max + pad_max)

            for i in range(N):                # sample loop
                ax = axs[current_row, i]

                # mask stripes
                on_positions = _mask_positions(mask_k, i)
                if on_positions.size > 0:
                    ax.vlines(on_positions, *y_range, colors=colours[j], linewidth=lw)

                # signal
                ax.plot(range(T), _to_np(inputs[i, ch, :]), color="black")

                # cosmetics
                ax.set_ylim(y_range)
                if i:  # hide y-ticks except first column
                    ax.set_yticks([])
                if current_row != n_rows - 1:
                    ax.set_xticks([])
                ax.tick_params(labelsize=font_ticks)

                # row label = importance + channel name
                if i == 0:
                    imp_val = imp_vec[ch]
                    ch_name = channel_names[ch] if channel_names else f"ch{ch}"
                    ax.set_ylabel(f"{imp_val:.2f}\n({ch_name})",
                                  rotation=0, labelpad=60, fontsize=font_label,
                                  ha='right', va='center')
                    ax.yaxis.set_label_position('left')
                    ax.get_yaxis().set_label_coords(-0.4, 0.5)

                # first row: prediction / truth
                if current_row == 0 and (i % every == 0):
                    if task == "multilabel":
                        truth = int(targets[i, out_index].item())
                        pred  = int((outputs[i, out_index] > 0.5).item())
                        title = rf"$y$: {truth} $\quad\hat y$: {pred}"
                    elif task == "binary":
                        truth = int(targets[i, out_index].item() > 0.5)
                        pred  = int(outputs[i, out_index].item() > 0.5)
                        title = rf"$y$: {truth} $\quad\hat y$: {pred}"
                    elif task == "multiclass":
                        truth = int(targets[i, 0].item())
                        pred  = int(torch.argmax(outputs[i]).item())
                        title = rf"$y$: {truth:.0f} $\quad\hat y$: {pred:.0f}"
                    else:  # regression (default)
                        truth = targets[i, out_index].item()
                        pred  = outputs[i, out_index].item()
                        label = (label_names[out_index] if label_names and 0 <= out_index < len(label_names)
                                 else f"y[{out_index}]")
                        title = rf"$y$: {truth:.2e} $\quad\hat y$: {pred:.2e}"
                    axs[0, i].set_title(title, fontsize=font_title, pad=20)

            current_row += 1

    # ------------------------------
    # Save
    # ------------------------------
    os.makedirs(save_path, exist_ok=True)
    out_png = os.path.join(save_path, "concepts.png")
    out_pdf = os.path.join(save_path, "concepts.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"report saved at: {out_png}")

    return out_png, out_pdf


# --- thin adapter from a HF Dataset (or similar) ---
def plot_concepts_report_from_HF(
    ds,                                   # Hugging Face Dataset with columns: x, y, y_hat, masks
    importances: Dict[Union[int, str], Sequence[float]],
    cfg_DG: Optional[object] = None,
    importance_thr: float = 0.5,
    out_index: int = 0,
    task: str = "regression",
    save_path: str = r"D:/Datasets/DS_TS/WZL/SPP/ConceptMasks",
    font_base: int = 5,
    n_repeats: Optional[int] = None,
    rows: Optional[Union[int, Sequence[int]]] = None,  # e.g. 16 or [0,5,7,...]
):
    """
    Read tensors from a Hugging Face Dataset and call plot_concepts_report_from_DS.

    Expected column formats per row:
      - x: (Ch, T) array
      - y: (Ch_out,) array
      - y_hat: (Ch_out,) array
      - masks: either
          (A) dict {str(k): list[bool] length T}, or
          (B) 2D array (K, T) of bools (if your dataset stored fixed K)
    """
    # choose which rows
    if rows is None:
        idxs = list(range(len(ds)))
    elif isinstance(rows, int):
        idxs = list(range(min(rows, len(ds))))
    else:
        idxs = list(rows)

    # pull rows
    batch = ds.select(idxs) if hasattr(ds, "select") else [ds[i] for i in idxs]

    # convert to tensors
    x_list, y_list, yhat_list = [], [], []
    masks_rows = []  # keep raw per-row masks to assemble below

    # If it's an HF Dataset, iterate like dicts:
    iterable = batch if isinstance(batch, list) else [batch[i] for i in range(len(batch))]

    for row in iterable:
        x_list.append(torch.as_tensor(row["x"]))        # (Ch, T)
        y_list.append(torch.as_tensor(row["y"]))        # (Ch_out,)
        yhat_list.append(torch.as_tensor(row.get("y_hat", row.get("yhat", row.get("yhat_pred", None)))))
        masks_rows.append(row["masks"])

    if any(v is None for v in yhat_list):
        raise KeyError("Missing 'y_hat' column in dataset rows.")

    # stack
    inputs  = torch.stack(x_list, dim=0)        # (N, Ch, T)
    targets = torch.stack(y_list, dim=0)        # (N, Ch_out)
    outputs = torch.stack(yhat_list, dim=0)     # (N, Ch_out)

    N, Ch, T = inputs.shape

    # assemble masks dict {k: (N, T)}
    masks_dict: Dict[int, torch.Tensor] = {}

    # detect format A (dict per row) vs B (Array2D KxT)
    if isinstance(masks_rows[0], dict):
        # union of concept keys across selected rows
        all_keys = sorted({int(k) for row in masks_rows for k in row.keys()})
        for k in all_keys:
            M = torch.zeros((N, T), dtype=torch.bool)
            for i, row in enumerate(masks_rows):
                if str(k) in row:
                    v = torch.as_tensor(row[str(k)])
                    v = v.flatten()[:T]
                    M[i, :v.shape[0]] = (v != 0)
            masks_dict[k] = M
    else:
        # format B: each row has a (K, T) boolean array
        # assume consistent K across rows
        K = torch.as_tensor(masks_rows[0]).shape[0]
        for k in range(K):
            M = torch.zeros((N, T), dtype=torch.bool)
            for i, row in enumerate(masks_rows):
                v = torch.as_tensor(row)[k]
                v = v.flatten()[:T]
                M[i, :v.shape[0]] = (v != 0)
            masks_dict[k] = M

    # finally, call your original plotting core
    return plot_concepts_report_from_DS(
        inputs=inputs,
        outputs=outputs,
        targets=targets,
        masks=masks_dict,
        importances=importances,
        cfg_DG=cfg_DG,
        importance_thr=importance_thr,
        out_index=out_index,
        task=task,
        save_path=save_path,
        font_base=font_base,
        n_repeats=n_repeats,
    )


def plot_primitives(
    primitives,          # {name: (N, C, L) tensor} row-aligned with test_inputs
    test_inputs,         # (N, C, T) standardized model inputs
    save_path,
    fs=1,
    test_truth=None,     # (N,) class labels for column titles
    DS_cfg=None,
    destandardize=False,
    font_base=5,
    show=False,
):
    """Plot ground-truth primitive masks as full-height alpha bands over each sample signal."""
    # 1) validate masks and detect the plotting domain from their length
    N, C, T = test_inputs.shape
    if not primitives:
        raise ValueError("primitives is empty")
    names = list(primitives)
    P = len(names)
    shapes = {tuple(primitives[n].shape) for n in names}
    if len(shapes) != 1:
        raise ValueError(f"primitives must share one shape, got {shapes}")
    n_m, c_m, L = shapes.pop()
    if n_m != N or c_m != C:
        raise ValueError(f"primitives shape {(n_m, c_m, L)} does not match inputs {(N, C, T)}")
    if L == T:
        domain = 'input'
    elif L == T // 2 + 1:
        domain = 'fourier'
    else:
        raise ValueError(
            f"mask length {L} matches neither T={T} nor T//2+1={T // 2 + 1}; "
            "resize the masks first if a transform changed the input length"
        )

    # 2) domain processing, mirroring CENDRE.report
    if domain == 'fourier':
        signal = (torch.abs(torch.fft.rfft(test_inputs)) ** 2).detach().cpu().numpy()
        x_scale = fs * np.arange(0, L) / (2 * L)
        destandardize = False  # time-domain mean/std does not apply to a power spectrum
    else:
        signal = test_inputs.detach().cpu().numpy()
        x_scale = np.arange(0, T) / fs
    dx = x_scale[1] - x_scale[0]

    # 3) figure, colors (desaturated plasma, cf. the viridis concept colors), fonts
    fig, axs = plt.subplots(C, N, figsize=(4 * N, 3 * C), squeeze=False,
                            gridspec_kw={'hspace': 0.05})
    plt.subplots_adjust(wspace=0.05)

    cmap = plt.get_cmap("plasma")(np.linspace(0.0, 0.75, P))
    colors = 0.6 * cmap + 0.4 * np.ones((P, 4))

    font_base = font_base or 10
    font_title = font_base * 3.8
    font_label = font_base * 3.4
    font_ticks = font_base * 3.0

    usetex = plt.rcParams.get('text.usetex', False)
    labels = [n.replace('_', r'\_') if usetex else n for n in names]

    do_destd = destandardize and DS_cfg is not None and getattr(DS_cfg, 'mean', None) is not None
    c_names = getattr(DS_cfg, 'xcols', None) if DS_cfg is not None else None

    for ch in range(C):
        # 4) per-channel y-range (10% headroom)
        chan = signal[:, ch, :]
        y_min = chan.min() - abs(chan.min()) * 0.1
        y_max = chan.max() + abs(chan.max()) * 0.1
        if do_destd:
            y_range = (destandardise(DS_cfg, y_min, ch), destandardise(DS_cfg, y_max, ch))
        else:
            y_range = (y_min, y_max)

        for i in range(N):
            ax = axs[ch, i]

            # 5) bands: full-height span wherever primitive p's mask > 0
            for p, name in enumerate(names):
                m = primitives[name][i, ch].detach().cpu().numpy()
                edges = np.diff(np.concatenate(([0.0], (m > 0).astype(np.float64), [0.0])))
                starts = np.flatnonzero(edges > 0)
                ends = np.flatnonzero(edges < 0)  # exclusive
                for s, e in zip(starts, ends):
                    ax.axvspan(x_scale[s], x_scale[e - 1] + dx,
                               color=colors[p], alpha=0.5, linewidth=0)

            # 6) signal on top of the bands, plus cosmetics
            sig = destandardise(DS_cfg, chan[i], ch) if do_destd else chan[i]
            ax.plot(x_scale, sig, color="black", linewidth=1)
            ax.set_ylim(y_range)
            ax.tick_params(axis='both', labelsize=font_ticks)
            if i > 0:
                ax.set_yticks([])
            if ch < C - 1:
                ax.tick_params(labelbottom=False)

            if ch == 0 and test_truth is not None:
                ax.set_title(rf"$y$: {test_truth[i].item():.0f}", fontsize=font_title, pad=20)
            if i == 0 and C > 1:
                ch_name = c_names[int(ch)] if c_names else f"ch{ch}"
                if usetex:
                    ch_name = ch_name.replace('_', r'\_')
                ax.set_ylabel(ch_name, fontsize=font_label, labelpad=font_label * 0.3)
                ax.get_yaxis().set_label_coords(-0.35, 0.5)

    # 7) legend above the grid
    handles = [Patch(facecolor=colors[p], alpha=0.5, label=labels[p]) for p in range(P)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=min(P, 4), fontsize=font_label, frameon=False)

    # 8) single centered x-axis label, fixed 28pt below the bottom-most subplot
    if fs and fs != 1:
        xlab = "Frequency [Hz]" if domain == 'fourier' else "Time [s]"
        offset_fig = 28 / 72 / fig.get_figheight()
        y_bottom = axs[-1, 0].get_position().y0
        fig.text(0.5, y_bottom - offset_fig, xlab,
                 ha='center', va='top', fontsize=font_label * 0.8)

    # 9) save next to the concept grids
    os.makedirs(save_path, exist_ok=True)
    out_png = os.path.join(save_path, "primitives.png")
    out_pdf = os.path.join(save_path, "primitives.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    if show:
        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            pass
    plt.close(fig)
    print(f"primitives plot saved at: {out_png}")

    return out_png, out_pdf