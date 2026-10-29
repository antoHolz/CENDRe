"""Generate example plots of synthetic datasets for the CENDRE paper appendix.

Usage:
    python runs/plot_synthetic_examples.py --data_root D:/Datasets/DS_TS/synthetic --output_dir D:/DSME/Code/Repos/CENDRE_icml2026/figures
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset


SHAPE_COLORS = {
    "square": "#1f77b4",   # blue
    "circle": "#d62728",   # red
    "triangle": "#2ca02c", # green
}
FREQ_COLORS = {
    "midFreq": "#1f77b4",
    "midhighFreq": "#2ca02c",
    "highFreq": "#d62728",
}


def find_sample_by_class(ds, class_id):
    """Return index of first sample with given class label."""
    for i in range(len(ds)):
        if int(ds[i]["y"]) == class_id:
            return i
    raise ValueError(f"No sample with class {class_id}")


def plot_synthl(ds, output_path):
    """Plot SynthL example: one sample per class with GT mask overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), sharey=True)

    for cls, ax in enumerate(axes):
        idx = find_sample_by_class(ds, cls)
        sample = ds[idx]
        x = np.array(sample["X"]).squeeze()
        t = np.arange(len(x))

        ax.plot(t, x, color="0.3", linewidth=0.6, zorder=2)

        # Overlay GT masks
        for name, color in SHAPE_COLORS.items():
            key = f"{name}_X"
            if key not in sample:
                continue
            mask = np.array(sample[key]).squeeze()
            if mask.max() > 0:
                # Find contiguous mask regions
                diff = np.diff(np.concatenate([[0], mask, [0]]))
                starts = np.where(diff > 0)[0]
                ends = np.where(diff < 0)[0]
                for s, e in zip(starts, ends):
                    ax.axvspan(s, e, alpha=0.25, color=color, label=name if s == starts[0] else None, zorder=1)

        ax.set_title(f"Class {cls}", fontsize=10)
        ax.set_xlabel("Time step")
        ax.tick_params(labelsize=8)
        if cls == 0:
            ax.set_ylabel("Amplitude")
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_synthf(ds, output_path, fs=400):
    """Plot SynthF example: time domain + FFT magnitude with GT frequency bands."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    for cls in range(2):
        idx = find_sample_by_class(ds, cls)
        sample = ds[idx]
        x = np.array(sample["X"]).squeeze()
        T = len(x)
        t = np.arange(T)

        # Time domain
        ax_t = axes[0, cls]
        ax_t.plot(t, x, color="0.3", linewidth=0.5)
        ax_t.set_title(f"Class {cls} — time domain", fontsize=10)
        ax_t.set_xlabel("Time step")
        ax_t.set_ylabel("Amplitude")
        ax_t.tick_params(labelsize=8)

        # Frequency domain
        ax_f = axes[1, cls]
        X_fft = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(T, d=1.0 / fs)
        ax_f.plot(freqs, X_fft, color="0.3", linewidth=0.5)
        ax_f.set_title(f"Class {cls} — frequency domain", fontsize=10)
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel("|FFT|")
        ax_f.tick_params(labelsize=8)

        # Overlay GT frequency masks
        for name, color in FREQ_COLORS.items():
            key = f"{name}_X"
            if key not in sample:
                continue
            mask = np.array(sample[key]).squeeze()
            if mask.max() > 0:
                # Mask is in frequency domain (length T/2+1)
                mask_indices = np.where(mask > 0)[0]
                if len(mask_indices) > 0:
                    f_start = freqs[mask_indices[0]]
                    f_end = freqs[mask_indices[-1]]
                    ax_f.axvspan(f_start, f_end, alpha=0.25, color=color, label=name, zorder=1)

        ax_f.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset example plots")
    parser.add_argument("--data_root", type=str, default="D:/Datasets/DS_TS/synthetic",
                        help="Root directory containing synthetic datasets")
    parser.add_argument("--output_dir", type=str, default="D:/DSME/Code/Repos/CENDRE_icml2026/figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SynthL: square-circle dataset
    synthl_path = data_root / "square-circle_ch1_specificChannels"
    print(f"Loading SynthL from {synthl_path}")
    ds_synthl = Dataset.load_from_disk(str(synthl_path))
    plot_synthl(ds_synthl, output_dir / "synthl_example.pdf")

    # SynthF: midFreq-highFreq dataset
    synthf_path = data_root / "midFreq-highFreq_ch1_specificChannels"
    print(f"Loading SynthF from {synthf_path}")
    ds_synthf = Dataset.load_from_disk(str(synthf_path))
    plot_synthf(ds_synthf, output_dir / "synthf_example.pdf", fs=400)


if __name__ == "__main__":
    main()
