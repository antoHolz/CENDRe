import torch
import torch.nn as nn
import torch.nn.functional as F
from ptwt import MatrixWavedec, MatrixWaverec


class g_nn_v2(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(self, task="classification", mode='classwise'):
        super().__init__()
        self.task = task
        self.bla = nn.Parameter(torch.empty(0)) #to make sure that gradients do not ignore this layer
        self.mode = mode

    def forward(self, y):
        """Wrapper for y in Eq. 2."""
        device = next(self.parameters()).device
        if (
            self.task == "regression" or self.task == "multilabel classification"
        ):  # Single output
            return y, None
        else:
            # y = torch.softmax(y, dim=1) # Softmax to get probabilities N x K
            out_id = torch.argmax(y, dim=1)
            n_classes = y.shape[1]
            y = y.unsqueeze(-1).to(device)
            y_T = torch.transpose(y, 1, 2)
            a = torch.bmm(y, torch.ones(y_T.shape, device=device)) #Repeated on 2-dim
            b = torch.bmm(torch.ones(y.shape, device=device), torch.transpose(y, 1, 2)) # N x K x K, repeated on 3rd dim
            dist = (a-b)
            if self.mode == 'classwise':
                dist = torch.sum(dist, dim=2)/(n_classes-1) # N x K
            elif self.mode == 'full':
                dist = torch.norm(dist, p=2, dim=(1, 2))
        return dist, out_id


class g_nn(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(self, out_nr=0, task="classification"):
        super().__init__()
        self.out_nr = out_nr
        self.task = task
        self.bla = nn.Parameter(torch.empty(0))

    def forward(self, y):
        """Wrapper for y in Eq. 2."""
        device = next(self.parameters()).device
        if (
            self.task == "regression" or self.task == "multilabel classification"
        ):  # Single output
            if y.ndim==1: return y[:]
            return y[:,self.out_nr] # y has shape B x Labels 
        else:
            y = y.unsqueeze(-1).to(device)
            y_T = torch.transpose(y, 1, 2)
            a = torch.bmm(y, torch.ones(y_T.shape, device=device))
            b = torch.bmm(torch.ones(y.shape, device=device), torch.transpose(y, 1, 2))
        return torch.norm(a - b, p=2, dim=(1, 2))


class FTLayer(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.fft.rfft(x)  # w times x + b


class IFTLayer(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.fft.irfft(x)  # w times x + b


class STFTLayer(nn.Module):
    """Forward STFT virtual layer: real (B, C, T) -> complex (B, C, F, Frames)."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length, periodic=True))

    def forward(self, x):
        B, C, T = x.shape  # torch.stft only accepts 1D/2D input
        X = torch.stft(
            x.reshape(B * C, T),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )  # (B*C, F, Frames)
        return X.reshape(B, C, *X.shape[-2:])


class ISTFTLayer(nn.Module):
    """Inverse STFT virtual layer: complex (B, C, F, Frames) -> real (B, C, length).

    `length` must be the original (pre-STFT) signal length: `torch.istft` needs it to trim the
    `center=True` padding exactly, the same role `torch.istft`'s `length=` plays for `torch.stft`.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.length = length
        self.register_buffer("window", torch.hann_window(win_length, periodic=True))

    def forward(self, X):
        B, C, F, Frames = X.shape
        x = torch.istft(
            X.reshape(B * C, F, Frames),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=self.length,
        )  # (B*C, length)
        return x.reshape(B, C, self.length)


def dwt_band_lengths(length: int, wavelet: str, level: int) -> tuple:
    """Per-band coefficient lengths `ptwt.MatrixWavedec` produces for a given signal length.

    Deterministic given (length, wavelet, level) -- data-independent -- so it can be computed once
    from a dummy zero signal and reused to split a flat coefficient vector back into bands.
    """
    dummy = torch.zeros(1, 1, length)
    coeffs = MatrixWavedec(wavelet=wavelet, level=level)(dummy)
    return tuple(c.shape[-1] for c in coeffs)


class DWTLayer(nn.Module):
    """Forward DWT virtual layer: real (B, C, T) -> flat real coefficient vector (B, C, sum(bands)).

    Wraps `ptwt.MatrixWavedec` (orthogonalized boundary-wavelet transform, exactly reversible),
    concatenating the per-level [cA_L, cD_L, ..., cD_1] coefficient list into one flat tensor so the
    rest of CENDRe's single-leaf-tensor machinery needs no changes.
    """

    def __init__(self, wavelet: str, level: int):
        super().__init__()
        self.dec = MatrixWavedec(wavelet=wavelet, level=level)

    def forward(self, x):
        coeffs = self.dec(x)  # list of (B, C, band_len_i) real tensors
        return torch.cat(coeffs, dim=-1)


class IDWTLayer(nn.Module):
    """Inverse DWT virtual layer: flat real coefficient vector -> real (B, C, length).

    `length` must be the original (pre-DWT) signal length: `MatrixWavedec` pads odd lengths
    internally, so the reconstruction is trimmed back to `length` explicitly.
    """

    def __init__(self, wavelet: str, level: int, length: int):
        super().__init__()
        self.band_lengths = dwt_band_lengths(length, wavelet, level)
        self.length = length
        self.rec = MatrixWaverec(wavelet=wavelet)

    def forward(self, x_flat):
        coeffs = list(torch.split(x_flat, self.band_lengths, dim=-1))
        x = self.rec(coeffs)
        return x[..., : self.length]


def dwt_bands_to_image(flat: torch.Tensor, band_lengths: tuple) -> torch.Tensor:
    """Unflatten a DWT coefficient vector into a time-aligned image for display.

    `flat` is `[cA_L, cD_L, cD_{L-1}, ..., cD_1]` concatenated (`DWTLayer`'s output order: coarsest
    band first). Each band is nearest-neighbor upsampled to the finest band's length so all bands
    align on a common time axis, then stacked into `(..., n_bands, finest_length)` with row 0 =
    coarsest (lowest-frequency) band -- the same low-to-high ordering `torch.stft`'s frequency axis
    uses with `origin='lower'`. Interpolation (not exact repetition) since band lengths only halve
    exactly for even signal lengths (`MatrixWavedec` pads odd lengths internally).
    """
    lead_shape = flat.shape[:-1]
    target_length = band_lengths[-1]
    rows = []
    for band in torch.split(flat, band_lengths, dim=-1):
        band = band.reshape(-1, 1, band.shape[-1])
        band = F.interpolate(band, size=target_length, mode="nearest")
        rows.append(band.reshape(*lead_shape, target_length))
    return torch.stack(rows, dim=-2)  # (..., n_bands, target_length)
