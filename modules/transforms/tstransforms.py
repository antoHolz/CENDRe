"""
This file implements various time-series transforms using PyTorch. 
**Inspired by multiple repositories, including tsai**.

All transforms in this file operate on PyTorch tensors. Whenever relevant, we 
assume the data is shaped like [B, C, L], where:
    B = Batch size
    C = Number of channels (features)
    L = Length of the time dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import stft
from scipy.interpolate import interp1d, CubicSpline

###############################################################################
# TSToTensor
###############################################################################
class TSToTensor(nn.Module):
    """Converts the input into a torch.Tensor (if it isn't one already).

    This transform ensures the input is a torch.Tensor, which is necessary 
    for subsequent PyTorch operations.

    Example:
        >>> transform = TSToTensor()
        >>> x = [1, 2, 3]
        >>> out = transform(x)  # out is now a torch.Tensor: tensor([1, 2, 3])
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward pass to convert input to a torch.Tensor.

        Args:
            x: Input data of any type convertible to torch.Tensor.

        Returns:
            torch.Tensor: The input cast (or copied) into a torch.Tensor.
        """
        return torch.tensor(x,dtype=torch.float32)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


###############################################################################
# TSGaussianNoise
###############################################################################
class TSGaussianNoise(nn.Module):
    """Adds Gaussian noise to the input tensor.

    This transform operates element-wise over the input. When applied, each 
    element will have i.i.d. Gaussian noise added to it, controlled by `std`.

    For time-series data of shape [B, C, L], the noise shape will also be [B, C, L].

    Example:
        >>> transform = TSGaussianNoise(std=0.05, p=1.0)
        >>> x = torch.zeros(2, 3, 10)  # shape [B=2, C=3, L=10]
        >>> out = transform(x)  # out now has Gaussian noise
    """

    def __init__(self, std: float = 0.1, p: float = 1.0):
        """
        Args:
            std (float): Standard deviation of the Gaussian noise.
            p (float): Probability of applying noise.
        """
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to apply Gaussian noise.

        Args:
            x (torch.Tensor): Input data of shape [..., L].

        Returns:
            torch.Tensor: Noisy version of `x` if applied, otherwise returns `x` unchanged.
        """
        if torch.rand(1).item() > self.p or self.std <= 0:
            return x
        noise = torch.randn_like(x) * self.std
        return x + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std}, p={self.p})"


###############################################################################
# TSRandomCropPad
###############################################################################
class TSRandomCropPad(nn.Module):
    """Randomly selects a window within the time dimension and zeros out the rest.

    Conceptually, this is equivalent to:
    1. Cropping the data to a random sub-window in the time dimension.
    2. Padding it back to the original length (with zeros in the same location).

    For an input of shape [B, C, L], the output is also [B, C, L], but only 
    the cropped window is retained while everything else is zeroed out.

    Example:
        >>> transform = TSRandomCropPad(magnitude=0.1, p=1.0)
        >>> x = torch.randn(2, 3, 10)  # shape [B=2, C=3, L=10]
        >>> out = transform(x)  # out retains a sub-window of length ~L*magnitude
    """

    def __init__(self, magnitude: float = 0.05, p: float = 1.0):
        """
        Args:
            magnitude (float): Beta distribution parameter (tsai uses Beta(mag, mag)).
            p (float): Probability of applying the transform.
        """
        super().__init__()
        self.magnitude = magnitude
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to randomly crop and pad the input.

        Args:
            x (torch.Tensor): Input data of shape [B, C, L] or similar.

        Returns:
            torch.Tensor: Tensor with the same shape as `x`, but zeros outside 
                the randomly selected sub-window.
        """
        if torch.rand(1).item() > self.p or self.magnitude <= 0:
            return x
        
        seq_len = x.shape[-1]
        if seq_len < 2:
            return x

        # Sample from a Beta(magnitude, magnitude) distribution:
        lambd = np.random.beta(self.magnitude, self.magnitude)
        # Keep only the upper half
        lambd = max(lambd, 1.0 - lambd)
        win_len = int(round(seq_len * lambd))

        if win_len >= seq_len:
            return x
        
        start = np.random.randint(0, seq_len - win_len)

        # Zero out everything except the selected window
        out = torch.zeros_like(x)
        out[..., start:start + win_len] = x[..., start:start + win_len]
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(magnitude={self.magnitude}, p={self.p})"


class TSResize(nn.Module):
    """Resizes (interpolates) the last dimension of the tensor to a new length.

    This merged transform can:
      - Use a fixed new_length, OR
      - Derive a new length from a magnitude factor .
    
    If `new_length` is provided, it overrides the `magnitude` logic.
    If `new_length` is None and `magnitude` != 0, it scales the current length by (1 + magnitude).
    If the final length ends up being the same as the original length, no resizing is done.
    
    Uses `torch.nn.functional.interpolate` for 1D interpolation.

    Args:
        new_length (int, optional): Fixed target length for the time dimension.
            If not None, it overrides the magnitude-based approach.
        magnitude (float): Scaling factor for the time dimension. 
            final_length = round((1 + magnitude) * current_length).
            Ignored if new_length is not None.
        mode (str): Interpolation mode, e.g. 'nearest', 'linear', 'area'.

    Example:
        >>> # 1) Using a fixed new_length
        >>> transform = TSResize(new_length=15, mode='linear')
        >>> x = torch.randn(2, 3, 10)  # shape [B=2, C=3, L=10]
        >>> out = transform(x)         # out has shape [2, 3, 15]
        
        >>> # 2) Using magnitude-based resizing
        >>> transform = TSResize(magnitude=-0.3, mode='nearest')
        >>> x = torch.randn(10)        # shape [L=10]
        >>> out = transform(x)         # new length is ~ 70% of original => ~7
    """

    def __init__(
        self, 
        new_length: int = None, 
        magnitude: float = 0.0, 
        mode: str = 'nearest', 
    ):
        super().__init__()
        self.new_length = new_length
        self.magnitude = magnitude
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to resize the last dimension.

        Args:
            x (torch.Tensor): Input data of shape [..., L].
        
        Returns:
            torch.Tensor: Resized tensor (in the last dimension) or `x` unchanged.
        """

        orig_shape = x.shape
        original_length = orig_shape[-1]

        # Decide the final length
        if self.new_length is not None:
            final_length = self.new_length
        else:
            # Use magnitude to compute new length
            final_length = int(round((1 + self.magnitude) * original_length))
        
        # Ensure at least length 1
        final_length = max(final_length, 1)

        # If final length hasn't changed, skip
        if final_length == original_length:
            return x

        # Reshape for F.interpolate => (N, C, L)
        if x.ndim == 1:
            # (L) -> (1, 1, L)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            # (C, L) -> (1, C, L)
            x = x.unsqueeze(0)
        # else assume (N, C, L) already

        # Interpolate
        out = F.interpolate(
            x, 
            size=final_length, 
            mode=self.mode, 
            align_corners=None if self.mode in ['nearest', 'area'] else False
        )

        # Reshape back to original rank
        if len(orig_shape) == 1:
            # was (L), now (1,1,final_length)
            out = out.squeeze(0).squeeze(0)
        elif len(orig_shape) == 2:
            # was (C, L), now (1,C,final_length)
            out = out.squeeze(0)

        return out

    def __repr__(self):
        return (f"{self.__class__.__name__}(new_length={self.new_length}, "
                f"magnitude={self.magnitude}, mode={self.mode})")


###############################################################################
# Helper function and TSTimeWarp
###############################################################################
def random_cum_curve_generator(seq_len: int, magnitude: float = 0.1, order: int = 4):
    """Generates a monotonically increasing curve (cumulative) used for time warping.

    We sample 3*(order-1)+1 random points from a normal distribution, then 
    build a spline and evaluate it at regular intervals from 0 to seq_len-1. 
    Finally, the curve is normalized to the range [0, seq_len-1].

    Args:
        seq_len (int): Length of the time dimension (L).
        magnitude (float): Controls how strongly to warp.
        order (int): Polynomial spline order (typically 3 or 4).

    Returns:
        np.ndarray: 1D array of shape [L], representing the warped time indices.
    """
    # 1) Sample random points for the spline
    random_points = np.random.normal(loc=1.0, scale=magnitude, size=3 * (order - 1) + 1)
    x_vals = np.linspace(-seq_len, 2 * seq_len - 1, len(random_points))

    # 2) Build the spline
    f = CubicSpline(x_vals, random_points, axis=-1)

    # 3) Evaluate it on [0, seq_len-1], then normalize to [0, seq_len-1]
    warped = f(np.arange(seq_len))
    warped -= warped[0]
    warped /= warped[-1]
    warped *= (seq_len - 1)
    return warped

class TSTimeWarp(nn.Module):
    """Applies time warping along the last dimension by sampling a smooth 
    cumulative curve and using a spline to re-index the original signal.

    This transform warps the time axis of the input. If the input is shaped 
    like [B, C, L], it will warp along L for each [B, C, :].

    Example:
        >>> transform = TSTimeWarp(magnitude=0.1, spline_order=4, p=1.0)
        >>> x = torch.randn(2, 3, 10)  # shape [B=2, C=3, L=10]
        >>> out = transform(x)  # out is time-warped
    """

    def __init__(self, magnitude: float = 0.1, spline_order: int = 4, p: float = 1.0):
        """
        Args:
            magnitude (float): How strongly to warp in time.
            spline_order (int): Polynomial order for the CubicSpline.
            p (float): Probability of applying the warp.
        """
        super().__init__()
        self.magnitude = magnitude
        self.spline_order = spline_order
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to apply time warping.

        Args:
            x (torch.Tensor): Input data of shape [..., L]. The last dimension
                will be warped.

        Returns:
            torch.Tensor: Time-warped version of `x`.
        """
        if torch.rand(1).item() > self.p or self.magnitude <= 0:
            return x

        seq_len = x.shape[-1]
        if seq_len < 2:
            return x

        # Convert to NumPy (CPU) for spline interpolation
        # Convert to numpy if needed
        was_torch = isinstance(x, torch.Tensor)
        if was_torch:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        # Build the base spline
        base_x = np.arange(seq_len)
        spline = CubicSpline(base_x, x, axis=-1)

        # Generate warp indices via random cumulative curve
        warp_indices = random_cum_curve_generator(
            seq_len, magnitude=self.magnitude, order=self.spline_order
        )

        # Interpolate
        warped_np = spline(warp_indices)

        # Convert back to torch
        if was_torch:
            warped_x = torch.tensor(warped_np, device=x.device, dtype=x.dtype)
        else:
            warped_x = warped_np
        return warped_x

    def __repr__(self):
        return (f"{self.__class__.__name__}(magnitude={self.magnitude}, "
                f"spline_order={self.spline_order}, p={self.p})")

###############################################################################
# TSSampleWindow
###############################################################################

class TSSampleWindow(nn.Module):
    """Samples a fixed-size window from the time dimension.

    This transform extracts a contiguous sub-window of specified length 
    from the input tensor along the last dimension. If the input is shaped 
    like [B, L, C], it will sample along L for each [B, :, C].

    Example:
        >>> transform = TSSampleWindow(window_size=5)
        >>> x = torch.randn(2, 10, 3)  # shape [B=2, L=10, C=3]
        >>> out = transform(x)  # out has shape [2, 5, 3]
    """

    def __init__(self, window_size: int):
        """
        Args:
            window_size (int): Length of the window to sample.
        """
        super().__init__()
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to sample a window.

        Args:
            x (torch.Tensor): Input data of shape [..., L]. The last dimension
                will be sampled.

        Returns:
            torch.Tensor: Sampled window of shape [..., window_size].
        """
        x_len = x.shape[0]
        if self.window_size >= x_len:
            return x

        start = np.random.randint(0, x_len - self.window_size + 1)
        end = start + self.window_size
        return x[start:end,:]

    def __repr__(self):
        return f"{self.__class__.__name__}(window_size={self.window_size})"




###############################################################################
# TSSplitWindow
###############################################################################

class TSSplitWindow(nn.Module):
    """Splits the time dimension into two windows of fixed size. FOR FORECASTING MODELS WITH DATA OF SHAPE B,L,C

    This transform divides the input tensor along the last dimension into
    two parts: the first `seq_len` elements and the last `label_len + pred_len` elements.
    Example:
        >>> transform = TSSplitWindow(seq_len=90, pred_len=48, label_len=48)
        >>> x = torch.randn(2, 144, 3)  # shape [B=2, L=144, C=3]
        >>> out1, out2 = transform(x)  # out1 has shape [2, 90, 3], out2 has shape [2, 96, 3]
    """

    def __init__(self, seq_len: int, pred_len: int, label_len: int = None):
        """
        Args:
            window_size (int): Length of each window.
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        # If label_len is not provided, default to seq_len - pred_len.
        self.label_len = seq_len - pred_len if label_len is None else label_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to split into windows.

        Args:
            x (torch.Tensor): Input data of shape [..., L]. The last dimension
                will be split.

        Returns:
            torch.Tensor: Tensor of shape [..., num_windows, window_size].
        """
        part1 = x[0:self.seq_len]
        # Part 2: elements starting from index (label_len + pred_len) to the end.
        part2 = x[-(self.label_len + self.pred_len):]
        
        return part1,part2

    def __repr__(self):
        return f"{self.__class__.__name__}(window_size={self.window_size})"


###############################################################################
# TSSwapAxes
###############################################################################

class TSSwapAxes:
    """
    Swap two axes in the specified fields of a sample dictionary.

    Parameters
    ----------
    fields : str or list of str
        The key or keys in the sample dict whose content will have axes swapped.
    axis1 : int
        The first axis to swap.
    axis2 : int
        The second axis to swap.
    """
    def __init__(self,  axis1: int, axis2: int):
        self.axis1 = axis1
        self.axis2 = axis2

    def __call__(self, x: dict) -> dict:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"X is not a torch.Tensor (got type {type(x)}).")
        x = torch.transpose(x, self.axis1, self.axis2)
        return x


###############################################################################
# TSToFrequency
###############################################################################
class TSToFrequency(nn.Module):
    """Transforms a 1D time-domain signal into its frequency representation 
    (magnitude of STFT).

    This uses `scipy.signal.stft` internally with a fixed sampling frequency 
    and window size (nperseg). The output is the magnitude of the STFT.

    Example:
        >>> transform = TSToFrequency(sampling_frequency=64_000, segment=32)
        >>> x = np.random.randn(128)  # e.g. a 1D NumPy array
        >>> out = transform(x)        # out is the magnitude of the STFT
    """

    def __init__(self, sampling_frequency: float = 64e3, segment: int = 32):
        """
        Args:
            sampling_frequency (float): Sampling frequency to pass to stft().
            segment (int): The nperseg (window size) for STFT.
        """
        super().__init__()
        self.sampling_frequency = sampling_frequency
        self.segment = segment

    def forward(self, x):
        """Forward pass to compute the magnitude of the STFT.

        Args:
            x (ndarray or torch.Tensor): 1D signal array. 
                If a torch.Tensor is passed, it will be converted to numpy.

        Returns:
            np.ndarray: Magnitude of the STFT of shape [freq, time].
        """

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # Compute STFT and return magnitude
        f, t, Zxx = stft(x, fs=self.sampling_frequency, nperseg=self.segment)
        return np.abs(Zxx)

    def __repr__(self):
        return (f"{self.__class__.__name__}(sampling_frequency={self.sampling_frequency}, "
                f"segment={self.segment})")


###############################################################################
# TSJitter
###############################################################################
class TSJitter(nn.Module):
    """Adds random jitter (Gaussian noise) to the input time-series.

    Example:
        >>> transform = TSJitter(sigma=0.03, p=1.0)
        >>> x = np.random.randn(10, 3)
        >>> out = transform(x)  # out is x + Gaussian noise
    """

    def __init__(self, sigma: float = 0.03, p: float = 1.0):
        """
        Args:
            sigma (float): Standard deviation of the noise to add.
            p (float): Probability of applying the transform.
        """
        super().__init__()
        self.sigma = sigma
        self.p = p

    def forward(self, x):
        """Forward pass to add jitter.

        Args:
            x (array-like or torch.Tensor): Input data.

        Returns:
            Same shape as x, with random noise added.
        """
        if torch.rand(1).item() > self.p or self.sigma == 0:
            return x

        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # Add noise
        jittered = x + np.random.normal(loc=0.0, scale=self.sigma, size=x.shape)

        # Convert back to torch if original was torch
        if torch.is_tensor(x):
            jittered = torch.tensor(jittered, dtype=torch.float32)
        return jittered

    def __repr__(self):
        return f"{self.__class__.__name__}(sigma={self.sigma}, p={self.p})"


###############################################################################
# TSMagnitudeWarp
###############################################################################
class TSMagnitudeWarp(nn.Module):
    """Randomly warps the magnitude of each dimension over time 
    using spline interpolation.

    NOTE: This version is adapted from an external source.

    Example:
        >>> transform = TSMagnitudeWarp(sigma=0.2, knot=4, p=1.0)
        >>> x = np.random.randn(100, 2)  # shape [L=100, C=2]
        >>> out = transform(x)
    """

    def __init__(self, sigma: float = 0.2, knot: int = 4, p: float = 1.0):
        """
        Args:
            sigma (float): Standard deviation for the magnitude warp.
            knot (int): Number of spline knots (knot + 2 control points).
            p (float): Probability of applying the transform.
        """
        super().__init__()
        self.sigma = sigma
        self.knot = knot
        self.p = p

    def forward(self, x):
        """Forward pass to apply the magnitude warp.

        Args:
            x (array-like or torch.Tensor): Shape [L, C].

        Returns:
            Warped signal of the same shape [L, C].
        """
        if torch.rand(1).item() > self.p or self.sigma == 0:
            return x

        # Convert to numpy if needed
        was_torch = isinstance(x, torch.Tensor)
        if was_torch:
            x = x.detach().cpu().numpy()

        L, C = x.shape
        orig_steps = np.arange(L)

        # random warps shape: (knot+2, C)
        random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot + 2, C))
        warp_steps = (np.ones((C, 1)) * np.linspace(0, L - 1, num=self.knot + 2)).T

        ret = np.zeros_like(x)
        for dim in range(C):
            # For each dimension, we create a custom magnitude warp
            warper = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[:, dim])
            time_warp = warper(orig_steps)
            scale = (L - 1) / time_warp[-1] if time_warp[-1] != 0 else 1.0
            ret[:, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, L - 1), x[:, dim])

        if was_torch:
            ret = torch.tensor(ret, dtype=torch.float32)
        return ret

    def __repr__(self):
        return (f"{self.__class__.__name__}(sigma={self.sigma}, knot={self.knot}, p={self.p})")


###############################################################################
# TSTimeWarp2
###############################################################################
class TSTimeWarp2(nn.Module):
    """Performs time warping by re-indexing the time axis with a warped curve.

    NOTE: This version is conceptually similar to TSTimeWarp in your current file, 
    but uses a simpler approach adapted from another source.

    Example:
        >>> transform = TSTimeWarp2(sigma=0.2, knot=4, p=1.0)
        >>> x = np.random.randn(3, 100)  # shape [C=3, L=100]
        >>> out = transform(x)           # time-warped output
    """

    def __init__(self, sigma: float = 0.2, knot: int = 4, p: float = 1.0):
        """
        Args:
            sigma (float): Standard deviation for the warp curve.
            knot (int): Number of knots for the spline.
            p (float): Probability of applying the transform.
        """
        super().__init__()
        self.sigma = sigma
        self.knot = knot
        self.p = p

    def forward(self, x):
        """Forward pass to apply time warping.

        Args:
            x (array-like or torch.Tensor): Shape [C, L].

        Returns:
            Warped signal of shape [C, L].
        """
        if torch.rand(1).item() > self.p or self.sigma == 0:
            return x

        was_torch = isinstance(x, torch.Tensor)
        if was_torch:
            x = x.detach().cpu().numpy()

        C, L = x.shape
        orig_steps = np.arange(L)

        # random warps shape: (knot+2, C)
        random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot + 2, C))
        warp_steps = (np.ones((C, 1)) * np.linspace(0, L - 1, num=self.knot + 2)).T

        ret = np.zeros_like(x)
        for dim in range(C):
            warper = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[:, dim])
            time_warp = warper(orig_steps)
            scale = (L - 1) / time_warp[-1] if time_warp[-1] != 0 else 1.0
            ret[dim, :] = np.interp(orig_steps, np.clip(scale * time_warp, 0, L - 1), x[dim, :])

        if was_torch:
            ret = torch.tensor(ret, dtype=torch.float32)
        return ret

    def __repr__(self):
        return (f"{self.__class__.__name__}(sigma={self.sigma}, knot={self.knot}, p={self.p})")



    
###############################################################################
# TSStandardize
###############################################################################
class TSStandardize(nn.Module):
    """Standardizes the input tensor by subtracting the mean and dividing by the standard deviation.
    
    Example:
        >>> transform = TSStandardize()
        >>> np.random.randn(3, 100)
        >>> out = transform(x)
    """
    
    def __init__(self, mean: list = None, std: list = None, eps: float = 1e-8):
        super().__init__()
        self.mean = np.array(mean)[:, None]
        self.std = np.array(std)[:, None]
        self.eps = eps
    
    def forward(self, x):
        was_torch = isinstance(x, torch.Tensor)
        if was_torch:
            x = x.detach().cpu().numpy()

        mean = self.mean if self.mean is not None else x.mean(dim=(0, 2), keepdim=True)
        std = self.std if self.std is not None else x.std(dim=(0, 2), keepdim=True)

        ret =  (x - mean) / (std + self.eps)

        if was_torch:
            ret = torch.tensor(ret, dtype=torch.float32)
            
        return ret
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, eps={self.eps})"
    
class TSIdentity(nn.Module):
    """
    Convert the content of specified fields in a sample to torch.Tensors.
    
    The sample can be:
      - A dict: If fields is None, attempt to convert every convertible field.
                Otherwise, only convert the specified fields.
      - A list or tuple: Convert each convertible item to a torch.Tensor.
      - A single np.ndarray: Convert it to a torch.Tensor.
      - A single torch.Tensor: Returned as is.
    
    """
    def __init__(self):
        super().__init__()

    def forward(self, sample):
        return sample

