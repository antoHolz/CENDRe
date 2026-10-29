import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional

###############################################################################
# Reversible Instance Normalization
###############################################################################
class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) layer.

    This module applies a normalization step (based on mean or the last timestep)
    and an optional learned affine transformation to input data. It also provides
    the ability to reverse (de-normalize) the transformation using stored statistics.

    Args:
        num_features (int): 
            Number of features (channels) in the input. For a 3D input of the form 
            [batch_size, seq_len, num_features], this corresponds to `num_features`.
        eps (float, optional): 
            A small constant added to the denominator for numerical stability. 
            Defaults to 1e-5.
        affine (bool, optional): 
            If True, RevIN includes learnable affine parameters (weight and bias) 
            applied after normalization. Defaults to True.
        subtract_last (bool, optional): 
            If True, subtract the values of the last timestep (per batch and feature) 
            instead of the mean. Defaults to False.

    Attributes:
        num_features (int): 
            The number of input features.
        eps (float): 
            A small value added to avoid division by zero in normalization.
        affine (bool): 
            Indicates whether learnable affine parameters are used.
        subtract_last (bool): 
            If True, the last timestep values are used in place of the mean 
            for normalization/de-normalization.
        affine_weight (torch.nn.Parameter): 
            Learnable scale parameter (initialized as ones) if `affine=True`.
        affine_bias (torch.nn.Parameter): 
            Learnable bias parameter (initialized as zeros) if `affine=True`.
        mean (torch.Tensor): 
            Stores the computed mean (per batch and feature) if `subtract_last=False`.
        stdev (torch.Tensor): 
            Stores the computed standard deviation (per batch and feature).
        last (torch.Tensor): 
            Stores the last timestep values (per batch and feature) 
            if `subtract_last=True`.

    Example:
        >>> x = torch.randn(8, 10, 16)  # [batch_size, seq_len, num_features]
        >>> revin = RevIN(num_features=16, affine=True, subtract_last=False)
        >>> # Normalization pass
        >>> x_norm = revin(x, mode='norm')
        >>> # De-normalization pass
        >>> x_denorm = revin(x_norm, mode='denorm')
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        # Initialize learnable parameters if affine is enabled
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Forward pass of RevIN. Depending on the mode, applies normalization or 
        de-normalization.

        Args:
            x (torch.Tensor): 
                The input tensor of shape [batch_size, seq_len, num_features].
            mode (str): 
                The operation mode, either 'norm' for normalization 
                or 'denorm' for de-normalization.

        Returns:
            torch.Tensor: 
                The transformed tensor. If mode='norm', it's normalized. 
                If mode='denorm', it's de-normalized.

        Raises:
            NotImplementedError: 
                If an invalid mode string is provided.
        """
        if mode == 'norm':
            # Compute necessary statistics (mean/last and stdev)
            self._get_statistics(x)
            # Apply normalization (subtract mean/last, divide by stdev, apply affine if enabled)
            x = self._normalize(x)
        elif mode == 'denorm':
            # Reverse the normalization using stored statistics (mean/last, stdev, affine)
            x = self._denormalize(x)
        else:
            raise NotImplementedError("Mode should be either 'norm' or 'denorm'.")
        return x

    def _init_params(self):
        """
        Initializes the learnable affine parameters: weight and bias. 
        These parameters are used to scale and shift the normalized output if `affine=True`.
        """
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor):
        """
        Computes and stores the mean (or last timestep) and standard deviation 
        required for normalization.

        Args:
            x (torch.Tensor): 
                Input tensor of shape [batch_size, seq_len, num_features]. 
        """
        # We'll reduce across the time dimension(s), keeping batch_size and features
        # so that we have statistics per [batch_size, 1, num_features].
        dim2reduce = tuple(range(1, x.ndim - 1))

        if self.subtract_last:
            # If subtract_last is True, store the last timestep for each [batch, feature].
            # Shape after unsqueeze is [batch_size, 1, num_features].
            self.last = x[:, -1, :].unsqueeze(1).detach()
        else:
            # Otherwise, compute the mean over the specified dimensions.
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        # Compute and store the standard deviation (always computed, regardless of subtract_last).
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input tensor using either:
          - (x - mean) / stdev, or
          - (x - last) / stdev
        based on the `subtract_last` flag.

        If affine parameters are enabled, also applies:
          normalized_tensor * affine_weight + affine_bias

        Args:
            x (torch.Tensor): 
                The input tensor of shape [batch_size, seq_len, num_features].

        Returns:
            torch.Tensor: 
                The normalized tensor with the same shape as `x`.
        """
        # Subtract either mean or the last-timestep values
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean

        # Divide by standard deviation
        x = x / self.stdev

        # Optionally apply learnable scale and shift
        if self.affine:
            x = x * self.affine_weight  # per-feature scaling
            x = x + self.affine_bias    # per-feature bias

        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        De-normalizes the tensor using stored statistics (mean/last and stdev),
        effectively reversing the normalization process.

        If affine parameters are enabled, the operation reverses the scaling and bias:
          (x - affine_bias) / affine_weight

        Args:
            x (torch.Tensor): 
                The normalized tensor of shape [batch_size, seq_len, num_features].

        Returns:
            torch.Tensor: 
                The de-normalized tensor with the same shape as `x`.
        """
        # If affine is enabled, first reverse the learnable transformation
        if self.affine:
            x = x - self.affine_bias
            # Use (affine_weight + self.eps*self.eps) if that's intended for stability
            # but typically you'd do x = x / (self.affine_weight + self.eps).
            x = x / (self.affine_weight + self.eps * self.eps)

        # Multiply by the stored standard deviation to revert
        x = x * self.stdev

        # Finally, add back the mean or the last timestep to complete de-normalization
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean

        return x

###############################################################################
# Blocks and Stages
###############################################################################

class Flatten_Head(nn.Module):
    """
    Forecast task head that flattens and applies a linear transformation.

    This can operate either in an individual (per-variable) manner or collectively.

    Args:
        individual (bool): If True, applies a separate linear layer per variable.
        n_vars (int): Number of variables in the input.
        nf (int): Number of features in the flattened dimension.
        target_window (int): Output dimension of the linear projection (e.g., prediction length).
        head_dropout (float, optional): Dropout probability. Defaults to 0.
    """
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super(Flatten_Head, self).__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        Forward pass of the Flatten_Head.

        Args:
            x (Tensor): Input of shape [batch_size, n_vars, d_model, patch_num].

        Returns:
            Tensor: Output of shape [batch_size, n_vars, target_window] if individual=True,
                    otherwise [batch_size, n_vars, target_window].
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                # Flatten the [d_model, patch_num] dimensions
                z = self.flattens[i](x[:, i, :, :])  # z: [batch_size, d_model * patch_num]
                z = self.linears[i](z)              # z: [batch_size, target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)          # [batch_size, n_vars, target_window]
        else:
            x = self.flatten(x)                    # Flatten across [d_model, patch_num]
            x = self.linear(x)                     # [batch_size, n_vars, target_window]
            x = self.dropout(x)
        return x

class ConvBN(nn.Module):
    """
    A 1D Convolution + 1D BatchNorm block.

    By default, if `padding` is not specified (`None`), it is automatically set
    to `kernel_size // 2`, a common choice to maintain spatial resolution when
    `stride=1`.

    Args:
        in_channels (int): 
            Number of channels in the input signal.
        out_channels (int): 
            Number of channels produced by the convolution.
        kernel_size (int): 
            Size of the convolving kernel.
        stride (int, optional): 
            Stride of the convolution. Defaults to 1.
        padding (int, optional): 
            Zero-padding added to both sides of the input. If None, set to 
            `kernel_size // 2`. Defaults to None.
        dilation (int, optional): 
            Spacing between kernel elements. Defaults to 1.
        groups (int, optional): 
            Number of blocked connections from input channels to output channels.
            Defaults to 1.
        bias (bool, optional): 
            If True, adds a learnable bias to the output of the convolution. 
            Defaults to False.

    Example:
        >>> x = torch.randn(8, 16, 50)  # (batch_size=8, in_channels=16, length=50)
        >>> conv_bn_block = ConvBN(in_channels=16, out_channels=32, kernel_size=3)
        >>> y = conv_bn_block(x)  # shape: (8, 32, 50)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False
    ):
        super().__init__()

        # If no padding is specified, use half the kernel size for "same" padding (common practice).
        if padding is None:
            padding = kernel_size // 2

        # Define the 1D convolution layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # Define the 1D batch normalization layer
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Conv + BatchNorm block.

        Args:
            x (torch.Tensor): 
                Input tensor of shape [batch_size, in_channels, length].

        Returns:
            torch.Tensor:
                Output tensor of shape [batch_size, out_channels, new_length].
        """
        # Apply the convolution
        x = self.conv(x)
        # Apply batch normalization
        x = self.bn(x)
        return x

class ReparamLargeKernelConv(nn.Module):
    """
    Re-parameterizable convolution module that can optionally fuse:
      1) A 'large kernel' conv+BN branch, and
      2) An optional 'small kernel' conv+BN branch

    Into a single convolution layer for inference-time optimizations.

    By default, you can train with both large and small kernels in parallel.
    Later, you can call :meth:`merge_kernel` to fuse the weights and biases
    into a single equivalent convolution layer, which improves efficiency
    in deployment without changing the output.

    Args:
        in_channels (int): 
            Number of input channels.
        out_channels (int): 
            Number of output channels.
        kernel_size (int): 
            Size of the larger kernel.
        stride (int): 
            Stride for the convolution.
        groups (int): 
            Number of groups for the convolution.
        small_kernel (int or None): 
            Size of the smaller kernel (must be <= `kernel_size`). If None, 
            only the large kernel branch is used.
        small_kernel_merged (bool, optional): 
            If True, indicates that the module is already fused into a 
            single convolution. Defaults to False.
        nvars (int, optional): 
            Number of variables used to parameterize certain advanced logic 
            (not fully demonstrated in this snippet). Defaults to 7.

    Example:
        >>> # Suppose we have a 1D input of shape (batch=8, channels=16, length=64)
        >>> x = torch.randn(8, 16, 64)
        >>>
        >>> # Re-param module with a large kernel=7 and small kernel=3
        >>> reparam_conv = ReparamLargeKernelConv(
        ...     in_channels=16,
        ...     out_channels=32,
        ...     kernel_size=7,
        ...     stride=1,
        ...     groups=1,
        ...     small_kernel=3
        ... )
        >>>
        >>> # Forward pass uses both large and small kernel branches
        >>> y = reparam_conv(x)
        >>> print(y.shape)
        ... # torch.Size([8, 32, 64])
        >>>
        >>> # Merge the kernels into a single Conv1d module for faster inference
        >>> reparam_conv.merge_kernel()
        >>> y_merged = reparam_conv(x)
        >>> # y and y_merged should be numerically the same within floating-point error
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int,
        small_kernel_merged: bool = False,
        nvars: int = 7
    ):
        super(ReparamLargeKernelConv, self).__init__()

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # By default, we set padding such that the convolution does not change the output length
        padding = kernel_size // 2

        if small_kernel_merged:
            # If already merged, we only keep a single convolution layer (with bias).
            self.lkb_reparam = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=True
            )
        else:
            # Otherwise, we create two parallel paths:
            # 1) The large-kernel branch (conv + BN)
            # 2) Optionally, the small-kernel branch (conv + BN)

            self.lkb_origin = ConvBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=False  # Typically, BN replaces bias
            )

            if small_kernel is not None:
                assert small_kernel <= kernel_size, (
                    "The small kernel cannot be larger than the large kernel!"
                )
                self.small_conv = ConvBN(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=small_kernel,
                    stride=stride,
                    padding=small_kernel // 2,
                    groups=groups,
                    dilation=1,
                    bias=False
                )

    @staticmethod
    def fuse_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
        """
        Compute fused weights and bias for a Conv1d + BatchNorm1d pair.

        This function extracts the parameters of the convolution (`conv.weight` and 
        optionally `conv.bias`) and the batch normalization (`bn.running_mean`, 
        `bn.running_var`, `bn.weight`, `bn.bias`) to produce:

        - A fused convolution weight that includes BN scaling
        - A fused bias that includes BN shift

        Args:
            conv (nn.Conv1d): 
                The convolution layer to fuse.
            bn (nn.BatchNorm1d): 
                The batch normalization layer to fuse.

        Returns:
            (torch.Tensor, torch.Tensor): 
                (fused_weight, fused_bias).
        """
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        # Compute the standard deviation of the running variance
        std = (running_var + eps).sqrt()
        # Scale factor for each output channel
        t = (gamma / std).reshape(-1, 1, 1)

        # Fused kernel is elementwise scaled by t
        fused_weight = kernel * t
        # Fused bias includes BN shift
        # If conv.bias is present, it would be added here as well, but in this code
        # we assume 'bias=False' for conv layers.
        fused_bias = beta - running_mean * gamma / std

        return fused_weight, fused_bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the re-parameterizable large kernel conv module.

        Depending on whether the module is already fused (`lkb_reparam`) or not 
        (`lkb_origin`, possibly plus `small_conv`), the forward pass changes:

        - If `self.lkb_reparam` exists, we apply it directly.
        - Otherwise, we apply `self.lkb_origin` and add the result of `self.small_conv` 
          if it exists.

        Args:
            inputs (torch.Tensor): 
                The input tensor of shape [B, C, L], where B is batch size, 
                C is number of channels, and L is the sequence length.

        Returns:
            torch.Tensor:
                The output tensor of shape [B, out_channels, L'].
        """
        if hasattr(self, 'lkb_reparam'):
            # Fused single convolution path
            return self.lkb_reparam(inputs)
        else:
            # Large kernel branch
            out = self.lkb_origin(inputs)
            # Small kernel branch (if defined), add the outputs
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
            return out

    def PaddingTwoEdge1d(
        self, 
        x: torch.Tensor, 
        pad_length_left: int, 
        pad_length_right: int, 
        pad_values: float = 0
    ) -> torch.Tensor:
        """
        Zero-pads (or constant-pads) a 1D convolution kernel on both left and right edges.

        This is used when fusing small kernel weights into the larger kernel 
        dimension during re-parameterization.

        Args:
            x (torch.Tensor): 
                The convolution kernel of shape [out_channels, in_channels, kernel_size].
            pad_length_left (int): 
                Number of elements to pad on the left side.
            pad_length_right (int): 
                Number of elements to pad on the right side.
            pad_values (float, optional): 
                Value used for padding. Defaults to 0.

        Returns:
            torch.Tensor:
                The padded kernel tensor with shape 
                [out_channels, in_channels, kernel_size + pad_length_left + pad_length_right].
        """
        D_out, D_in, ks = x.shape

        # Create left and right padding
        if pad_values == 0:
            pad_left = torch.zeros(D_out, D_in, pad_length_left, device=x.device, dtype=x.dtype)
            pad_right = torch.zeros(D_out, D_in, pad_length_right, device=x.device, dtype=x.dtype)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left, device=x.device, dtype=x.dtype) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right, device=x.device, dtype=x.dtype) * pad_values

        # Concatenate paddings around the original kernel
        x = torch.cat([pad_left, x, pad_right], dim=-1)
        return x

    def get_equivalent_kernel_bias(self):
        """
        Fuse the large and (optionally) small kernel conv paths into a single 
        equivalent kernel and bias.

        This involves:
          1) Fusing conv + BN of the large kernel path into a single kernel/bias.
          2) Fusing conv + BN of the small kernel path (if it exists).
          3) Zero/constant-padding the small kernel to match the large kernel's size.
          4) Summing them together to form a single fused kernel/bias.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                A 2-tuple of (fused_kernel, fused_bias).
        """
        # 1) Fuse the large kernel path
        eq_k, eq_b = self.fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)

        # 2) If small kernel path exists, fuse it and combine
        if hasattr(self, 'small_conv'):
            small_k, small_b = self.fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            # Match the larger kernel size by padding
            left_pad = (self.kernel_size - self.small_kernel) // 2
            right_pad = (self.kernel_size - self.small_kernel) - left_pad
            small_k_padded = self.PaddingTwoEdge1d(small_k, left_pad, right_pad, pad_values=0)
            eq_k += small_k_padded

        return eq_k, eq_b

    def merge_kernel(self):
        """
        Fuse the large and small kernel branches into a single Conv1d module.

        This method:
          - Computes the fused kernel and bias via :meth:`get_equivalent_kernel_bias`.
          - Creates a new `nn.Conv1d` (self.lkb_reparam) with those fused parameters.
          - Removes the original large/small kernel modules (`lkb_origin` and `small_conv`).

        After calling this, the forward pass will use `self.lkb_reparam` alone, 
        effectively reducing inference overhead.
        """
        # Get the fused kernel and bias
        eq_k, eq_b = self.get_equivalent_kernel_bias()

        # Create a new single convolution with bias
        # The settings match those of the original large kernel convolution
        self.lkb_reparam = nn.Conv1d(
            in_channels=self.lkb_origin.conv.in_channels,
            out_channels=self.lkb_origin.conv.out_channels,
            kernel_size=self.lkb_origin.conv.kernel_size,
            stride=self.lkb_origin.conv.stride,
            padding=self.lkb_origin.conv.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.lkb_origin.conv.groups,
            bias=True
        )

        # Set its parameters from the fused results
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b

        # Remove the original sub-modules
        del self.lkb_origin
        if hasattr(self, 'small_conv'):
            del self.small_conv

class Block(nn.Module):
    """
    A basic block that performs:
    1) A depthwise large-kernel convolution (optionally re-parameterized),
    2) Followed by two Conv-FeedForward (ConvFFN) layers,
    3) With a residual connection at the end.

    The block expects input of shape [B, M, D, N], where:
    - B = batch_size
    - M = number_of_variables
    - D = model_dimension
    - N = sequence_length

    Internally, it reshapes data for depthwise convolution (merging M and D),
    applies batch normalization, then uses two separate ConvFFN sub-blocks (with
    dropout and GELU activation), and finally adds the original input (residual).

    Args:
        large_size (int): 
            Size of the larger convolution kernel.
        small_size (int): 
            Size of the smaller convolution kernel (optional; if None, only the large kernel is used).
        dmodel (int): 
            Model dimension (D).
        dff (int): 
            Dimension of the feed-forward layers (hidden dimension).
        nvars (int): 
            Number of variables (M).
        small_kernel_merged (bool, optional): 
            Indicates whether the small kernel has already been merged into
            the large kernel for inference. Defaults to False.
        drop (float, optional): 
            Dropout probability used in the feed-forward layers. Defaults to 0.1.
    """

    def __init__(
        self, 
        large_size: int, 
        small_size: int, 
        dmodel: int, 
        dff: int, 
        nvars: int, 
        small_kernel_merged: bool = False, 
        drop: float = 0.1
    ):
        super(Block, self).__init__()

        # 1) Depthwise large kernel conv (ReparamLargeKernelConv may handle large+small internally).
        #    in_channels/out_channels = nvars * dmodel to match shape [B, M*D, N].
        self.dw = ReparamLargeKernelConv(
            in_channels=nvars * dmodel,
            out_channels=nvars * dmodel,
            kernel_size=large_size,
            stride=1,
            groups=nvars * dmodel,       # depthwise => groups = in_channels
            small_kernel=small_size,
            small_kernel_merged=small_kernel_merged,
            nvars=nvars
        )

        # 2) Batch Normalization (across the 'D' dimension after depthwise conv).
        #    We'll reshape x to [B*M, D, N] before applying BN.
        self.norm = nn.BatchNorm1d(dmodel)

        # 3) First ConvFFN sub-block.
        #    - Pointwise Conv 1: (nvars*dmodel) -> (nvars*dff)
        #    - Activation (GELU)
        #    - Pointwise Conv 2: (nvars*dff) -> (nvars*dmodel)
        #    - Dropout before/after the activation
        self.ffn1pw1 = nn.Conv1d(
            in_channels=nvars * dmodel,
            out_channels=nvars * dff,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=nvars  # separate groups for each variable
        )
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(
            in_channels=nvars * dff,
            out_channels=nvars * dmodel,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=nvars
        )
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # 4) Second ConvFFN sub-block.
        #    Similar structure, but the grouping is set differently: groups = dmodel.
        self.ffn2pw1 = nn.Conv1d(
            in_channels=nvars * dmodel,
            out_channels=nvars * dff,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=dmodel
        )
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(
            in_channels=nvars * dff,
            out_channels=nvars * dmodel,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=dmodel
        )
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        # Keep track of the ratio for reference if needed.
        self.ffn_ratio = dff // dmodel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Block.

        The input and output tensors have shape [B, M, D, N], where:
            B = batch_size,
            M = number_of_variables,
            D = model_dimension,
            N = sequence_length.

        Steps:
          1) Reshape to [B, M*D, N] and apply depthwise large kernel conv.
          2) Reshape to [B*M, D, N] and apply batch normalization.
          3) First ConvFFN sub-block: 
             - 1x1 conv, dropout, GELU, 1x1 conv, dropout
          4) Second ConvFFN sub-block (with dimension permute and reshape):
             - permute to [B, D, M, N], reshape to [B, D*M, N], 1x1 conv, etc.
          5) Residual connection with the original input.

        Args:
            x (torch.Tensor): 
                Input tensor of shape [B, M, D, N].

        Returns:
            torch.Tensor:
                Output tensor of shape [B, M, D, N].
        """
        # Save the original input for the residual connection
        input_res = x
        B, M, D, N = x.shape

        # --- Depthwise Convolution ---
        x = x.reshape(B, M * D, N)  # [B, M*D, N]
        x = self.dw(x)              # apply reparam large kernel conv
        x = x.reshape(B, M, D, N)   # revert shape

        # --- Batch Normalization ---
        x = x.reshape(B * M, D, N)  # [B*M, D, N]
        x = self.norm(x)
        x = x.reshape(B, M, D, N)   # revert shape

        # --- First ConvFFN ---
        x = x.reshape(B, M * D, N)            # [B, M*D, N]
        x = self.ffn1drop1(self.ffn1pw1(x))   # 1x1 conv + dropout
        x = self.ffn1act(x)                  # GELU
        x = self.ffn1drop2(self.ffn1pw2(x))   # 1x1 conv + dropout
        x = x.reshape(B, M, D, N)            # revert shape

        # --- Second ConvFFN ---
        # permute to [B, D, M, N] for grouping by D
        x = x.permute(0, 2, 1, 3)             # => [B, D, M, N]
        x = x.reshape(B, D * M, N)            # => [B, D*M, N]
        x = self.ffn2drop1(self.ffn2pw1(x))   # 1x1 conv + dropout
        x = self.ffn2act(x)                  # GELU
        x = self.ffn2drop2(self.ffn2pw2(x))   # 1x1 conv + dropout
        x = x.reshape(B, D, M, N)            # => [B, D, M, N]
        x = x.permute(0, 2, 1, 3)             # => [B, M, D, N]

        # --- Residual Connection ---
        x = input_res + x

        return x

class Stage(nn.Module):
    """
    A Stage that stacks multiple consecutive Blocks.

    Each block is configured with:
      - A depthwise large kernel convolution (via ReparamLargeKernelConv),
      - Two ConvFFN sub-blocks,
      - Residual connection.

    Args:
        ffn_ratio (int): 
            The ratio of hidden dimension (dff) to model dimension (dmodel).
        num_blocks (int): 
            Number of consecutive blocks in this stage.
        large_size (int): 
            Size of the larger convolution kernel for each block.
        small_size (int): 
            Size of the smaller convolution kernel (for re-param). 
            If None, only large kernel path is used.
        dmodel (int): 
            Model dimension for each block.
        dw_model (int): 
            Depthwise model dimension (not strictly used differently here, 
            but available for customization).
        nvars (int): 
            Number of variables.
        small_kernel_merged (bool, optional): 
            Whether the small kernel has already been merged for inference 
            in each block. Defaults to False.
        drop (float, optional): 
            Dropout probability for feed-forward layers. Defaults to 0.1.

    Example:
        >>> # Suppose we have a batch of data x with shape [B=2, M=8, D=64, N=100].
        >>> x = torch.randn(2, 8, 64, 100)
        >>> stage = Stage(
        ...     ffn_ratio=4,
        ...     num_blocks=2,
        ...     large_size=7,
        ...     small_size=3,
        ...     dmodel=64,
        ...     dw_model=64,
        ...     nvars=8,
        ...     small_kernel_merged=False,
        ...     drop=0.1
        ... )
        >>> y = stage(x)
        >>> print(y.shape) 
        ... # Expected [2, 8, 64, 100]
    """

    def __init__(
        self, 
        ffn_ratio: int, 
        num_blocks: int, 
        large_size: int, 
        small_size: int, 
        dmodel: int, 
        dw_model: int, 
        nvars: int,
        small_kernel_merged: bool = False, 
        drop: float = 0.1
    ):
        super(Stage, self).__init__()

        # Compute feed-forward dimension
        d_ffn = dmodel * ffn_ratio

        # Create a list of 'num_blocks' Blocks
        blks = []
        for i in range(num_blocks):
            blk = Block(
                large_size=large_size,
                small_size=small_size,
                dmodel=dmodel,
                dff=d_ffn,
                nvars=nvars,
                small_kernel_merged=small_kernel_merged,
                drop=drop
            )
            blks.append(blk)

        # Wrap blocks in a ModuleList for PyTorch
        self.blocks = nn.ModuleList(blks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Stage, applying each Block sequentially.

        The input and output are of shape [B, M, D, N].

        Args:
            x (torch.Tensor): 
                Input tensor of shape [B, M, D, N].

        Returns:
            torch.Tensor:
                Output tensor of shape [B, M, D, N].
        """
        # Sequentially apply each block
        for blk in self.blocks:
            x = blk(x)
        return x


###############################################################################
# Model ModernTCN
###############################################################################
class ModernTCN(nn.Module):
    """
    The main TCN (Temporal Convolutional Network) backbone with a modern design.

    This architecture includes:
    - An optional RevIN normalization layer (for mean/last subtraction),
    - A stem layer for initial patching/downsampling,
    - Multiple stages (each contains downsampling + Stage of Blocks),
    - A head for either classification or other tasks.

    Args:
        task_name (str): 
            The type of task. For example, "classification" or "forecasting".
        c_in (int): 
            Number of input channels/variables (features). Often labeled as M.
        patch_size (int, optional): 
            The kernel size used in the initial stem convolution. Defaults to 16.
        patch_stride (int, optional): 
            The stride used in the initial stem convolution. Defaults to 8.
        downsample_ratio (int, optional): 
            Downsampling factor applied in each downsample layer. Defaults to 2.
        ffn_ratio (int, optional): 
            Ratio for the feed-forward dimension in the blocks, i.e., d_ff = d_model * ffn_ratio. 
            Defaults to 2.
        num_blocks (List[int], optional): 
            Number of blocks in each stage. Defaults to [1, 1, 1, 1].
        large_size (List[int], optional): 
            Large convolution kernel sizes for each stage. Defaults to [31, 29, 27, 13].
        small_size (List[int], optional): 
            Small convolution kernel sizes for each stage (for re-param). 
            Defaults to [5, 5, 5, 5].
        dims (List[int], optional): 
            List of model dimensions for each stage. Defaults to [256, 256, 256, 256].
        dw_dims (List[int], optional): 
            Depthwise dimensions for each stage. Defaults to [256, 256, 256, 256].
        small_kernel_merged (bool, optional): 
            If True, the large and small kernels are merged for inference in each block. 
            Defaults to False.
        backbone_dropout (float, optional): 
            Dropout probability used in the backbone. Defaults to 0.1.
        head_dropout (float, optional): 
            Dropout probability used in the head. Defaults to 0.1.
        use_multi_scale (bool, optional): 
            Whether to use multi-scale features in the head. Defaults to False.
        revin (bool, optional): 
            If True, uses RevIN for normalization before the backbone. Defaults to True.
        affine (bool, optional): 
            If RevIN is used and this is True, it has learnable affine parameters. Defaults to True.
        subtract_last (bool, optional): 
            If RevIN is used and this is True, subtracts the last timestep instead of the mean. 
            Defaults to False.
        seq_len (int, optional): 
            Sequence length of the input. Defaults to 512.
        individual (bool, optional): 
            If True, uses an individual linear projection per variable in the head. Defaults to False.
        target_window (int, optional): 
            Output length for forecasting tasks. Defaults to 6.
        class_drop (float, optional): 
            Dropout probability for classification. Defaults to 0.
        c_out (int, optional): 
            Number of classes for classification. Defaults to 10.
    """

    def __init__(
        self,
        task_name: str,
        c_in: int = 7,
        patch_size: int = 16,
        patch_stride: int = 8,
        downsample_ratio: int = 2,
        ffn_ratio: int = 2,
        num_blocks: List[int] = [1, 1, 1, 1],
        large_size: List[int] = [31, 29, 27, 13],
        small_size: List[int] = [5, 5, 5, 5],
        dims: List[int] = [256, 256, 256, 256],
        dw_dims: List[int] = [256, 256, 256, 256],
        small_kernel_merged: bool = False,
        backbone_dropout: float = 0.1,
        head_dropout: float = 0.1,
        use_multi_scale: bool = False,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        seq_len: int = 512,
        individual: bool = False,
        target_window: int = 6,
        class_drop: float = 0.0,
        c_out: int = 10
    ):
        super(ModernTCN, self).__init__()

        self.task_name = task_name
        self.class_drop = class_drop
        self.c_out = c_out

        # ---------------------------
        # 1) Optional RevIN Layer
        # ---------------------------
        # If 'revin' is True, apply reversible instance normalization 
        # (which can subtract the mean or last timestep).
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # ------------------------------------------------
        # 2) Stem Layer + Downsampling Layers
        # ------------------------------------------------
        # This first "stem" converts the input dimension into dims[0].
        # patch_size = convolution kernel size, patch_stride = stride
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(c_in, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)

        # If multiple stages are used, set up additional downsample layers.
        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(
                        dims[i],
                        dims[i + 1],
                        kernel_size=downsample_ratio,
                        stride=downsample_ratio
                    ),
                )
                self.downsample_layers.append(downsample_layer)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        # ------------------------------------------------
        # 3) Main Backbone (Stages of Blocks)
        # ------------------------------------------------
        self.stages = nn.ModuleList()
        # Each stage is composed of multiple blocks (in a Stage module).
        # e.g., Stage(...) -> multiple Blocks inside.
        for stage_idx in range(self.num_stage):
            layer = Stage(
                ffn_ratio=ffn_ratio,
                num_blocks=num_blocks[stage_idx],
                large_size=large_size[stage_idx],
                small_size=small_size[stage_idx],
                dmodel=dims[stage_idx],
                dw_model=dw_dims[stage_idx],
                nvars=c_in,
                small_kernel_merged=small_kernel_merged,
                drop=backbone_dropout
            )
            self.stages.append(layer)

        # ------------------------------------------------
        # 4) Head (Flattening + Final Linear Layers)
        # ------------------------------------------------
        # patch_num = length of feature dimension after the initial stride
        patch_num = seq_len // patch_stride
        self.c_in = c_in
        self.individual = individual
        d_model = dims[self.num_stage - 1]

        # Two options for multi-scale usage: either produce multi-scale outputs,
        # or a single flattened representation.
        if use_multi_scale:
            # Multi-scale approach: the entire d_model * patch_num is used
            self.head_nf = d_model * patch_num
            self.head = Flatten_Head(
                self.individual,
                self.c_in,
                self.head_nf,
                target_window,
                head_dropout=head_dropout
            )
        else:
            # If not multi-scale, account for further downsampling 
            # from (self.num_stage - 1) extra layers.
            total_downsample = pow(downsample_ratio, (self.num_stage - 1))
            if patch_num % total_downsample == 0:
                self.head_nf = d_model * (patch_num // total_downsample)
            else:
                self.head_nf = d_model * (patch_num // total_downsample + 1)

            self.head = Flatten_Head(
                self.individual,
                self.c_in,
                self.head_nf,
                target_window,
                head_dropout=head_dropout
            )

        # ------------------------------------
        # 5) Classification-Specific Elements
        # ------------------------------------
        # If the task is classification, define an activation function, dropout,
        # and linear head to generate class logits.
        if self.task_name == 'classification':
            self.act_class = F.gelu
            self.class_dropout = nn.Dropout(self.class_drop)

            # The classification head expects a flattened vector of dimension
            # (B, in_dim_for_class). We multiply the final # of variables 
            # by the flattened dimension from the backbone.
            in_dim_for_class = self.c_in * self.head_nf

            self.head_class = nn.Linear(in_dim_for_class, self.c_out)

    def forward_feature(self, x: torch.Tensor, te: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the backbone (stem -> downsampling -> stages).

        Args:
            x (torch.Tensor): 
                Input tensor of shape [B, M, L],
                where B = batch_size, M = number_of_channels, L = sequence_length.
            te (Optional[torch.Tensor]): 
                Additional temporal embeddings (not used in this implementation).

        Returns:
            torch.Tensor: 
                Backbone features of shape [B, M, D, N], where D = dims[-1] and 
                N depends on the final downsampling.
        """
        B, M, L = x.shape

        # If RevIN is enabled, apply 'norm' before the backbone
        if self.revin:
            x = self.revin_layer(x, mode='norm')  # shape remains [B, M, L]

        # Reshape to [B, M, 1, L] to conform with the subsequent operations
        x = x.unsqueeze(-2)  # => [B, M, 1, L]

        # For each stage, apply the corresponding downsample layer and then the stage blocks
        for i in range(self.num_stage):
            B_, M_, D_, N_ = x.shape

            # Flatten M_ and D_ so that the downsampling layer sees 
            # a single dimension of "M_*D_"
            x = x.reshape(B_ * M_, D_, N_)

            # ---------------------------
            # Stem / Downsample Padding
            # ---------------------------
            if i == 0:
                # For the very first stem, if patch_size != patch_stride, 
                # we pad to ensure the convolution can move properly.
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    # Repeat the last value 'pad_len' times along the sequence axis
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                # For subsequent downsample layers,
                # if the sequence length isn't divisible by 'downsample_ratio',
                # pad so that the convolution's stride can be applied.
                remainder = N_ % self.downsample_ratio
                if remainder != 0:
                    pad_len = self.downsample_ratio - remainder
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)

            # ---------------------------
            # Downsample
            # ---------------------------
            x = self.downsample_layers[i](x)  # => new shape [B_*M_, D' (dims[i+1]), N']
            _, D__, N__ = x.shape

            # Reshape back to [B_, M_, D__, N__]
            x = x.reshape(B_, M_, D__, N__)

            # Apply the stage (multiple blocks)
            x = self.stages[i](x)

        return x

    def classification(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification tasks.

        This method:
         - Passes the input through the backbone,
         - Applies a GELU activation + dropout,
         - Flattens the features,
         - Finally applies a linear layer for class logits.

        Args:
            x (torch.Tensor): 
                Input tensor of shape [B, M, L].

        Returns:
            torch.Tensor: 
                Logits tensor of shape [B, c_out].
        """
        # Feature extraction
        x = self.forward_feature(x)
        # Classification activation + dropout
        x = self.act_class(x)
        x = self.class_dropout(x)
        # Flatten for linear layer
        x = x.reshape(x.shape[0], -1)
        # Linear classifier
        x = self.head_class(x)
        return x

    def forward(self, x: torch.Tensor, te: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward method for ModernTCN.

        If task_name is "classification", returns logits of shape [B, c_out].
        Otherwise, you can customize it for forecasting or other tasks by 
        using different heads or returning different shapes.

        Args:
            x (torch.Tensor): 
                Input tensor of shape [B, M, L].
            te (Optional[torch.Tensor]): 
                Additional temporal embeddings (not used here).

        Returns:
            torch.Tensor:
                Task-specific output. If classification -> [B, c_out].
                If other tasks, can be shaped differently (e.g., [B, M, target_window]).
        """
        if self.task_name == 'classification':
            return self.classification(x)
        else:
            # For other tasks (e.g., forecasting), you might do:
            features = self.forward_feature(x, te=te)
            # Then pass to the head for final output
            out = self.head(features)
            return out

    def structural_reparam(self):
        """
        Merges large and small kernels in all modules that have the 'merge_kernel' attribute.

        This is useful for inference optimization: after training with
        both large and small kernels, merge them into a single kernel
        to reduce runtime overhead.
        """
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()


