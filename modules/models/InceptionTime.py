import torch
import torch.nn as nn
import torch.nn.functional as F

class NoOp(nn.Module):
    """A placeholder module that returns the input unchanged."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class InceptionModule(nn.Module):
    """
    One inception module for 1D signals.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for each branch.
    kernel_size : int
        Base kernel size for the main convolution branches.
    bottleneck : bool
        Whether or not to include a 1x1 'bottleneck' convolution 
        that reduces the channel dimension before the larger convs.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 40, bottleneck: bool = True):
        super().__init__()
        # Compute the kernel sizes for the three parallel conv branches
        ks = [kernel_size // (2 ** i) for i in range(3)]
        # Ensure kernel sizes are odd
        ks = [k if k % 2 != 0 else k - 1 for k in ks]
        
        # Disable bottleneck if in_channels is too small
        if in_channels <= 1:
            bottleneck = False

        # Bottleneck (1x1 conv) or NoOp
        self.bottleneck = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if bottleneck else NoOp()
        )

        # Parallel 1D convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(
                out_channels if bottleneck else in_channels, 
                out_channels, 
                kernel_size=k,
                padding=k//2, 
                bias=False
            ) 
            for k in ks
        ])

        # MaxPool + Conv branch
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        # BatchNorm + Activation
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through bottleneck or NoOp
        y = self.bottleneck(x)

        # Convolution branches
        conv_outputs = [conv(y) for conv in self.convs] + [self.maxconvpool(x)]

        # Concatenate along the channel dimension
        out = torch.cat(conv_outputs, dim=1)

        # BatchNorm + ReLU
        out = self.bn(out)
        out = self.act(out)
        return out


class InceptionBlock(nn.Module):
    """
    A stack of InceptionModules with optional residual connections.
    
    Every 3 layers, a residual connection is applied:
    - If the input and output dimensions match, a simple BatchNorm is used.
    - Otherwise, a 1x1 convolution adjusts dimensions.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels to the first InceptionModule.
    out_channels : int
        Number of output channels for each branch in the InceptionModules.
    residual : bool
        Whether or not to apply residual connections.
    depth : int
        How many InceptionModules to stack.
    kernel_size : int
        Base kernel size for each InceptionModule branch.
    bottleneck : bool
        Whether or not to include 1x1 'bottleneck' conv in InceptionModules.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int = 32, 
        residual: bool = True, 
        depth: int = 6,
        kernel_size: int = 40,
        bottleneck: bool = True
    ):
        super().__init__()
        self.residual = residual
        self.depth = depth
        
        self.inception_layers = nn.ModuleList()
        self.shortcut_layers = nn.ModuleList()

        for d in range(depth):
            # Decide the input dim for this inception layer
            in_ch = in_channels if d == 0 else out_channels * 4
            # Create the InceptionModule
            self.inception_layers.append(
                InceptionModule(in_ch, out_channels, kernel_size=kernel_size, bottleneck=bottleneck)
            )

            # Every 3rd layer (d % 3 == 2) potentially needs a shortcut
            if residual and d % 3 == 2:
                if d == 2:
                    shortcut_in = in_channels
                else:
                    shortcut_in = out_channels * 4
                    
                shortcut_out = out_channels * 4

                # If input and output dims match, use BN only
                if shortcut_in == shortcut_out:
                    self.shortcut_layers.append(nn.BatchNorm1d(shortcut_in))
                else:
                    # Else, use a 1x1 conv (no activation)
                    self.shortcut_layers.append(
                        nn.Conv1d(shortcut_in, shortcut_out, kernel_size=1, bias=False)
                    )

        # Activation after residual addition
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        shortcut_count = 0

        for d in range(self.depth):
            x = self.inception_layers[d](x)

            # Every 3rd layer, add residual
            if self.residual and d % 3 == 2:
                shortcut = self.shortcut_layers[shortcut_count]
                shortcut_count += 1

                # Apply the shortcut layer to the old residual
                sc_out = shortcut(residual)
                x = x + sc_out
                x = self.act(x)
                
                # Update residual
                residual = x

        return x


class InceptionTime(nn.Module):
    """
    An InceptionTime model for time-series classification/regression.
    
    Parameters
    ----------
    c_in : int
        Number of input channels (features).
    c_out : int
        Number of output channels (classes).
    seq_len : int, optional
        Input sequence length (not used here, but kept for API consistency).
    nf : int
        Number of output filters (out_channels) for InceptionModules.
    kernel_size : int
        Base kernel size for each InceptionModule branch.
    bottleneck : bool
        Whether or not to use a 1x1 'bottleneck' conv in InceptionModules.
    residual : bool
        Whether or not to use residual connections in the InceptionBlock.
    depth : int
        Number of InceptionModules to stack in the InceptionBlock.
    """
    def __init__(
        self, 
        c_in: int, 
        c_out: int, 
        seq_len: int = None, 
        nf: int = 32, 
        kernel_size: int = 40,
        bottleneck: bool = True,
        residual: bool = True,
        depth: int = 6
    ):
        super().__init__()

        # The main Inception block
        self.inception_block = InceptionBlock(
            in_channels=c_in,
            out_channels=nf,
            residual=residual,
            depth=depth,
            kernel_size=kernel_size,
            bottleneck=bottleneck
        )
        
        # Global Average Pooling over time dimension
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        
        # Final linear layer
        self.fc = nn.Linear(nf * 4, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through Inception block
        x = self.inception_block(x)
        
        # Global average pooling (squeezes the time dimension)
        x = self.gap(x)
        x = x.squeeze(-1)  # (B, nf*4, 1) -> (B, nf*4)
        
        # Linear classification/regression head
        x = self.fc(x)
        return x
