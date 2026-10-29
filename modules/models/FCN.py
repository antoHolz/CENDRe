import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    """
    Chomp layer to remove excess padding in 1D convolutions.

    Args:
        chomp_size (int): The number of elements to remove from the end.
    """
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    A single Temporal Block in the FCN architecture.

    Args:
        ni (int): Number of input channels.
        nf (int): Number of output channels.
        ks (int): Kernel size.
        stride (int): Stride for convolutions.
        dilation (int): Dilation rate for convolutions.
        padding (int): Padding size for convolutions.
        dropout (float): Dropout rate.
        ndim (int): Number of spatial dimensions (1 or 2).
    """
    def __init__(
        self, 
        ni: int, 
        nf: int, 
        ks: int, 
        stride: int, 
        dilation: int, 
        padding: int, 
        dropout: float = 0.0,
        ndim: int = 1
    ):
        super(TemporalBlock, self).__init__()
        Conv = nn.Conv1d if ndim == 1 else nn.Conv2d
        Chomp = Chomp1d if ndim ==1 else nn.Identity
        self.conv1 = weight_norm(Conv(ni, nf, kernel_size=ks, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp(padding) if ndim ==1 else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = weight_norm(Conv(nf, nf, kernel_size=ks, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp(padding) if ndim ==1 else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2
        )
        self.downsample = Conv(ni, nf, kernel_size=1) if ni != nf else None
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network consisting of multiple Temporal Blocks.

    Args:
        c_in (int): Number of input channels.
        layers (List[int]): Number of output channels for each Temporal Block.
        ks (int): Kernel size.
        dropout (float): Dropout rate.
        ndim (int): Number of spatial dimensions (1 or 2).
    """
    def __init__(
        self, 
        c_in: int, 
        layers: List[int], 
        ks: int = 2, 
        dropout: float = 0.0,
        ndim: int =1
    ):
        super(TemporalConvNet, self).__init__()
        self.layers = nn.ModuleList()
        num_levels = len(layers)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = c_in if i ==0 else layers[i-1]
            out_channels = layers[i]
            padding = (ks -1) * dilation_size
            self.layers.append(
                TemporalBlock(
                    ni=in_channels,
                    nf=out_channels,
                    ks=ks,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                    ndim=ndim
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class FCN(nn.Module):
    """
    Fully Convolutional Network for sequence or image classification.

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output classes.
        layers (List[int], optional): List specifying the number of channels in each Temporal Block. Defaults to [25]*8.
        ks (int, optional): Kernel size for convolutions. Defaults to 7.
        conv_dropout (float, optional): Dropout rate after convolutional layers. Defaults to 0.0.
        fc_dropout (float, optional): Dropout rate before the fully connected layer. Defaults to 0.0.
        ndim (int, optional): Number of spatial dimensions (1 or 2). Defaults to 1.
    """
    def __init__(
        self, 
        c_in: int, 
        c_out: int, 
        layers: Optional[List[int]] = None, 
        ks: int =7, 
        conv_dropout: float =0.0, 
        fc_dropout: float =0.0, 
        ndim: int =1
    ):
        super(FCN, self).__init__()
        if layers is None:
            layers = [25]*8
        self.tcn = TemporalConvNet(c_in, layers, ks=ks, dropout=conv_dropout, ndim=ndim)
        self.gap = nn.AdaptiveAvgPool1d(1) if ndim ==1 else nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout >0 else None
        self.linear = nn.Linear(layers[-1], c_out)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        return x
