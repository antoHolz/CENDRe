import math
from typing import Any, List, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class _DenseLayer(nn.Module):
    """
    Single layer within a Dense Block, supporting 1D and 2D inputs.
    
    Args:
        num_input_features (int): Number of input channels.
        growth_rate (int): Growth rate for the DenseNet.
        bn_size (int): Multiplicative factor for the bottleneck layers.
        drop_rate (float): Dropout rate after each Dense Layer.
        ndim (int): Number of spatial dimensions (1 or 2).
        norm_layer (callable, optional): Normalization layer constructor.
        activation (callable, optional): Activation function constructor.
    """
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        ndim: int,
        norm_layer: Optional[callable] = None,
        activation: Optional[callable] = nn.ReLU
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.ndim = ndim
        Conv = nn.Conv1d if ndim == 1 else nn.Conv2d
        BatchNorm = norm_layer if norm_layer else (nn.BatchNorm1d if ndim ==1 else nn.BatchNorm2d)
        
        self.norm1 = BatchNorm(num_input_features)
        self.relu1 = activation(inplace=True)
        self.conv1 = Conv(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        
        self.norm2 = BatchNorm(bn_size * growth_rate)
        self.relu2 = activation(inplace=True)
        self.conv2 = Conv(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1 if ndim ==2 else 1,
            bias=False
        )
        
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = torch.cat([x, out], dim=1)
        return out

class _DenseBlock(nn.Module):
    """
    Dense Block consisting of multiple Dense Layers.
    
    Args:
        num_layers (int): Number of layers in the Dense Block.
        num_input_features (int): Number of input channels.
        bn_size (int): Multiplicative factor for the bottleneck layers.
        growth_rate (int): Growth rate for the DenseNet.
        drop_rate (float): Dropout rate after each Dense Layer.
        ndim (int): Number of spatial dimensions (1 or 2).
        norm_layer (callable, optional): Normalization layer constructor.
        activation (callable, optional): Activation function constructor.
    """
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        ndim: int,
        norm_layer: Optional[callable] = None,
        activation: Optional[callable] = nn.ReLU
    ) -> None:
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
                ndim,
                norm_layer,
                activation
            )
            layers.append(layer)
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class _Transition(nn.Module):
    """
    Transition layer between Dense Blocks.
    
    Args:
        num_input_features (int): Number of input channels.
        num_output_features (int): Number of output channels.
        ndim (int): Number of spatial dimensions (1 or 2).
        norm_layer (callable, optional): Normalization layer constructor.
        activation (callable, optional): Activation function constructor.
    """
    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        ndim: int,
        norm_layer: Optional[callable] = None,
        activation: Optional[callable] = nn.ReLU
    ) -> None:
        super(_Transition, self).__init__()
        Conv = nn.Conv1d if ndim ==1 else nn.Conv2d
        BatchNorm = norm_layer if norm_layer else (nn.BatchNorm1d if ndim ==1 else nn.BatchNorm2d)
        
        self.norm = BatchNorm(num_input_features)
        self.relu = activation(inplace=True)
        self.conv = Conv(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2) if ndim ==1 else nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.relu(self.norm(x)))
        out = self.pool(out)
        return out

class DenseNet(nn.Module):
    """
    DenseNet implementation supporting both 1D and 2D inputs.
    
    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output classes.
        growth_rate (int): Growth rate for the DenseNet.
        block_config (List[int]): List indicating the number of layers in each Dense Block.
        compression (float): Compression factor for transition layers.
        num_init_features (int, optional): Number of features after the initial convolution. Defaults to 64.
        ndim (int, optional): Number of spatial dimensions (1 or 2). Defaults to 2.
        norm_layer (callable, optional): Normalization layer constructor. Defaults to BatchNorm1d or BatchNorm2d based on ndim.
        activation (callable, optional): Activation function constructor. Defaults to ReLU.
        bn_size (int, optional): Multiplicative factor for bottleneck layers within Dense Blocks. Defaults to 4.
        drop_rate (float, optional): Dropout rate after each Dense Layer. Defaults to 0.
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        growth_rate: int,
        block_config: List[int],
        compression: float,
        num_init_features: int = 64,
        ndim: int = 2,
        norm_layer: Optional[callable] = None,
        activation: Optional[callable] = nn.ReLU,
        bn_size: int =4,
        drop_rate: float =0.0
    ) -> None:
        super(DenseNet, self).__init__()
        self.ndim = ndim
        Conv = nn.Conv1d if ndim ==1 else nn.Conv2d
        BatchNorm = norm_layer if norm_layer else (nn.BatchNorm1d if ndim ==1 else nn.BatchNorm2d)
        AvgPool = nn.AvgPool1d if ndim ==1 else nn.AvgPool2d
        AdaptiveAvgPool = nn.AdaptiveAvgPool1d if ndim ==1 else nn.AdaptiveAvgPool2d

        # Initial convolution
        kernel_size = 7 if ndim ==2 else 3
        stride = 2 if ndim ==2 else 1
        padding = 3 if ndim ==2 else 1
        self.features = nn.Sequential(OrderedDict([
            ('conv0', Conv(c_in, num_init_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
            ('norm0', BatchNorm(num_init_features)),
            ('relu0', activation(inplace=True)),
            ('pool0', AvgPool(kernel_size=3, stride=2, padding=1) if ndim ==2 else nn.Identity())
        ]))

        # Each DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                ndim=ndim,
                norm_layer=norm_layer,
                activation=activation
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) -1:
                trans_out_features = int(math.floor(num_features * compression))
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=trans_out_features,
                    ndim=ndim,
                    norm_layer=norm_layer,
                    activation=activation
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = trans_out_features

        # Final BatchNorm
        self.features.add_module('norm_final', BatchNorm(num_features))
        self.relu = activation(inplace=True)

        # Linear layer
        self.classifier = nn.Linear(num_features, c_out)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = self.relu(features)
        if self.ndim ==2:
            out = F.adaptive_avg_pool2d(out, (1,1))
        elif self.ndim ==1:
            out = F.adaptive_avg_pool1d(out, 1)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
