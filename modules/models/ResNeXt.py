import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNeXtBlock(nn.Module):
    """
    A ResNeXt Block with grouped convolutions, supporting 1D and 2D inputs.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels before expansion.
        stride (int): Stride for the first convolutional layer.
        expansion (int): Expansion factor for the block.
        cardinality (int): Number of groups for the grouped convolution.
        base_width (int): Base width for the grouped convolution.
        ndim (int): Number of spatial dimensions (1 or 2).
        norm_layer (callable, optional): Normalization layer constructor. Defaults to nn.BatchNorm1d or nn.BatchNorm2d based on ndim.
        activation (callable, optional): Activation function constructor. Defaults to nn.ReLU.
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion=4, 
                 cardinality=32, base_width=4, ndim=2, norm_layer=None, 
                 activation=nn.ReLU):
        super(ResNeXtBlock, self).__init__()
        self.expansion = expansion
        self.cardinality = cardinality
        self.base_width = base_width
        self.ndim = ndim

        # Select the appropriate convolution and normalization layers based on ndim
        Conv = nn.Conv1d if ndim == 1 else nn.Conv2d
        BatchNorm = norm_layer if norm_layer else (nn.BatchNorm1d if ndim == 1 else nn.BatchNorm2d)
        
        # Calculate the width of each group
        D = int(math.floor(out_channels * (base_width / 64)) * cardinality)
        
        # Define the convolutional layers
        self.conv1 = Conv(in_channels, D, kernel_size=1, stride=1, bias=False)
        self.bn1 = BatchNorm(D)
        
        self.conv2 = Conv(D, D, kernel_size=3, stride=stride, padding=1, 
                          groups=cardinality, bias=False)
        self.bn2 = BatchNorm(D)
        
        self.conv3 = Conv(D, out_channels * expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = BatchNorm(out_channels * expansion)
        
        # Define downsampling layer if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels * expansion:
            self.downsample = nn.Sequential(
                Conv(in_channels, out_channels * expansion, kernel_size=1, 
                     stride=stride, bias=False),
                BatchNorm(out_channels * expansion)
            )
        
        # Activation function
        self.relu = activation(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

import math

class ResNeXt(nn.Module):
    """
    A ResNeXt implementation supporting both 1D and 2D inputs.
    
    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output classes.
        expansion (int): Expansion factor for the residual blocks.
        layers (list): A list where each element specifies the number of blocks in that layer.
        cardinality (int): Number of groups for the grouped convolution.
        base_width (int): Base width for the grouped convolution.
        initial_channels (int, optional): Number of channels after the initial convolution. Defaults to 64.
        ndim (int, optional): Number of spatial dimensions (1 or 2). Defaults to 2.
        norm_layer (callable, optional): Normalization layer constructor. Defaults to nn.BatchNorm2d or nn.BatchNorm1d based on ndim.
        activation (callable, optional): Activation function constructor. Defaults to nn.ReLU.
    """
    def __init__(self, c_in, c_out, expansion, layers, cardinality=32, base_width=4, 
                 initial_channels=64, ndim=2, norm_layer=None, activation=nn.ReLU):
        super(ResNeXt, self).__init__()
        self.expansion = expansion
        self.cardinality = cardinality
        self.base_width = base_width
        self.in_channels = initial_channels
        self.ndim = ndim

        # Select the appropriate convolution and normalization layers based on ndim
        Conv = nn.Conv1d if ndim == 1 else nn.Conv2d
        MaxPool = nn.MaxPool1d if ndim == 1 else nn.MaxPool2d
        AdaptiveAvgPool = nn.AdaptiveAvgPool1d if ndim == 1 else nn.AdaptiveAvgPool2d
        BatchNorm = norm_layer if norm_layer else (nn.BatchNorm1d if ndim == 1 else nn.BatchNorm2d)
        
        # Initial convolutional layer
        kernel_size = 7 if ndim == 2 else 3
        stride = 2 if ndim == 2 else 1
        padding = 3 if ndim == 2 else 1
        self.conv1 = Conv(c_in, initial_channels, kernel_size=kernel_size, 
                          stride=stride, padding=padding, bias=False)
        self.bn1 = BatchNorm(initial_channels)
        self.relu = activation(inplace=True)
        
        # Define max pooling layer
        pool_kernel = 3 if ndim == 2 else 2
        pool_stride = 2 if ndim == 2 else 2
        pool_padding = 1 if ndim == 2 else 0
        self.maxpool = MaxPool(kernel_size=pool_kernel, stride=pool_stride, 
                               padding=pool_padding)
        
        # Create ResNeXt layers dynamically
        self.layers = nn.ModuleList()
        for idx, num_blocks in enumerate(layers):
            # Typically, channels double with each subsequent layer
            out_channels = initial_channels * (2 ** idx)
            stride = 1 if idx == 0 else 2
            layer = self._make_layer(out_channels, num_blocks, stride=stride, 
                                     expansion=expansion, 
                                     cardinality=cardinality, 
                                     base_width=base_width, 
                                     norm_layer=norm_layer, 
                                     activation=activation)
            self.layers.append(layer)
            self.in_channels = out_channels * expansion
        
        # Adaptive average pooling and fully connected layer
        pool_output_size = 1 if ndim ==1 else (1, 1)
        self.avgpool = AdaptiveAvgPool(pool_output_size)
        self.fc = nn.Linear(self.in_channels, c_out)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels, blocks, stride, expansion, 
                   cardinality, base_width, norm_layer, activation):
        """
        Creates a sequential layer composed of ResNeXt blocks.
        
        Args:
            out_channels (int): Number of output channels for the blocks.
            blocks (int): Number of ResNeXt blocks in this layer.
            stride (int): Stride for the first block in the layer.
            expansion (int): Expansion factor for the blocks.
            cardinality (int): Number of groups for the grouped convolution.
            base_width (int): Base width for the grouped convolution.
            norm_layer (callable): Normalization layer constructor.
            activation (callable): Activation function constructor.
        
        Returns:
            nn.Sequential: A sequential container of ResNeXt blocks.
        """
        layers = []
        layers.append(ResNeXtBlock(self.in_channels, out_channels, stride=stride, 
                                   expansion=expansion, cardinality=cardinality, 
                                   base_width=base_width, ndim=self.ndim, 
                                   norm_layer=norm_layer, activation=activation))
        self.in_channels = out_channels * expansion
        for _ in range(1, blocks):
            layers.append(ResNeXtBlock(self.in_channels, out_channels, stride=1, 
                                       expansion=expansion, cardinality=cardinality, 
                                       base_width=base_width, ndim=self.ndim, 
                                       norm_layer=norm_layer, activation=activation))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initializes the weights of the network modules.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Defines the forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Example Usage
if __name__ == "__main__":
    # Example for 2D ResNeXt (Images)
    print("=== 2D ResNeXt Example ===")
    resnext2d = ResNeXt(
        c_in=3,
        c_out=1000,
        expansion=4,  # 4 for Bottleneck blocks
        layers=[3, 4, 6, 3],  # ResNeXt-50
        cardinality=32,
        base_width=4,
        initial_channels=64,
        ndim=2,
        norm_layer=None,  # Defaults to BatchNorm2d
        activation=nn.ReLU
    )
    print(resnext2d)
    x2d = torch.randn(1, 3, 224, 224)
    output2d = resnext2d(x2d)
    print(f"2D Output shape: {output2d.shape}\n")  # Expected: [1, 1000]
    
    # Example for 1D ResNeXt (Time Series)
    print("=== 1D ResNeXt Example ===")
    resnext1d = ResNeXt(
        c_in=1,
        c_out=10,
        expansion=4,  # 4 for Bottleneck blocks
        layers=[3, 4, 6, 3],  # Similar depth
        cardinality=32,
        base_width=4,
        initial_channels=64,
        ndim=1,
        norm_layer=None,  # Defaults to BatchNorm1d
        activation=nn.ReLU
    )
    print(resnext1d)
    x1d = torch.randn(1, 1, 1000)  # Example time series with length 1000
    output1d = resnext1d(x1d)
    print(f"1D Output shape: {output1d.shape}")  # Expected: [1, 10]
