##############################################################################
# Acknowledgments
##############################################################################
"""
This implementation is based on the following works:

1. Time-Series-Library (TimesNet)
   GitHub: https://github.com/thuml/Time-Series-Library
   Citation:
   @inproceedings{wu2023timesnet,
     title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
     author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
     booktitle={International Conference on Learning Representations},
     year={2023}
   }

2. Open-Source Large Time-Series Models (Timer)
   GitHub: https://github.com/thuml/OpenLTM
   Citation:
   @inproceedings{liutimer,
     title={Timer: Generative Pre-trained Transformers Are Large Time Series Models},
     author={Liu, Yong and Zhang, Haoran and Li, Chenyu and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
     booktitle={Forty-first International Conference on Machine Learning}
   }
"""

##############################################################################
# Imports and Setup
##############################################################################

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Math and numerical operations
import numpy as np
from math import sqrt
import math

##############################################################################
# Embeddings
##############################################################################

class PositionalEmbedding(nn.Module):
    """
    Positional embedding module for adding positional encodings to input sequences.

    Args:
        d_model (int): The dimension of the input embeddings.
        max_len (int, optional): The maximum length of the input sequences. Defaults to 5000.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass of the positional embedding module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The positional encodings corresponding to the input tensor.
        """
        return self.pe[:, :x.size(1)]

class PatchEmbedding(nn.Module):
    """
    PatchEmbedding module for encoding input sequences.

    Args:
        d_model (int): The dimensionality of the output feature vectors.
        patch_len (int): The length of each patch.
        stride (int): The stride for patching.
        padding (int): The padding size for patching.
        dropout (float): The dropout probability.

    Attributes:
        patch_len (int): The length of each patch.
        stride (int): The stride for patching.
        padding_patch_layer (nn.ReplicationPad1d): The padding layer for patching.
        value_embedding (nn.Linear): The linear layer for input encoding.
        position_embedding (PositionalEmbedding): The positional embedding layer.
        dropout (nn.Dropout): The dropout layer.
    """

    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the PatchEmbedding module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size * num_patches, sequence_length, d_model).
            int: The number of variables in the input tensor.
        """
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride).contiguous()
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])).contiguous()
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

##############################################################################
# Attention Mechanisms
##############################################################################

class TriangularCausalMask():
    """
    Class for creating a triangular causal mask used in sequence modeling.
    This mask is typically used to prevent the model from attending to future positions.
    """

    def __init__(self, B, L, device="cpu"):
        """
        Initialize the TriangularCausalMask.

        Args:
            B (int): Batch size.
            L (int): Sequence length.
            device (str, optional): Device to create the mask on. Defaults to "cpu".
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # Create an upper-triangular matrix filled with ones and then shift the diagonal by 1.
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """
        Returns the mask tensor.

        Returns:
            torch.Tensor: The triangular causal mask tensor of shape [B, 1, L, L].
        """
        return self._mask

class FullAttention(nn.Module):
    """
    FullAttention module performs self-attention mechanism on the input queries, keys, and values.

    Args:
        mask_flag (bool, optional): Flag indicating whether to apply attention mask. Default is True.
        factor (int, optional): Scaling factor for attention scores. Default is 5.
        scale (float, optional): Scaling factor for attention weights. If not provided, it is set to 1/sqrt(E),
            where E is the dimension of queries.
        attention_dropout (float, optional): Dropout rate for attention weights. Default is 0.1.
        output_attention (bool, optional): Flag indicating whether to output attention weights. Default is False.
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, n_vars=None, n_tokens=None, tau=None, delta=None):
        """
        Forward pass of the FullAttention module.

        Args:
            queries (torch.Tensor): Input queries of shape (B, L, H, E), where B is the batch size, L is the sequence length,
                H is the number of attention heads, and E is the dimension of queries.
            keys (torch.Tensor): Input keys of shape (B, S, H, D), where S is the sequence length of keys and D is the
                dimension of keys.
            values (torch.Tensor): Input values of shape (B, S, H, D), where S is the sequence length of values and D is the
                dimension of values.
            attn_mask (torch.Tensor, optional): Attention mask of shape (B, 1, L, S) or (B, L, S), where L is the sequence
                length of queries and S is the sequence length of keys/values. If not provided, a causal mask is used by
                default.
            n_vars (int, optional): Number of variables. Defaults to None.
            n_tokens (int, optional): Number of tokens. Defaults to None.
            tau (float, optional): Temperature parameter for attention scores. Default is None.
            delta (float, optional): Offset parameter for attention scores. Default is None.

        Returns:
            tuple: A tuple containing the following elements:
                - V (torch.Tensor): Output values after applying attention, of shape (B, L, H, D), where B is the batch size,
                    L is the sequence length, H is the number of attention heads, and D is the dimension of values.
                - A (torch.Tensor or None): Attention weights, of shape (B, H, L, S), where B is the batch size, H is the
                    number of attention heads, L is the sequence length of queries, and S is the sequence length of keys/values.
                    If output_attention is False, None is returned instead.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class AttentionLayer(nn.Module):
    """
    Attention Layer that combines query, key, and value projections with an attention mechanism.

    Args:
        attention (nn.Module): Attention mechanism to use.
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        d_keys (int, optional): Dimension of keys. Defaults to None.
        d_values (int, optional): Dimension of values. Defaults to None.
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, n_vars=None, n_tokens=None, tau=None, delta=None):
        """
        Forward pass of the AttentionLayer.

        Args:
            queries (torch.Tensor): Query tensor of shape [B, L, D].
            keys (torch.Tensor): Key tensor of shape [B, S, D].
            values (torch.Tensor): Value tensor of shape [B, S, D].
            attn_mask (torch.Tensor): Attention mask tensor.
            n_vars (int, optional): Number of variables. Defaults to None.
            n_tokens (int, optional): Number of tokens. Defaults to None.
            tau (float, optional): Temperature parameter for attention. Defaults to None.
            delta (float, optional): Offset parameter for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after attention.
            torch.Tensor or None: Attention weights if output_attention is True, otherwise None.
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            n_vars=n_vars,
            n_tokens=n_tokens,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

##############################################################################
# Encoder Components
##############################################################################

class EncoderLayer(nn.Module):
    """
    This class represents a single layer of the encoder in a Transformer model.

    Args:
        attention (nn.Module): The attention module used in the encoder layer.
        d_model (int): The dimensionality of the input and output tensors.
        d_ff (int, optional): The dimensionality of the feed-forward layer. Defaults to 4 * d_model.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        activation (str, optional): The activation function to be used. Can be "relu" or "gelu". Defaults to "relu".

    Attributes:
        attention (nn.Module): The attention module used in the encoder layer.
        conv1 (nn.Conv1d): The first convolutional layer.
        conv2 (nn.Conv1d): The second convolutional layer.
        norm1 (nn.LayerNorm): The first layer normalization module.
        norm2 (nn.LayerNorm): The second layer normalization module.
        dropout (nn.Dropout): The dropout module.
        activation (function): The activation function.

    Methods:
        forward(x, attn_mask=None, tau=None, delta=None): Performs a forward pass of the encoder layer.

    Returns:
        tuple: A tuple containing the output tensor and the attention tensor.
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        Performs a forward pass of the encoder layer.

        Args:
            x (torch.Tensor): The input tensor.
            attn_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
            tau (float, optional): The temperature parameter for attention. Defaults to None.
            delta (float, optional): The offset parameter for attention. Defaults to None.

        Returns:
            tuple: A tuple containing the output tensor and the attention tensor.
        """
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class Encoder(nn.Module):
    """
    Encoder module of a Transformer model.

    Args:
        attn_layers (list): List of attention layers.
        conv_layers (list, optional): List of convolutional layers. Defaults to None.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.

    Attributes:
        attn_layers (nn.ModuleList): ModuleList of attention layers.
        conv_layers (nn.ModuleList, optional): ModuleList of convolutional layers. Defaults to None.
        norm (nn.Module, optional): Normalization layer.

    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        Forward pass of the Encoder module.

        Args:
            x (torch.Tensor): Input tensor of shape [B, L, D].
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            tau (float, optional): Temperature parameter for attention. Defaults to None.
            delta (float, optional): Delta parameter for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [B, L, D].
            list: List of attention tensors.

        """
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class TimerLayer(nn.Module):
    """
    TimerLayer module for time-aware Transformer models.

    Args:
        attention (nn.Module): The attention module used in the TimerLayer.
        d_model (int): The dimensionality of the input and output tensors.
        d_ff (int, optional): The dimensionality of the feed-forward layer. Defaults to 4 * d_model.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        activation (str, optional): The activation function to be used. Can be "relu" or "gelu". Defaults to "relu".

    Attributes:
        attention (nn.Module): The attention module used in the TimerLayer.
        conv1 (nn.Conv1d): The first convolutional layer.
        conv2 (nn.Conv1d): The second convolutional layer.
        norm1 (nn.LayerNorm): The first layer normalization module.
        norm2 (nn.LayerNorm): The second layer normalization module.
        dropout (nn.Dropout): The dropout module.
        activation (function): The activation function.

    Methods:
        forward(x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None): Performs a forward pass of the TimerLayer.

    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(TimerLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        """
        Performs a forward pass of the TimerLayer.

        Args:
            x (torch.Tensor): The input tensor.
            n_vars (int): Number of variables.
            n_tokens (int): Number of tokens.
            attn_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
            tau (float, optional): The temperature parameter for attention. Defaults to None.
            delta (float, optional): The offset parameter for attention. Defaults to None.

        Returns:
            tuple: A tuple containing the output tensor and the attention tensor.
        """
        new_x, attn = self.attention(
            x, x, x,
            n_vars=n_vars,
            n_tokens=n_tokens,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class TimerBlock(nn.Module):
    """
    TimerBlock module for stacking multiple TimerLayers.

    Args:
        attn_layers (list): List of attention layers.
        conv_layers (list, optional): List of convolutional layers. Defaults to None.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.

    Attributes:
        attn_layers (nn.ModuleList): ModuleList of attention layers.
        conv_layers (nn.ModuleList, optional): ModuleList of convolutional layers. Defaults to None.
        norm (nn.Module, optional): Normalization layer.

    Methods:
        forward(x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None): Forward pass of the TimerBlock module.

    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(TimerBlock, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(
            conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        """
        Forward pass of the TimerBlock module.

        Args:
            x (torch.Tensor): Input tensor of shape [B, L, D].
            n_vars (int): Number of variables.
            n_tokens (int): Number of tokens.
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            tau (float, optional): Temperature parameter for attention. Defaults to None.
            delta (float, optional): Delta parameter for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [B, L, D].
            list: List of attention tensors.

        """
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, n_vars,
                                           n_tokens, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, n_vars, n_tokens,
                                     attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

##############################################################################
# Timer Model
##############################################################################

class TimerClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout_rate: float = 0.1, hidden_dim: int = None, activation: str = 'relu'):
        """
        Args:
            in_features (int): Expected flattened input dimension,
                i.e., int(seq_len/patch_len) * c_in * d_model.
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout probability.
            hidden_dim (int, optional): If provided, use a two-layer MLP with this hidden dimension.
            activation (str): Activation function to use ('relu' or 'gelu').
        """
        super(TimerClassificationHead, self).__init__()
        self.flatten = nn.Flatten()  # Ensures input is flattened to [B, -1]
        self.dropout = nn.Dropout(dropout_rate)
        
        if hidden_dim is not None:
            self.fc1 = nn.Linear(in_features, hidden_dim)
            if activation.lower() == 'relu':
                self.activation = nn.ReLU()
            elif activation.lower() == 'gelu':
                self.activation = nn.GELU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            self.fc2 = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc = nn.Linear(in_features, num_classes)
            self.activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects input x of shape [B, C * num_patches * d_model].

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits of shape [B, num_classes].
        """
        x = self.flatten(x)    # Ensure x is [B, -1]
        x = self.dropout(x)    # Apply dropout

        if self.activation is not None:
            x = self.fc1(x)
            x = self.activation(x)
            x = self.dropout(x)  # Optionally, add dropout between layers
            x = self.fc2(x)
        else:
            x = self.fc(x)
        return x

class Timer(nn.Module):
    """
    Timer model for various tasks including forecasting, imputation, and anomaly detection.

    Args:
        task (str): Name of the task.
        patch_len (int): Length of the patch.
        d_model (int): Dimension of the model.
        d_ff (int): Dimension of the feed-forward network.
        e_layers (int): Number of encoder layers.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        output_attention (bool): Whether to output attention weights.
        factor (float): Attention factor for the FullAttention mechanism.
        activation (str): Activation function to use (e.g., 'relu', 'gelu').
        c_in (int, optional): Number of input channels. Default is 1.
        c_out (int, optional): Number of output channels. Default is 1.
        last_patch (bool, optional): Whether to use the last patch. Default is False.
    """
    def __init__(self, task, patch_len, d_model, d_ff, e_layers, n_heads, dropout,
                 output_attention, factor, activation, c_in=1, c_out=1, seq_len=None, last_patch=False):
        super(Timer, self).__init__()
        self.task = task
        self.patch_len = patch_len
        self.stride = patch_len  # Using non-overlapping patches
        self.d_model = d_model
        self.d_ff = d_ff
        self.layers = e_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.output_attention = output_attention
        self.last_patch = last_patch 

        # Patch Embedding
        self.enc_embedding = PatchEmbedding(
            d_model=d_model,
            patch_len=patch_len,
            stride=self.stride,
            padding=0,
            dropout=dropout
        )

        # Encoder
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=True,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=output_attention
                        ),
                        d_model=d_model,
                        n_heads=n_heads
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Prediction Head
        self.proj = nn.Linear(d_model, patch_len, bias=True)

        # If the task is classification (or regression), we add an extra classification head.
        if self.task in ['classification', 'regression']:
            # Use a single patch or all the patches
            classifier_in_features = (c_in * d_model) if last_patch else (int(seq_len/patch_len) * c_in * d_model)
            # Create a classifier head – you could also add dropout/activation as needed.
            self.classifier = TimerClassificationHead(
                in_features = classifier_in_features, 
                num_classes = c_out, 
                dropout_rate = 0.1, 
                hidden_dim = d_model, 
                activation = 'gelu')


    def forward(self, x_enc, mask=None):
        """
        Forward pass for the model based on the specified task.

        Args:
            x_enc (torch.Tensor): Input tensor for encoding [B, L, D].
            x_mark_enc (torch.Tensor): Mark tensor for encoding [B, L, D].
            x_dec (torch.Tensor): Input tensor for decoding [B, L, D].
            x_mark_dec (torch.Tensor): Mark tensor for decoding [B, L, D].
            mask (torch.Tensor, optional): Mask tensor for imputation task [B, L, D].

        Returns:
            torch.Tensor or tuple: Model output, depending on the task and whether attention is output.
        """
        if self.task in ['long_term_forecast', 'short_term_forecast', 'forecasting', 'segmentation']:
            return self.forecast(x_enc)
        elif self.task == 'imputation':
            return self.imputation(x_enc, mask)
        elif self.task == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task in ['classification', 'regression']:
            return self.classification(x_enc)
        else:
            raise NotImplementedError(f"Task '{self.task}' is not supported.")

    def forecast(self, x_enc):
        """
        Forecasting method for time-series prediction tasks.

        Args:
            x_enc (torch.Tensor): Input tensor for encoding [B, L, D].
            x_mark_enc (torch.Tensor): Mark tensor for encoding [B, L, D].
            x_dec (torch.Tensor): Input tensor for decoding [B, L, D].
            x_mark_dec (torch.Tensor): Mark tensor for decoding [B, L, D].

        Returns:
            torch.Tensor or tuple: Forecasted output. If `output_attention` is True, also returns attention weights.
        """
        B, L, M = x_enc.shape  # B: Batch size, L: Sequence length, M: Number of variables

        # Standardization
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, unbiased=False, keepdim=True) + 1e-5).detach()
        x_enc = x_enc / stdev

        # Patch Embedding
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # Shape: [B, M, L]
        dec_in, n_vars = self.enc_embedding(x_enc)  # dec_in: [B * M, Num_Patches, d_model]

        # Encoding
        dec_out, attns = self.decoder(dec_in)  # dec_out: [B * M, Num_Patches, d_model]

        # Projection
        dec_out = self.proj(dec_out)  # Shape: [B * M, Num_Patches, patch_len]

        # Reshape and permute to get output in shape [B, T, M]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2).contiguous()  # [B, T, M]

        # Denormalization
        dec_out = dec_out * stdev + means

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out

    def imputation(self, x_enc, mask):
        """
        Imputation method for filling missing data.

        Args:
            x_enc (torch.Tensor): Input tensor with missing values [B, L, D].
            x_mark_enc (torch.Tensor): Mark tensor for encoding [B, L, D].
            x_dec (torch.Tensor): Not used in imputation.
            x_mark_dec (torch.Tensor): Not used in imputation.
            mask (torch.Tensor): Mask tensor where 1 indicates observed data and 0 indicates missing data [B, L, D].

        Returns:
            torch.Tensor: Imputed data tensor [B, L, D].
        """
        # Normalization
        means = torch.sum(x_enc, dim=1, keepdim=True) / torch.sum(mask == 1, dim=1, keepdim=True)
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc ** 2, dim=1, keepdim=True) / torch.sum(mask == 1, dim=1, keepdim=True) + 1e-5).detach()
        x_enc = x_enc / stdev

        # Patch Embedding
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # Shape: [B, D, L]
        dec_in, n_vars = self.enc_embedding(x_enc)

        # Encoding
        dec_out, attns = self.decoder(dec_in)

        # Projection
        dec_out = self.proj(dec_out)  # Shape: [B * D, Num_Patches, patch_len]

        # Reshape and permute to get output in shape [B, L, D]
        dec_out = dec_out.view(-1, n_vars, dec_out.shape[-2] * self.patch_len).transpose(1, 2).contiguous()  # [B, L, D]

        # Denormalization
        dec_out = dec_out * stdev + means

        return dec_out

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection method.

        Args:
            x_enc (torch.Tensor): Input tensor for encoding [B, L, D].

        Returns:
            torch.Tensor: Reconstructed data tensor [B, L, D], useful for anomaly detection.
        """
        # Normalization
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, unbiased=False, keepdim=True) + 1e-5).detach()
        x_enc = x_enc / stdev

        # Patch Embedding
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # Shape: [B, D, L]
        dec_in, n_vars = self.enc_embedding(x_enc)

        # Encoding
        dec_out, attns = self.decoder(dec_in)

        # Projection
        dec_out = self.proj(dec_out)  # Shape: [B * D, Num_Patches, patch_len]

        # Reshape and permute to get output in shape [B, L, D]
        dec_out = dec_out.view(-1, n_vars, dec_out.shape[-2] * self.patch_len).transpose(1, 2).contiguous()  # [B, L, D]

        # Denormalization
        dec_out = dec_out * stdev + means

        return dec_out
    
    def classification(self, x_enc):
        """
        Classification branch:
          - Normalize the input.
          - Apply patch embedding.
          - Process through the encoder.
          - Pool over the patch (and variable) dimension(s) to produce a per-sample representation.
          - Apply a linear classifier.
        Args:
            x_enc (torch.Tensor): Input tensor [B, L, D] (for example, a time series with B samples).
        Returns:
            torch.Tensor: Logits of shape [B, num_classes].
        """
        B, L, C = x_enc.shape  # B: batch size, L: sequence length, C: number of channels (variables)

        # Normalization
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, unbiased=False, keepdim=True) + 1e-5).detach()
        x_enc = x_enc / stdev

        # Patch Embedding
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # Shape: [B, D, L]
        dec_in, n_vars = self.enc_embedding(x_enc) # dec_in: [B * C, num_patches, d_model]

        # Encoding
        dec_out, attns = self.decoder(dec_in)

        # Project
        if self.last_patch:
            dec_out = dec_out[:, -1, :].reshape(B, -1)  # [B * C, num_patches, d_model] -> [B * C, d_model] -> [B, C * d_model]
        else:
            dec_out = dec_out.reshape(B, -1).contiguous()  # [B * C, num_patches, d_model] -> [B, C * num_patches * d_model]

        # Classifier: generate logits.
        logits = self.classifier(dec_out)  # [B, num_classes]
        return logits
