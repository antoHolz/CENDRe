##############################################################################
# Acknowledgments
##############################################################################
"""
This implementation is based on the following works:

1. A decoder-only foundation model for time-series forecasting
   GitHub: https://github.com/google-research/timesfm
   Citation:
   @inproceedings{das2024decoder,
     title={A decoder-only foundation model for time-series forecasting},
     author={Das, Abhimanyu and Kong, Weihao and Sen, Rajat and Zhou, Yichen},
     booktitle={Forty-first International Conference on Machine Learning},
     year={2024}
   }
"""

##############################################################################
# Imports and Setup
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.layers.normalization import Normalize
from modules.layers.patching import Patch


##############################################################################
# ResidualBlock
##############################################################################

class ResidualBlock(nn.Module):
    """
    Residual Block used for input preprocessing in TimesFM.
    
    Args:
        input_dims (int): Dimensionality of the input features.
        hidden_dims (int): Dimensionality of the hidden layer.
        output_dims (int): Dimensionality of the output features.
    
    Attributes:
        hidden_layer (nn.Sequential): Hidden layer with activation.
        output_layer (nn.Linear): Output linear layer.
        residual_layer (nn.Linear): Linear layer for the residual connection.
    """
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(ResidualBlock, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        # Hidden Layer
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.SiLU(),
        )

        # Output Layer
        self.output_layer = nn.Linear(hidden_dims, output_dims)
        
        # Residual Layer
        self.residual_layer = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        hidden = self.hidden_layer(x)
        output = self.output_layer(hidden)
        residual = self.residual_layer(x)
        return output + residual

##############################################################################
# Attention Mechanisms
##############################################################################

class TimesFMAttention(nn.Module):
    """
    Custom attention mechanism tailored for TimesFM.

    Args:
        hidden_size (int): Dimensionality of the input and output embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each head.

    Attributes:
        scaling (nn.Parameter): Per-dimension scaling parameter.
        qkv_proj (nn.Linear): Linear layer to project inputs to queries, keys, and values.
        o_proj (nn.Linear): Output linear layer after attention.
    """
    def __init__(self, hidden_size, num_heads, head_dim):
        super(TimesFMAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim

        assert hidden_size == num_heads * head_dim, "hidden_size must be num_heads * head_dim"

        self.scaling = nn.Parameter(torch.ones(head_dim))

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        # Linear projections
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.hidden_size, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Per-dimension scaling
        q = q * self.scaling

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            scores += mask  # Mask should be additive
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        output = torch.matmul(attn_weights, v)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        output = self.o_proj(output)

        return output

##############################################################################
# Normalization Layers
##############################################################################

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Args:
        d_model (int): Dimensionality of the input and output.
        eps (float): Epsilon value for numerical stability.
    """
    def __init__(self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Compute RMS Norm
        norm_x = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * norm_x

##############################################################################
# FeedForward Layer
##############################################################################

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Neural Network.

    Args:
        d_model (int): The dimensionality of the input and output.
        d_ff (int): The dimensionality of the hidden layer.
        dropout (float): Dropout probability.

    Attributes:
        linear1 (nn.Linear): First linear layer.
        dropout (nn.Dropout): Dropout layer.
        linear2 (nn.Linear): Second linear layer.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Forward pass for the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, d_model].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_length, d_model].
        """
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

##############################################################################
# Transformer Components
##############################################################################

class TransformerDecoderLayer(nn.Module):
    """
    Single layer of a Transformer decoder using TimesFMAttention and RMSNorm.

    Args:
        d_model (int): The dimensionality of the input and output.
        n_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the feed-forward network's hidden layer.
        head_dim (int): Dimensionality of each attention head.
        dropout (float): Dropout probability.

    Attributes:
        attention (TimesFMAttention): Custom attention layer with per-dimension scaling.
        norm1 (RMSNorm): RMS normalization layer before attention.
        feed_forward (FeedForward): Feed-forward sublayer.
        norm2 (RMSNorm): RMS normalization layer before feed-forward network.
    """
    def __init__(self, d_model, n_heads, d_ff, head_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.attention = TimesFMAttention(hidden_size=d_model, num_heads=n_heads, head_dim=head_dim)
        self.norm1 = RMSNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # Feed-forward network with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x

##############################################################################
# TimesFM Model
##############################################################################

class TimesFM(nn.Module):
    """
    Time Series Forecasting Model (TimesFM) updated with TimesFMAttention, RMSNorm, and ResidualBlock.

    Args:
        task (str): The task to perform.
        dec_in (int): Number of input features (channels).
        seq_len (int): Input sequence length.
        pred_len (int): Prediction sequence length (for forecasting).
        d_model (int): Dimensionality of the model (embedding size).
        n_heads (int): Number of attention heads.
        d_layers (int): Number of Transformer decoder layers.
        d_ff (int): Dimensionality of the feed-forward network's hidden layer.
        c_out (int): Number of output features (channels).
        patch_len (int, optional): Length of each patch (if patching is used).
        stride (int, optional): Stride for patching (if patching is used).
        dropout (float): Dropout probability.
        num_class (int, optional): Number of classes (for classification task).
        use_revin (bool): Whether to use RevIN for normalization and denormalization.
        use_patching (bool): Whether to use patching.

    Attributes:
        revin (Normalize): Instance of RevIN normalization (if use_revin is True).
        patch_manager (Patch): Instance of the Patch manager (if use_patching is True).
        input_ff_layer (ResidualBlock): Residual block for input preprocessing.
        pos_embedding (nn.Parameter): Positional embeddings.
        layers (nn.ModuleList): List of Transformer decoder layers.
        output_layer (nn.Linear): Linear layer for producing the final output.
        act (callable): Activation function (used in classification).
        dropout_layer (nn.Dropout): Dropout layer (used in classification).
    """
    def __init__(self, task, dec_in, seq_len, pred_len, d_model, n_heads, d_layers, d_ff, c_out,
                 patch_len=None, stride=None, dropout=0.1, num_class=None, use_revin=True, use_patching=False):
        super(TimesFM, self).__init__()
        self.task = task
        self.dec_in = dec_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_class = num_class
        self.use_revin = use_revin
        self.use_patching = use_patching

        self.head_dim = d_model // n_heads  # Ensure head_dim is an integer
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        # Initialize RevIN for normalization and denormalization
        if self.use_revin:
            self.revin = Normalize(num_features=dec_in, affine=True)

        # Initialize ResidualBlock for input preprocessing
        if self.use_patching:
            assert patch_len is not None, "patch_len must be provided if use_patching is True"
            self.patch_len = patch_len
            input_dims = patch_len * dec_in + patch_len  # Patches + patch masks
            self.input_ff_layer = ResidualBlock(
                input_dims=input_dims,
                hidden_dims=d_ff,
                output_dims=d_model,
            )
            self.patch_manager = Patch(patch_size=patch_len, stride=stride)
            self.pos_embedding = nn.Parameter(torch.zeros(1, patch_len, d_model))
        else:
            input_dims = dec_in + 1  # Features + mask indicator
            self.input_ff_layer = ResidualBlock(
                input_dims=input_dims,
                hidden_dims=d_ff,
                output_dims=d_model,
            )
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, self.head_dim, dropout)
            for _ in range(d_layers)
        ])

        # Output layer
        if self.use_patching:
            output_dim = patch_len * c_out if task != 'classification' else num_class
            self.output_layer = nn.Linear(d_model, output_dim)
        else:
            output_dim = c_out if task != 'classification' else num_class
            self.output_layer = nn.Linear(d_model, output_dim)

        if task == 'classification':
            # <- define CLS here
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.act = F.gelu
            self.dropout_layer = nn.Dropout(dropout)
            # Modify output layer to only process CLS token
            self.output_layer = nn.Linear(d_model, num_class)

    def get_attention_mask(self, mask):
        """
        Creates an attention mask for the input mask.

        Args:
            mask (torch.Tensor): Mask tensor where 1 indicates padding [B, L].

        Returns:
            torch.Tensor: Attention mask [B, 1, 1, L].
        """
        # Convert mask to shape [B, 1, 1, L]
        attention_mask = mask[:, None, None, :]
        attention_mask = attention_mask * -1e9  # Large negative value
        return attention_mask

    def forward_transformer(self, x, mask):
        """
        Processes the input through the Transformer decoder layers.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].
            mask (torch.Tensor): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after Transformer decoder layers.
        """
        # x is already processed by input_ff_layer and positional encoding has been added

        # Process through transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x

    def forecast(self, x_enc):
        """
        Performs forecasting on the input sequence.

        Args:
            x_enc (torch.Tensor): Input sequence tensor [B, L, D].
            x_mark_enc (torch.Tensor): Additional features for encoding (unused).
            x_dec (torch.Tensor): Input sequence for decoding (unused).
            x_mark_dec (torch.Tensor): Additional features for decoding (unused).

        Returns:
            torch.Tensor: Forecasted output tensor [B, pred_len, c_out].
        """
        B, L, D = x_enc.size()

        # Create mask (assuming padding is represented by zeros)
        mask = (torch.abs(x_enc) < 1e-5).all(dim=-1).float()  # [B, L]

        if self.use_revin:
            x_enc = self.revin(x_enc, mode='norm')  # Normalize the input

        if self.use_patching:
            # Create patches from the input sequence
            patches = self.patch_manager(x_enc, mode='patch')  # [B, num_patches, patch_len * dec_in]

            # Create patch masks
            patch_mask = self.patch_manager(mask.unsqueeze(-1), mode='patch')  # [B, num_patches, patch_len, 1]
            patch_mask = patch_mask.squeeze(-1)  # [B, num_patches, patch_len]

            # Concatenate patches with their masks
            patches = torch.cat([patches, patch_mask], dim=-1)  # [B, num_patches, patch_len * dec_in + patch_len]

            # Pass through input_ff_layer
            x = self.input_ff_layer(patches)  # [B, num_patches, d_model]

            # Positional encoding
            x += self.pos_embedding[:, :x.size(1), :]

            # Process through transformer layers
            x = self.forward_transformer(x, mask=None)  # Masking is handled in attention

            # Output projection
            output = self.output_layer(x)

            # Reconstruct the sequence from patches
            output = self.patch_manager(output, mode='depatch', seq_len=self.seq_len + self.pred_len)
        else:
            # Concatenate input with mask indicator
            mask_indicator = mask.unsqueeze(-1)  # [B, L, 1]
            x_input = torch.cat([x_enc, mask_indicator], dim=-1)  # [B, L, dec_in + 1]

            # Pass through input_ff_layer
            x = self.input_ff_layer(x_input)  # [B, L, d_model]

            # Positional encoding
            x += self.pos_embedding[:, :L, :]

            # Process through transformer layers
            attn_mask = self.get_attention_mask(mask)
            x = self.forward_transformer(x, attn_mask)

            # Output projection
            output = self.output_layer(x)

        if self.use_revin:
            output = self.revin(output, mode='denorm')  # Denormalize the output

        return output[:, -self.pred_len:, :]  # Return the last pred_len time steps

    def imputation(self, x_enc, mask):
        """
        Performs imputation on the input sequence.

        Args:
            x_enc (torch.Tensor): Input sequence tensor [B, L, dec_in].
            x_mark_enc (torch.Tensor): Additional features for encoding (unused).
            x_dec (torch.Tensor): Input sequence for decoding (unused).
            x_mark_dec (torch.Tensor): Additional features for decoding (unused).
            mask (torch.Tensor): Mask tensor indicating missing values [B, L].

        Returns:
            torch.Tensor: Imputed output tensor [B, L, c_out].
        """
        B, L, D = x_enc.size()

        if self.use_revin:
            x_enc = self.revin(x_enc, mode='norm')  # Normalize the input

        if self.use_patching:
            # Create patches from the input sequence
            patches = self.patch_manager(x_enc, mode='patch')  # [B, num_patches, patch_len * dec_in]

            # Create patch masks
            patch_mask = self.patch_manager(mask.unsqueeze(-1), mode='patch')  # [B, num_patches, patch_len, 1]
            patch_mask = patch_mask.squeeze(-1)  # [B, num_patches, patch_len]

            # Concatenate patches with their masks
            patches = torch.cat([patches, patch_mask], dim=-1)  # [B, num_patches, patch_len * dec_in + patch_len]

            # Pass through input_ff_layer
            x = self.input_ff_layer(patches)  # [B, num_patches, d_model]

            # Positional encoding
            x += self.pos_embedding[:, :x.size(1), :]

            # Process through transformer layers
            x = self.forward_transformer(x, mask=None)  # Masking is handled in attention

            # Output projection
            output = self.output_layer(x)

            # Reconstruct the sequence from patches
            output = self.patch_manager(output, mode='depatch', seq_len=L)
        else:
            # Concatenate input with mask indicator
            mask_indicator = mask.unsqueeze(-1)  # [B, L, 1]
            x_input = torch.cat([x_enc, mask_indicator], dim=-1)  # [B, L, dec_in + 1]

            # Pass through input_ff_layer
            x = self.input_ff_layer(x_input)  # [B, L, d_model]

            # Positional encoding
            x += self.pos_embedding[:, :L, :]

            # Process through transformer layers
            attn_mask = self.get_attention_mask(mask)
            x = self.forward_transformer(x, attn_mask)

            # Output projection
            output = self.output_layer(x)

        if self.use_revin:
            output = self.revin(output, mode='denorm')  # Denormalize the output

        return output

    def anomaly_detection(self, x_enc):
        """
        Performs anomaly detection on the input sequence.

        Args:
            x_enc (torch.Tensor): Input sequence tensor [B, L, dec_in].

        Returns:
            torch.Tensor: Reconstructed output tensor [B, L, c_out].
        """
        B, L, D = x_enc.size()

        # Create mask (assuming padding is represented by zeros)
        mask = (torch.abs(x_enc) < 1e-5).all(dim=-1).float()  # [B, L]

        if self.use_revin:
            x_enc = self.revin(x_enc, mode='norm')  # Normalize the input

        if self.use_patching:
            # Create patches from the input sequence
            patches = self.patch_manager(x_enc, mode='patch')  # [B, num_patches, patch_len * dec_in]

            # Create patch masks
            patch_mask = self.patch_manager(mask.unsqueeze(-1), mode='patch')  # [B, num_patches, patch_len, 1]
            patch_mask = patch_mask.squeeze(-1)  # [B, num_patches, patch_len]

            # Concatenate patches with their masks
            patches = torch.cat([patches, patch_mask], dim=-1)  # [B, num_patches, patch_len * dec_in + patch_len]

            # Pass through input_ff_layer
            x = self.input_ff_layer(patches)  # [B, num_patches, d_model]

            # Positional encoding
            x += self.pos_embedding[:, :x.size(1), :]

            # Process through transformer layers
            x = self.forward_transformer(x, mask=None)  # Masking is handled in attention

            # Output projection
            output = self.output_layer(x)

            # Reconstruct the sequence from patches
            output = self.patch_manager(output, mode='depatch', seq_len=L)
        else:
            # Concatenate input with mask indicator
            mask_indicator = mask.unsqueeze(-1)  # [B, L, 1]
            x_input = torch.cat([x_enc, mask_indicator], dim=-1)  # [B, L, dec_in + 1]

            # Pass through input_ff_layer
            x = self.input_ff_layer(x_input)  # [B, L, d_model]

            # Positional encoding
            x += self.pos_embedding[:, :L, :]

            # Process through transformer layers
            attn_mask = self.get_attention_mask(mask)
            x = self.forward_transformer(x, attn_mask)

            # Output projection
            output = self.output_layer(x)

        if self.use_revin:
            output = self.revin(output, mode='denorm')  # Denormalize the output

        return output

    def classification(self, x_enc):
        """
        Performs classification on the input sequence.

        Args:
            x_enc (torch.Tensor): Input sequence tensor [B, L, dec_in].
            x_mark_enc (torch.Tensor): Additional features for encoding (unused).

        Returns:
            torch.Tensor: Classification output tensor [B, num_class].
        """
        B, L, D = x_enc.size()

        # Create mask (assuming padding is represented by zeros)
        mask = (torch.abs(x_enc) < 1e-5).all(dim=-1).float()  # [B, L]

        if self.use_revin:
            x_enc = self.revin(x_enc, mode='norm')  # Normalize the input

        if self.use_patching:
            # Create patches from the input sequence
            patches = self.patch_manager(x_enc, mode='patch')  # [B, num_patches, patch_len * dec_in]

            # Create patch masks
            patch_mask = self.patch_manager(mask.unsqueeze(-1), mode='patch')  # [B, num_patches, patch_len, 1]
            patch_mask = patch_mask.squeeze(-1)  # [B, num_patches, patch_len]

            # Concatenate patches with their masks
            patches = torch.cat([patches, patch_mask], dim=-1)  # [B, num_patches, patch_len * dec_in + patch_len]

            # Pass through input_ff_layer
            x = self.input_ff_layer(patches)  # [B, num_patches, d_model]

            # Positional encoding
            x += self.pos_embedding[:, :x.size(1), :]
            cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, d_model]
            x = torch.cat((x, cls_tokens), dim=1)  # [B , L + 1, d_model]
            mask = torch.cat((mask, torch.zeros(B, 1).bool()), dim=1)  # [B, L + 1]

            # Process through transformer layers
            x = self.forward_transformer(x, mask=None)  # Masking is handled in attention
            x = x[:, -1:, :]  # Get CLS token output

            # Output projection
            output = self.output_layer(x)

            # Classification does not require depatching
        else:
            # Concatenate input with mask indicator
            mask_indicator = mask.unsqueeze(-1)  # [B, L, 1]
            x_input = torch.cat([x_enc, mask_indicator], dim=-1)  # [B, L, dec_in + 1]

            # Pass through input_ff_layer
            x = self.input_ff_layer(x_input)  # [B, L, d_model]

            # Positional encoding
            x += self.pos_embedding[:, :L, :]
            cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, d_model]
            x = torch.cat((x, cls_tokens), dim=1)  # [B, L + 1, d_model]
            mask = torch.cat((mask, torch.zeros(B, 1).bool()), dim=1)  # [B, L + 1]

            # Process through transformer layers
            attn_mask = self.get_attention_mask(mask)

            x = self.forward_transformer(x, attn_mask)
            x = x[:, -1:, :]  # Get CLS token output

            # Output projection
            output = self.output_layer(x)

        # Apply activation and dropout
        output = self.act(output)
        output = output.reshape(output.shape[0], -1)

        return output

    def forward(self, x_enc, mask=None):
        """
        Forward method to handle different tasks based on self.task.

        Args:
            x_enc (torch.Tensor): Input sequence tensor [B, L, dec_in].
            x_mark_enc (torch.Tensor): Additional features for encoding.
            x_dec (torch.Tensor): Input sequence for decoding.
            x_mark_dec (torch.Tensor): Additional features for decoding.
            mask (torch.Tensor, optional): Mask tensor (used in imputation).

        Returns:
            torch.Tensor: Output tensor, depends on the task.
        """
        if self.task in ['long_term_forecast', 'short_term_forecast', 'forecasting', 'segmentation']:
            return self.forecast(x_enc)
        elif self.task == 'imputation':
            return self.imputation(x_enc, mask)
        elif self.task == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task == 'classification':
            return self.classification(x_enc)
        return None
