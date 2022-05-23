import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from models.transformer import get_transformer_encoder
from utils.functions import masking_matrix



# Anomaly Transformer
class AnomalyTransformer(nn.Module):
    def __init__(self, linear_embedding, transformer_encoder, mlp_layers, d_embed, max_seq_len, mask_token_rate):
        """
        <class init args>
        linear_embedding : embedding layer to feed data into Transformer encoder
        transformer_encoder : Transformer encoder body
        mlp_layers : MLP layers to return output data
        d_embed : embedding dimension (in Transformer encoder)
        max_seq_len : maximum length of sequence (= window size)
        mask_token_rate : value or range of masking percentage
        """
        super(AnomalyTransformer, self).__init__()
        self.linear_embedding = linear_embedding
        self.transformer_encoder = transformer_encoder
        self.mlp_layers = mlp_layers
        
        _mask_token_rate = mask_token_rate if hasattr(mask_token_rate, '__iter__') else (mask_token_rate, mask_token_rate)
        self.mask_input = False if _mask_token_rate[1] == None else True if _mask_token_rate[1] > 0 else False
        self.max_seq_len = max_seq_len
        self._max_seq_len = max_seq_len + 1
        
        if self.mask_input:
            self.mask_token = nn.Parameter(torch.zeros(1, d_embed))  # learnable mask token
            trunc_normal_(self.mask_token, std=.02)
            
            # random mask token length table
            self.mask_token_table = list(np.random.randint(int(max_seq_len*_mask_token_rate[0]), int(max_seq_len*_mask_token_rate[1]), size=10000))
            
            # table index and length
            self.mask_table_index = 0
            self.mask_table_length = 10000
            
    # Get current mask token length.
    def get_mask_token_length(self, n_batch):
        current_index = self.mask_table_index
        self.mask_table_index += n_batch
        
        lengths = []
        if self.mask_table_index > self.mask_table_length:
            lengths = self.mask_token_table[current_index:]
            self.mask_table_index -= self.mask_table_length
            lengths = lengths + self.mask_token_table[:self.mask_table_index]
        else:
            lengths = self.mask_token_table[current_index:self.mask_table_index]
            if self.mask_table_index == self.mask_table_length:
                self.mask_table_index = 0

        return lengths
    
    def forward(self, x):
        """
        <input info>
        x : (n_batch, n_token, d_data) = (*, max_seq_len, *)
        """
        device = x.device
        n_batch = x.shape[0]
        
        embedded_out = self.linear_embedding(x)  # linear embedding
        
        # Mask input sequence.
        if self.mask_input:
            # Get mask token length and choose first mask index.
            mask_lengths = np.array(self.get_mask_token_length(n_batch))
            first_index = np.random.randint(0, self._max_seq_len-mask_lengths)
            
            # Get mask matrix.
            mask = masking_matrix(n_batch, self.max_seq_len, mask_lengths, first_index, device)
            
            # Mask values.
            mask_tokens = torch.matmul(mask.float(), self.mask_token)
            masked_out = embedded_out.masked_fill(mask, 0) + mask_tokens
        
        else:
            masked_out = embedded_out
            
        # Reconstruct data.
        transformer_out = self.transformer_encoder(masked_out)
        return self.mlp_layers(transformer_out)
    
    
    
# Get Anomaly Transformer.
def get_anomaly_transformer(d_data,
                            d_embed=512,
                            hidden_dim_rate=4.,
                            max_seq_len=512,
                            mask_token_rate=(0.05,0.15),
                            positional_encoding=None,
                            relative_position_embedding=True,
                            transformer_n_layer=6,
                            transformer_n_head=8,
                            dropout=0.1):
    """
    <input info>
    d_data : data input dimension
    d_embed : embedding dimension (in Transformer encoder)
    hidden_dim_rate : hidden layer dimension rate to d_embed
    max_seq_len : maximum length of sequence (= window size)
    mask_token_rate : value or range of masking percentage
    positional_encoding : positional encoding for embedded input; None/Sinusoidal/Absolute
    relative_position_embedding : relative position embedding option
    transformer_n_layer : number of Transformer encoder layers
    transformer_n_head : number of heads in multi-head attention module
    dropout : dropout rate
    """
    hidden_dim = int(hidden_dim_rate * d_embed)
    
    linear_embedding = nn.Linear(d_data, d_embed)
    transformer_encoder = get_transformer_encoder(d_embed=d_embed,
                                                  positional_encoding=positional_encoding,
                                                  relative_position_embedding=relative_position_embedding,
                                                  n_layer=transformer_n_layer,
                                                  n_head=transformer_n_head,
                                                  d_ff=hidden_dim,
                                                  max_seq_len=max_seq_len,
                                                  dropout=dropout)
    mlp_layers = nn.Sequential(nn.Linear(d_embed, hidden_dim),
                               nn.GELU(),
                               nn.Linear(hidden_dim, d_data))
    
    nn.init.xavier_uniform_(linear_embedding.weight)
    nn.init.zeros_(linear_embedding.bias)
    nn.init.xavier_uniform_(mlp_layers[0].weight)
    nn.init.zeros_(mlp_layers[0].bias)
    nn.init.xavier_uniform_(mlp_layers[2].weight)
    nn.init.zeros_(mlp_layers[2].bias)
    
    return AnomalyTransformer(linear_embedding, transformer_encoder, mlp_layers, d_embed, max_seq_len, mask_token_rate)