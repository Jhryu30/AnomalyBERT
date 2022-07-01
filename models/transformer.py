import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from utils.functions import clone_layer



# Main transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, positional_encoding_layer, encoder_layer, n_layer):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = clone_layer(encoder_layer, n_layer)
            
        self.positional_encoding = True if positional_encoding_layer is not None else False
        if self.positional_encoding:
            self.positional_encoding_layer = positional_encoding_layer
        
    def forward(self, x):
        """
        <input>
        x : (n_batch, n_token, d_embed)
        """
        position_vector = None
        if self.positional_encoding:
            out = self.positional_encoding_layer(x)
        else:
            out = x

        for layer in self.encoder_layers:
            out = layer(out)

        return out
    
    
# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, feed_forward_layer, norm_layer, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.norm_layers = clone_layer(norm_layer, 2)
        self.dropout_layer = nn.Dropout(p=dropout)
        
        for p in self.attention_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        for p in self.feed_forward_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        
    def forward(self, x):
        out1 = self.norm_layers[0](x)  # Layer norm first
        out1 = self.attention_layer(out1)
        out1 = self.dropout_layer(out1) + x
        
        out2 = self.norm_layers[1](out1)
        out2 = self.feed_forward_layer(out2)
        return self.dropout_layer(out2) + out1
    
    
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_embed, n_head, max_seq_len=512, relative_position_embedding=True):
        super(MultiHeadAttentionLayer, self).__init__()
        assert d_embed % n_head == 0  # Ckeck if d_model is divisible by n_head.
        
        self.d_embed = d_embed
        self.n_head = n_head
        self.d_k = d_embed // n_head
        self.scale = 1 / np.sqrt(self.d_k)
        
        self.word_fc_layers = clone_layer(nn.Linear(d_embed, d_embed), 3)
        self.output_fc_layer = nn.Linear(d_embed, d_embed)
        
        self.max_seq_len = max_seq_len
        self.relative_position_embedding = relative_position_embedding
        if relative_position_embedding:
            # Table of 1D relative position embedding
            self.relative_position_embedding_table = nn.Parameter(torch.zeros(2*max_seq_len-1, n_head))
            trunc_normal_(self.relative_position_embedding_table, std=.02)
            
            # Set 1D relative position embedding index.
            coords_h = np.arange(max_seq_len)
            coords_w = np.arange(max_seq_len-1, -1, -1)
            coords = coords_h[:, None] + coords_w[None, :]
            self.relative_position_index = coords.flatten()

    def forward(self, x):
        """
        <input>
        x : (n_batch, n_token, d_embed)
        """
        n_batch = x.shape[0]
        device = x.device
        
        # Apply linear layers.
        query = self.word_fc_layers[0](x)
        key = self.word_fc_layers[1](x)
        value = self.word_fc_layers[2](x)
        
        # Split heads.
        query_out = query.view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)
        key_out = key.view(n_batch, -1, self.n_head, self.d_k).contiguous().permute(0, 2, 3, 1)
        value_out = value.view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)
        
        # Compute attention and concatenate matrices.
        scores = torch.matmul(query_out * self.scale, key_out)
        
        # Add relative position embedding
        if self.relative_position_embedding:
            position_embedding = self.relative_position_embedding_table[self.relative_position_index].view(
                self.max_seq_len, self.max_seq_len, -1)
            position_embedding = position_embedding.permute(2, 0, 1).contiguous().unsqueeze(0)
            scores = scores + position_embedding
        
#         if masking_matrix != None:
#             scores = scores + masking_matrix * (-1e9) # Add very small negative number to padding columns.
        probs = F.softmax(scores, dim=-1)
        attention_out = torch.matmul(probs, value_out)
        
        # Convert 4d tensor to proper 3d output tensor.
        attention_out = attention_out.transpose(1, 2).contiguous().view(n_batch, -1, self.d_embed)
            
        return self.output_fc_layer(attention_out)

    
    
class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_embed, d_ff, dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.first_fc_layer = nn.Linear(d_embed, d_ff)
        self.second_fc_layer = nn.Linear(d_ff, d_embed)
        self.activation_layer = nn.GELU()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.first_fc_layer(x)
        out = self.dropout_layer(self.activation_layer(out))
        return self.second_fc_layer(out)
    
    
    
# Sinusoidal positional encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_seq_len=512, dropout=0.1):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        denominators = torch.exp(torch.arange(0, d_embed, 2) * (np.log(0.0001) / d_embed)).unsqueeze(0)
        encoding_matrix = torch.matmul(positions, denominators)
        
        encoding = torch.empty(1, max_seq_len, d_embed)
        encoding[0, :, 0::2] = torch.sin(encoding_matrix)
        encoding[0, :, 1::2] = torch.cos(encoding_matrix[:, :(d_embed//2)])

        self.register_buffer('encoding', encoding)
        
    def forward(self, x):
        """
        <input info>
        x : (n_batch, n_token, d_embed) == (*, max_seq_len, d_embed) (default)
        """
        return self.dropout_layer(x + self.encoding)
    
    
# Absolute position embedding
class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, d_embed, max_seq_len=512, dropout=0.1):
        super(AbsolutePositionEmbedding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_embed))
        trunc_normal_(self.embedding, std=.02)
        
    def forward(self, x):
        """
        <input info>
        x : (n_batch, n_token, d_embed) == (*, max_seq_len, d_embed) (default)
        """
        return self.dropout_layer(x + self.embedding)
    
    

# Get a transformer encoder with its parameters.
def get_transformer_encoder(d_embed=512,
                            positional_encoding=None,
                            relative_position_embedding=True,
                            n_layer=6,
                            n_head=8,
                            d_ff=2048,
                            max_seq_len=512,
                            dropout=0.1):
    
    if positional_encoding == 'Sinusoidal' or positional_encoding == 'sinusoidal' or positional_encoding == 'sin':
        positional_encoding_layer = SinusoidalPositionalEncoding(d_embed, max_seq_len, dropout)
    elif positional_encoding == 'Absolute' or positional_encoding =='absolute' or positional_encoding == 'abs':
        positional_encoding_layer = AbsolutePositionEmbedding(d_embed, max_seq_len, dropout)
    elif positional_encoding == None or positional_encoding == 'None':
        positional_encoding_layer = None
    
    attention_layer = MultiHeadAttentionLayer(d_embed, n_head, max_seq_len, relative_position_embedding)
    feed_forward_layer = PositionWiseFeedForwardLayer(d_embed, d_ff, dropout)
    norm_layer = nn.LayerNorm(d_embed, eps=1e-6)
    encoder_layer = EncoderLayer(attention_layer, feed_forward_layer, norm_layer, dropout)
    
    return TransformerEncoder(positional_encoding_layer, encoder_layer, n_layer)
