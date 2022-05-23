import numpy as np
import copy

import torch
import torch.nn as nn



# Make clones of a layer.
def clone_layer(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Make masking matrix.
def masking_matrix(n_batch, max_seq_len, mask_lengths, first_indices, device='cpu'):
    """
    <input info>
    n_batch : batch size (number of sequences)
    max_seq_len : maximum sequence length
    mask_lengths : (n_batch,), mask token length
    first_indices : (n_batch,), first indices of mask tokens
    device : device of masking matrix
    """
    mask = torch.zeros(n_batch, max_seq_len, 1, device=device).bool()
    last_indices = first_indices + mask_lengths
    for i, first, last in zip(range(n_batch), first_indices, last_indices):
        mask[i, first:last] = True
    return mask