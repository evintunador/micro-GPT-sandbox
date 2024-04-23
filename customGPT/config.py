from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class Config:
    # general
    dim: int = 128 
    vocab_len: int = None # will be set later according to what tokenizer you choose
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout_rate = 0.1 # percent of neurons to set to 0 during training as a way of adding randomness & improving generalization

    # Residual Layers
    num_layers: int = 12 # small models should err on the side of many many layers at the expense of attention & mlp sizes
    pre_connect_dropout: bool = False # True performs dropout before the residual connection
    second_resid_norm: bool = False # True adds an extra Norm after the attention & MLP, which Grok does. Only recommended for RMSNorm
    
    # MLP
    mlp_hidden_mult: int = 2 # 4 is the most common and 8 is the highest I've seen. Really adds a ton of parameters
    mlp_bias: bool = False # whether to use bias weights. Llama3 does not and I'm partial to their choice
    mlp_nonlinearity: str = 'GeLU' # options are 'GeLU', 'SiLU', and 'ReLU'. Add more options in 'model.py'
    mlp_gated: bool = True # True gives you 50% more MLP parameters to train. Turns GeLU into GeGLU, SiLU into SwiGLU, etc.

    # attention
    num_q_heads: int = 4 
    num_kv_heads: int = 1 
    assert num_q_heads % num_kv_heads == 0, f'{num_q_heads} must be divisible by {num_kv_heads}'
    head_dim = dim // num_q_heads # most common choices are 32, 64 and especially 128 bc those are what works with FlashAttention
    theta: float = 10_000 # 10_000 is the most common choice. Llama3 uses 50_000
    max_seq_len: int = 512 # 512 is the most my ran can handle

    # normalization
    scale_first_resid: bool = True # whether to multiply the first residual state by sqrt(dim)
    norm_type: str = 'RMSNorm' # options are 'RMSNorm', 'LayerNorm', and 'CosineNorm'. Add more options in 'model.py'
    norm_affine: bool = True # whether to use a linear layer after each norm
    norm_bias: bool = True # whether to add a bias to the linear layer after each norm. doesn't do anything if norm_affine == False
    eps: float = 1e-5 # small constant to prevent division by 0. Not really worth editing

    # inference (kv caching)
    max_batch_size: int = 1 # i think batched inference is probably broken rn bc of my shitty tokenizer. might fix in future