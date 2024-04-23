from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class Config:
    # general
    dim: int = 128 
    n_layers: int = 8
    vocab_size: int = None
    norm_eps: float = 1e-5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout_rate = 0.1

    # MLP
    mlp_hidden_mult: int = 4
    mlp_bias: bool = False
    gated: bool = True # true gives you 50% more parameters to train
    nonlinearity: str = 'GeLU' # options are 'GeLU', 'SiLU', and 'ReLU'. All are actually GLU

    # attention
    n_q_heads: int = 4 
    n_kv_heads: int = 1 
    rope_theta: float = 10000 
    max_seq_len: int = 512

    # normalization
    norm_type: str = 'RMSNorm' # options are 'RMSNorm', 'LayerNorm', and 'CosineNorm'
    norm_affine: bool = True
    norm_bias: bool = False # doesn't do anything if norm_affine is False

    # inference (kv caching)
    max_batch_size: int = 32
    memory_saver_div: int = 8
    def context_chunk(self):
        return self.max_seq_len // self.memory_saver_div