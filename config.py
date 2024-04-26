from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class ModelConfig:
    # general
    dim: int = 64
    vocab_len: int = None # will be set later according to what tokenizer you choose
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' # can't do MPS because pytorch metal doesn't support complex values used in RoPE
    dropout_rate = 0.1 # percent of neurons to set to 0 during training as a way of adding randomness & improving generalization

    # Residual Layers
    num_layers: int = 12 # small models should err on the side of many many layers at the expense of attention & mlp sizes
    pre_connect_dropout: bool = False # True performs dropout before the residual connection
    second_resid_norm: bool = True # True adds an extra Norm after the attention & MLP, which Grok does. Only recommended for RMSNorm
    
    # MLP
    mlp_hidden_mult: int = 4 # 4 is the most common and 8 is the highest I've seen. Really adds a ton of parameters
    mlp_bias: bool = True # whether to use bias weights. Llama3 does not and I'm partial to their choice
    mlp_nonlinearity: str = 'GeLU' # options are 'GeLU', 'SiLU', and 'ReLU'. Add more options in 'model.py'
    mlp_gated: bool = True # True gives you 50% more MLP parameters to train. Turns GeLU into GeGLU, SiLU into SwiGLU, etc.

    # attention
    num_q_heads: int = 4
    num_kv_heads: int = 1
    head_dim = 32 # most common choices are 32, 64 and especially 128 bc those are what works with FlashAttention
    theta: float = 10_000 # 10_000 is the most common choice. Llama3 uses 50_000
    max_seq_len: int = 512 # 512 is the most my ram can handle

    # normalization
    scale_first_resid: bool = True # whether to multiply the first residual state by sqrt(dim)
    norm_type: str = 'RMSNorm' # options are 'RMSNorm', 'LayerNorm', and 'CosineNorm'. Add more options in 'model.py'
    norm_affine: bool = True # whether to use a linear layer after each norm
    norm_bias: bool = True # whether to add a bias to the linear layer after each norm. doesn't do anything if norm_affine == False
    eps: float = 1e-6 # small constant to prevent division by 0. Not really worth editing

    # inference (kv caching)
    max_batch_size: int = 1 # i think batched inference is probably broken rn bc of my shitty tokenizer. might fix in future

@dataclass
class TrainConfig:
    # optimizer
    weight_decay = 0.02
    
    # learning rate annealing
    lr_max = 1e-2
    lr_min = 1e-5 # if you'd like a flat learning rate, set lr_max = lr_min and ignore the variables below
    max_iters = 1000 # total number of batches to run over the course of training
    warmup_iters = int(max_iters * 0.05) # if you don't want to use a lr warmup, set = 0
    final_flat_iters = int(max_iters * 0.2) # if you don't want to use a final flat lr at the end, set = 0
    num_restarts = 3 # if you don't want to use warm restarts, set =0
    T_mult = 2 # if you want your warm restarts to all be the same length, set =1
    anneal_type = 'cos' # type of annealment to use. options: 'cos' and 'lin'
    
    # Calculates T_0 in a way that ensures smooth transition to the final flat learning rate
    def T_0(self):
        middle_section = self.max_iters - self.warmup_iters - self.final_flat_iters
        return middle_section / sum(self.T_mult ** i for i in range(self.num_restarts+1))