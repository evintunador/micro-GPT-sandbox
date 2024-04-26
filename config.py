from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import time

@dataclass
class ModelConfig:
    # general
    dim: int = 32
    vocab_len: int = None # will be set later according to what tokenizer you choose
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' # can't do MPS bc metal doesn't support complex64 used in RoPE
    dropout_rate = 0.1 # percent of neurons to set to 0 during training as a way of adding randomness & improving generalization

    # Residual Layers
    num_layers: int = 2 # small models should err on the side of many many layers at the expense of attention & mlp sizes
    pre_connect_dropout: bool = False # True performs dropout before the residual connection (only when training)
    second_resid_norm: bool = False # True adds an extra Norm after the attn & MLP, like in Grok. Only recommended for RMSNorm
    
    # MLP
    mlp_hidden_mult: int = 2 # 4 is the most common and 8 is the highest I've seen. Really adds a ton of parameters
    mlp_bias: bool = False # whether to use bias weights. Llama3 does not and I'm partial to their choice
    mlp_nonlinearity: str = 'GeLU' # options are 'GeLU', 'SiLU', and 'ReLU'. Add more options in 'model.py'
    mlp_gated: bool = True # True gives you 50% more MLP parameters to train. Turns GeLU into GeGLU, SiLU into SwiGLU, etc.

    # attention
    num_q_heads: int = 4
    num_kv_heads: int = 1
    head_dim = 16 # most common choices are 32, 64 and especially 128 bc those are what works with FlashAttention
    theta: float = 10_000 # 10_000 is the most common choice. Llama3 uses 50_000
    max_seq_len: int = 128 # 512 is the most my ram can handle

    # normalization
    scale_first_resid: bool = True # whether to multiply the first residual state by sqrt(dim)
    norm_type: str = 'RMSNorm' # options are 'RMSNorm', 'LayerNorm', and 'CosineNorm'. Add more options in 'model.py'
    norm_affine: bool = True # whether to use a linear layer after each norm
    norm_bias: bool = True # whether to add a bias to the linear layer after each norm. doesn't do anything if norm_affine == False
    eps: float = 1e-6 # small constant to prevent division by 0. Not really worth editing

    # inference (kv caching)
    max_batch_size: int = 1 
    # i think batched inference is probably broken rn bc of my shitty tokenizer. might fix in future

@dataclass
class TrainConfig:
    weight_decay: float = 0.02
    batch_size: int = 32
    # name of the folder the model will be saved into
    model_name = f'{time.strftime("%Y-%m-%d|%H-%M")}'
    # total number of batches to run over the course of training
    max_iters: int = 10 # i recommend at least 1_000
    # how often to print out & record an update on how training is going
    eval_interval: int = 5 # doing this too often slows things down hella
    # how often to save a checkpoint
    checkpoint_interval: int = eval_interval # set to None if you don't want checkpoints
    
    ### to visualize the learning rate schedule you define here, see cell 7 of training.ipynb
    
    # if you'd like a flat learning rate, set lr_min = lr_max and ignore the variables below
    lr_max: float = 1e-2
    lr_min: float = 1e-5 
    
    # number of iterations for a linear warmup from lr_min to lr_max
    warmup_iters: int = int(max_iters * 0.1) # if you don't want to use a lr warmup, set = 0
    # number of iterations for a constant learning rate of lr_min at the end of training
    final_flat_iters: int = int(max_iters * 0.2) # if you don't want to use a final flat lr at the end, set = 0
    
    # number of times to bring the learning rate back up from lr_min to lr_max in-between the warmup & final flat
    num_restarts: int = 0 # if you don't want to use warm restarts, set =0
    # relative length of each warm restart compared to the previous.
    T_mult: int = 2 # if you want all to be the same length, set =1. <1 means they get shorter and >1 makes them longer
    # type of annealment to use. Annealment is when the learning rate decreases over the course of training
    anneal_type: str = 'cos' # options: 'cos' and 'lin'
    
    # Calculates T_0 in a way that ensures smooth transition to the final flat learning rate
    def T_0(self): # I DO NOT RECOMMEND EDITING THIS
        middle_section = self.max_iters - self.warmup_iters - self.final_flat_iters
        return middle_section / sum(self.T_mult ** i for i in range(self.num_restarts+1))