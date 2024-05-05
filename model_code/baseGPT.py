import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Tuple

from tools import LoggingModule, log_io

from model_code.modules.norms import Norm
from model_code.modules.attentions import MQSA, precompute_freqs_cis
from model_code.modules.feedforward import MLP

###################################################
################# ResidualLayer ####################
###################################################
class ResidualLayer(LoggingModule):
    def __init__(self, cfg):
        super().__init__()
        self.second_norm = cfg.second_resid_norm
        self.dropout_rate = cfg.dropout_rate
        
        self.pre_attn_norm = Norm(cfg.dim, cfg)
        self.attn = MQSA(cfg)
        if self.second_norm: self.post_attn_norm = Norm(cfg.dim, cfg)
        
        self.pre_mlp_norm = Norm(cfg.dim, cfg)
        self.mlp = MLP(
            cfg.dim,
            int(cfg.mlp_hidden_mult * cfg.dim),
            cfg.dim,
            cfg.mlp_nonlinearity,
            cfg.mlp_gated,
            cfg.mlp_bias,
            cfg.dropout_rate
        )
        if self.second_norm: self.post_mlp_norm = Norm(cfg.dim, cfg)

    @log_io
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_len: int = None,
        training = False,
    ) -> torch.Tensor:
        x = x + self.attn_connect(x, freqs_cis, mask, cache_len, training)
        x = x + self.mlp_connect(x, training)
        return x

    @log_io
    def attn_connect(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor, 
        mask: torch.Tensor, 
        cache_len: int, 
        training: bool,
    ) -> torch.Tensor:
        dx = self.attn(
            self.pre_attn_norm(x),
            freqs_cis, 
            mask, 
            cache_len
        )
        if training: F.dropout(dx, self.dropout_rate)
        if self.second_norm: dx = self.post_attn_norm(dx)
        return dx

    @log_io
    def mlp_connect(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        dx = self.mlp(self.pre_mlp_norm(x))
        if training: F.dropout(dx, self.dropout_rate)
        if self.second_norm: dx = self.post_mlp_norm(dx)
        return dx

###################################################
################# base_GPT ##########################
#####################################################
class baseGPT(LoggingModule):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        
        self.num_layers = cfg.num_layers
        self.max_seq_len = cfg.max_seq_len
        self.vocab_len = cfg.vocab_len
        self.max_batch_size = cfg.max_batch_size
        
        self.token_embedder = nn.Embedding(cfg.vocab_len, cfg.dim)
        self.scale = cfg.dim ** 0.5 if cfg.scale_first_resid else 1.0
        
        self.layers = nn.ModuleList(ResidualLayer(cfg) for _ in range(cfg.num_layers))
        self.final_norm = Norm(cfg.dim, cfg)

        freqs_cis = precompute_freqs_cis(
            cfg.head_dim,
            cfg.max_seq_len,
            cfg.theta
        ).to(cfg.device)
        self.register_buffer('freqs_cis', freqs_cis)

        mask = torch.full((cfg.max_seq_len, cfg.max_seq_len), 
                          float("-inf"), 
                          device=cfg.device)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

        self.criterion = nn.CrossEntropyLoss(ignore_index = cfg.vocab_len - 1) # ignore the padding token

    @log_io
    def forward( # this function is specifically for training, not inference
        self, 
        input_token_ids: torch.Tensor, 
        cache_len: int = 0,#None,
        target_token_ids: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor):
        batch_size, seq_len = input_token_ids.shape

        if target_token_ids is not None: # if training
            assert input_token_ids.shape == target_token_ids.shape
            assert seq_len == self.max_seq_len
            mask = self.mask
            freqs_cis = self.freqs_cis
            training = True
            cache_len = None
        else:#if cache_len is not None: # if performing inference
            assert batch_size <= self.max_batch_size # we had to initialize the kv cache to some maximum possible size
            freqs_cis = self.freqs_cis[cache_len : cache_len + seq_len]
            mask = self.mask[:seq_len, :seq_len]
            mask = torch.hstack([torch.zeros((seq_len, cache_len), device=self.device), mask])#.type_as(x)
            training = False
        #else:
            #assert InputError('both cache_len and target_token_ids cannot be NoneType')
        
        # initialize first residual state and run the model
        x = self.token_embedder(input_token_ids) * self.scale # [batch_size, seq_len, dim]
        for layer in self.layers:
            x = layer(
                x, 
                freqs_cis, 
                mask, 
                cache_len,
                training,
            )
        x = self.final_norm(x)
        logits = x @ self.token_embedder.weight.t() # [batch_size, seq_len, vocab_len]

        if training:
            loss = self.criterion(
                logits.view(batch_size * seq_len, self.vocab_len),
                target_token_ids.reshape(batch_size * seq_len)
            )
        else:
            loss = None
            
        return logits, loss