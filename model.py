import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Tuple

from tools import LoggingModule, log_io

###########################################################
#################### NORM ##################################
###########################################################
class Norm(LoggingModule):
    def __init__(self, dim, cfg):
        super().__init__()
        self.eps = cfg.eps

        # We start with ones for weight to keep the original scale initially, and zeros for bias.
        self.affine = cfg.norm_affine
        self.bias = cfg.norm_bias
        if cfg.norm_affine:
            self.w = nn.Parameter(torch.ones(cfg.dim))
            if cfg.norm_bias:
                self.b = nn.Parameter(torch.zeros(cfg.dim))
        elif cfg.norm_bias:
            print('cannot have both norm_affine=False and norm_bias=True. Skipping bias')

        # Mapping norm types to their respective methods
        self.type = cfg.norm_type
        self.norm_methods = {
            "RMSNorm": self.RMSNorm,
            "LayerNorm": self.LayerNorm,
            "CosineNorm": self.CosineNorm,
        }
        # Ensure the specified norm type exists, default to RMSNorm if not found
        if self.type not in self.norm_methods:
            print(f'norm type {self.type} not found. defaulting to RMSNorm')
            self.type = "RMSNorm"

    @log_io
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_method = self.norm_methods[self.type]
        x = norm_method(x)

        # Optionally apply the affine transformation
        if self.affine: 
            x = x * self.w
            if self.bias:
                x = x + self.b
            
        return x

    @log_io
    def CosineNorm(self, x):
        return x / torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=self.eps)

    @log_io
    def LayerNorm(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)

    @log_io
    def RMSNorm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

###########################################################
#################### ATTENTION #############################
###########################################################
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class MQSA(LoggingModule): # multi-head self-attention
    def __init__(self, cfg):
        super().__init__()
        self.num_q_heads = cfg.num_q_heads
        self.num_kv_heads = cfg.num_q_heads if cfg.num_kv_heads is None else cfg.num_kv_heads
        assert cfg.num_q_heads % cfg.num_kv_heads == 0, f'num_q_heads must be divisible by num_kv_heads'
        self.head_dim = cfg.dim // cfg.num_q_heads if cfg.head_dim is None else cfg.head_dim
        self.dropout_rate = cfg.dropout_rate

        self.Wq = nn.Linear(cfg.dim, cfg.num_q_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(cfg.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(cfg.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(cfg.num_q_heads * self.head_dim, cfg.dim, bias=False)

        self.cache_k = torch.zeros(
            (cfg.max_batch_size, cfg.max_seq_len, self.num_kv_heads, self.head_dim),
            requires_grad = False).to(cfg.device)
        self.cache_v = torch.zeros(
            (cfg.max_batch_size, cfg.max_seq_len, self.num_kv_heads, self.head_dim),
            requires_grad = False).to(cfg.device)
    
    @log_io
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_len: int = None,
        training: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.Wq(x), self.Wk(x), self.Wv(x)

        xq = xq.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis)

        if cache_len is not None: # if we're performing inference and using kv caching. it'll init at 0
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:batch_size, cache_len : cache_len + seq_len] = xk
            self.cache_v[:batch_size, cache_len : cache_len + seq_len] = xv

            keys = self.cache_k[:batch_size, : cache_len + seq_len]
            values = self.cache_v[:batch_size, : cache_len + seq_len]
        else: 
            # if we're training, do full sequence length
            keys, values = xk, xv
        queries = xq # for sake of keeping the naming scheme consistent

        # adjusts keys and values to match the query heads count.
        if self.num_kv_heads != self.num_q_heads:
            keys, values = self.match_headcount(keys, values) # (batch_sizes, cache_len + seq_len, num_q_heads, head_dim)

        queries = queries.transpose(1, 2)  # (bs, num_q_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (bs, num_q_heads, cache_len + seq_len, head_dim)
        values = values.transpose(1, 2)  # (bs, num_q_heads, cache_len + seq_len, head_dim)
        
        logits = self.attend(queries, keys, training)
        if mask is not None:
            logits = logits + mask  # (bs, num_q_heads, seq_len, cache_len + seq_len)
        scores = self.calc_output(logits, values, training) 
        
        output = self.Wo(scores)
        if training: output = F.dropout(output, self.dropout_rate)
        
        return output
    
    @log_io
    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis.to(xq.device), xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    @log_io
    def reshape_for_broadcast(
        self,
        freqs_cis: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor:
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f'freqs_cis.shape {freqs_cis.shape} != (x.shape[1], x.shape[-1]) {(x.shape[1], x.shape[-1])}'
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @log_io
    def match_headcount(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        keys = torch.repeat_interleave(keys, self.num_q_heads // self.num_kv_heads, dim=2)
        values = torch.repeat_interleave(values, self.num_q_heads // self.num_kv_heads, dim=2)
        return keys, values

    @log_io
    def attend(
        self, 
        queries: 
        torch.Tensor, 
        keys: torch.Tensor, 
        training: bool
    ) -> torch.Tensor:
        return torch.matmul(queries, keys.transpose(2, 3)) * (self.head_dim ** -0.5)
    
    @log_io
    def calc_output(
        self, 
        logits: 
        torch.Tensor, 
        values: torch.Tensor, 
        training: bool
    ) -> torch.Tensor:
        batch_size, _, seq_len, _ = logits.shape
        scores = F.softmax(logits, dim=-1)
        if training: scores = F.dropout(scores, self.dropout_rate)
        output = scores @ values # [batch_size, n_heads, seq_len, head_dim]
        return output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # [batch_size, seq_len, n_heads * head_dim]

###################################################
################# MLP ###############################
###################################################
class MLP(LoggingModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nonlinearity: str = 'GeLU',
        gated: bool = True,
        bias: bool = False, # i Stan Llama and set bias to false
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_dim
        self.dropout_rate = dropout_rate

        # the up, down, and (optional) gate projections
        self.gated = gated
        if gated: self.Wgate = nn.Linear(input_dim, self.hidden_size, bias)
        self.Wup = nn.Linear(input_dim, self.hidden_size, bias)
        self.Wdown = nn.Linear(self.hidden_size, output_dim, bias)

        # Mapping norm types to their respective methods
        self.nonlinearities = {
            "GeLU": nn.GELU(),
            "SiLU": nn.SiLU(),
            "ReLU": nn.ReLU(),
        }
        # Ensure the specified norm type exists, default to GeLU if not found
        if nonlinearity not in self.nonlinearities:
            self.nonlinearity = nn.GeLU
            print(f'nonlinearity {nonlinearity} not found. defaulting to GeLU')
        else:
            self.nonlinearity = self.nonlinearities[nonlinearity]
        
    @log_io
    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if self.gated:
            hidden_neurons = self.nonlinearity(self.Wgate(x)) * self.Wup(x)
        else:
            hidden_neurons = self.nonlinearity(self.Wup(x))
        if training: hidden_neurons = F.dropout(hidden_neurons, self.dropout_rate)
        return self.Wdown(hidden_neurons)

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
################# customGPT ##########################
#####################################################
class customGPT(LoggingModule):
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
        cache_len: int = None,
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
        elif cache_len is not None: # if performing inference
            assert batch_size <= self.max_batch_size # we had to initialize the kv cache to some maximum possible size
            freqs_cis = self.freqs_cis[cache_len : cache_len + seq_len]
            mask = self.mask[:seq_len, :seq_len]
            mask = torch.hstack([torch.zeros((seq_len, cache_len), device=self.device), mask])#.type_as(x)
            training = False
        else:
            assert InputError('both cache_len and target_token_ids cannot be NoneType')
        
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