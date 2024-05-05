import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import LoggingModule, log_io

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