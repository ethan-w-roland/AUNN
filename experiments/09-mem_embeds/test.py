'''
@author: ethan-w-roland
@desc: WIP test script for "true" AUNN architecture
'''

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_float32_matmul_precision("medium")

@dataclass
class Config:
    vocab_size: int = 10  # tiny bc simple stories tokenizer
    encode_dim: int = 4096
    embed_dim: int = 64
    mlp_dim: int = 1024 * 4
    block_size: int = 256
    n_layer: int = 2

class AUNN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        self.in_proj = nn.Linear(config.encode_dim, config.embed_dim)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.norm = nn.RMSNorm(config.embed_dim)
        self.out_cur_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_nxt_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_emb = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        #RFF stuff
        self.encode_dim = config.encode_dim
        self.block_size = config.block_size
        self.alpha = -np.log(0.1) / (self.block_size ** 2) #gaussian kernel decay        
        M = config.encode_dim // 2
        rff_w = torch.randn(M) * (2.0 * self.alpha) ** 0.5
        rff_b = torch.rand(M) * (2.0 * torch.pi)
        self.register_buffer("rff_w", rff_w)
        self.register_buffer("rff_b", rff_b)
        
        self.apply(self._init_weights)

        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params}")

    def _init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

    def encode(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Generate RFF-based embeddings for integer positions.
        Parameters:
            positions (torch.Tensor): integer tensor of shape (...), e.g., (B, T)
        Returns:
            torch.Tensor: embeddings of shape (..., K) where K == embed_dim
        """
        K = self.encode_dim
        device = positions.device
        dtype = self.proj.weight.dtype

        # Use fixed RFF params registered as buffers for deterministic encoding
        w = self.rff_w.to(device=device, dtype=dtype)
        b = self.rff_b.to(device=device, dtype=dtype)

        # Compute embedding using cos(w * n + b) and sin(w * n + b)
        pos = positions.to(dtype)
        z = pos[..., None] * w + b  # (..., M)
        cos_z = torch.cos(z)
        sin_z = torch.sin(z)
        embedding = torch.cat([cos_z, sin_z], dim=-1)  # (..., K)
        embedding = embedding * (2.0 / float(K)) ** 0.5
        return embedding
    
    def forward(
        self,
        positions: torch.Tensor, #(B, T)
    ) -> torch.Tensor:

        x = self.encode(positions) #(B, T) -> (B, T, E)
        x = self.proj(x)
        x = self.blocks(x)
        x = self.norm(x)
        a = self.out_cur_proj(x)
        b = self.out_nxt_proj(x)

        return a, b