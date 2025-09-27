"""
@author: ethan-w-roland
@date: 2025-07-20
@desc: AUNN Toeplitz Mixer Model
"""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

torch.set_float32_matmul_precision("medium")

@dataclass
class Config:
    vocab_size: int = 4096  # tiny bc simple stories tokenizer
    encode_dim: int = 2048
    embed_dim: int = 512
    mlp_dim: int = 512 * 4
    block_size: int = 256
    n_layer: int = 2


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.embed_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm = nn.RMSNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        x = x + self.mlp(self.norm(x))
        return x


class AUNN(nn.Module):

    def __init__(self, config: Config):

        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.encode_dim, config.embed_dim)
        self.b1 = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.n1 = nn.RMSNorm(config.embed_dim)
        self.b2 = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.n2 = nn.RMSNorm(config.embed_dim)
        self.inp_emb = nn.Embedding(config.vocab_size, config.embed_dim) #used adhoc, not in forward
        self.out_emb = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.inp_emb.weight = self.out_emb.weight #tie input and output embeds

        self.encode_dim = config.encode_dim
        self.block_size = config.block_size
        self.alpha = -np.log(0.1) / (self.block_size ** 2) #gaussian kernel decay
        
        # Register fixed random Fourier feature parameters as buffers (deterministic across steps)
        M = self.config.encode_dim // 2
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
        Generate RFF-based embeddings using PyTorch for integer positions.
        Parameters:
            positions (torch.Tensor): integer tensor of shape (...), e.g., (B, T)
        Returns:
            torch.Tensor: embeddings of shape (..., K) where K == embed_dim
        """
        K = self.encode_dim
        M = K // 2  # Use K/2 pairs of cos and sin
        device = positions.device
        dtype = self.out_emb.weight.dtype

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
        
        x1 = self.encode(positions) #(B, T) -> (B, T, E)
        x1 = self.proj(x1)
        x1 = self.b1(x1)
        x1 = self.n1(x1)
        x2 = self.b2(x1)
        x2 = self.n2(x2)
        logits = self.out_emb(x2) #(B, T, E) -> (B, T, V)

        return logits, x1, x2