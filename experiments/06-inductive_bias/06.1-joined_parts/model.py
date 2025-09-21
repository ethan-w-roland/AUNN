"""
@author: ethan-w-roland
@date: 2025-07-20
@desc: AUNN Toeplitz Mixer Model
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

torch.set_float32_matmul_precision("medium")

@dataclass
class Config:
    vocab_size: int = 4096  # tiny bc simple stories tokenizer
    embed_dim: int = 512
    mlp_dim: int = 512 * 4
    block_size: int = 512
    n_layer: int = 4


class Toeplitz(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.block_size = config.block_size
        self.weight = nn.Parameter(torch.zeros(config.block_size))
        self.bias = nn.Parameter(torch.zeros(config.block_size))

    def vector_to_matrix(self, v: torch.Tensor, seq_len: int) -> torch.Tensor:
        T = min(seq_len, v.numel())
        M = v.new_zeros((T, T))
        i, j = torch.triu_indices(T, T, offset=0, device=v.device)
        M[i, j] = v[j - i]
        return M

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)  # (B, E, T)
        B, E, T = x.shape
        W = self.vector_to_matrix(self.weight, T).to(x.dtype)  # (T, T)
        out = (x.reshape(B * E, T) @ W).view(B, E, T)
        out = out + self.bias[:T].view(1, 1, T)
        return out.transpose(1, 2)  # (B, T, E)


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


class MixerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.embed_dim)
        self.mixer = Toeplitz(config)
        self.norm2 = nn.RMSNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AUNN(nn.Module):

    def __init__(self, config: Config):

        super().__init__()
        self.config = config
        self.mem = nn.Sequential(*(MixerBlock(config) for _ in range(config.n_layer)))
        self.lm = nn.Sequential(*(MixerBlock(config) for _ in range(config.n_layer)))
        self.inp_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.out_emb = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.norm = nn.RMSNorm(config.embed_dim)
        self.inp_emb.weight = self.out_emb.weight #tie input and output embeds

        self.apply(self._init_weights)

        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params}")

    def _init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor): #absolute positional encoding

        # x: (B, T) integer positions
        B, T = x.shape

        dim = self.config.embed_dim
        assert dim % 2 == 0, "Encoding dimension (dim) must be even."

        # Number of frequencies (half of embedding dimension)
        num_frequencies = dim // 2

        # Frequencies corresponding to powers of two, starting from 2^2
        frequency_powers = torch.arange(
            start=2,
            end=2 + num_frequencies,
            dtype=torch.float32,
            device=x.device,
        )
        frequencies = 2 ** frequency_powers  # (num_frequencies,)

        # Compute angles with broadcasting: (B, T, num_frequencies)
        x_float = x.to(torch.float32)
        angles = (2 * torch.pi * x_float.unsqueeze(-1)) / frequencies  # (B, T, F)

        # Interleave sin and cos across the last dimension to form (B, T, E)
        sin_part = torch.sin(angles)
        cos_part = torch.cos(angles)
        encoding = torch.empty(B, T, dim, device=x.device, dtype=torch.float32)
        encoding[..., 0::2] = sin_part
        encoding[..., 1::2] = cos_part

        return encoding
    
    def forward(
        self,
        positions: torch.Tensor, #(B, T)
    ) -> torch.Tensor:
        
        #absolute positional encoding
        x = self.encode(positions) #(B, T) -> (B, T, E)

        #memory module
        x = self.mem(x)

        #lm module
        x = self.lm(x)

        #get logits output
        x = self.norm(x)
        logits = self.out_emb(x) #(B, T-1, E) -> (B, T-1, V)
            
        return logits