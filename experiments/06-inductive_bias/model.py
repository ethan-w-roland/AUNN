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

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, E) → out: (B, T, E)
        assert x.ndim == 3 and x.size(1) == 1, "expected (B, 1, E)"
        T = self.block_size
        w = self.weight[:T].to(dtype=x.dtype, device=x.device)      # (T,)
        b = self.bias[:T].to(dtype=x.dtype, device=x.device)        # (T,)
        out = x * w.view(1, T, 1)                                   # (B, 1, E) * (1, T, 1) → (B, T, E)
        out = out + b.view(1, T, 1)                                 # (B, T, E)
        return out


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


class ToeplitzMixer(nn.Module):
    
    def __init__(self, config: Config):
        super().__init__()
        self.inp_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.blocks = nn.ModuleList([MixerBlock(config) for _ in range(config.n_layer)])
        self.norm = nn.RMSNorm(config.embed_dim)
        self.out_emb = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def embed(self, x: torch.Tensor):
        return self.inp_emb(x)

    def get_params(self, embeds: bool = False):
        if embeds:
            yield from self.inp_emb.parameters()
        yield from self.blocks.parameters()
        yield from self.norm.parameters()

    def forward(
        self,
        embeds: torch.Tensor,
        targets: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = embeds
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.out_emb(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)),
                targets.reshape(-1),
            )
        return x, loss


class Projector(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.mixer = Toeplitz(config)
        self.norm = nn.RMSNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(
        self, 
        x: torch.Tensor):
        x = self.mixer.project(x) #(B, 1, E) -> (B, T, E)
        x = x + self.mlp(self.norm(x)) #(B, T, E) -> (B, T, E)
        return x


class AUNN(nn.Module):

    def __init__(self, config: Config):

        super().__init__()
        self.config = config
        self.proj = Projector(config)
        self.mem = ToeplitzMixer(config)
        self.lm = ToeplitzMixer(config)
        
        self.lm.inp_emb.weight = self.lm.out_emb.weight #tie input and output embeds of lm
        self.mem.inp_emb.weight = self.mem.out_emb.weight #tie input and output embeds of mem
        self.mem.out_emb.weight = self.lm.out_emb.weight #tie mem embeds to lm embeds

        self.lm.apply(self._init_weights)
        self.mem.apply(self._init_weights)

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
        position: torch.Tensor, #(B, 1)
        target: torch.Tensor | None = None #(B, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        #absolute positional encoding
        x = self.encode(position) #(B, 1) -> (B, 1, E)

        #project to block size
        x = self.proj(x) #(B, 1, E) -> (B, T, E)

        x, stub_embed = x[:, :-1, :], x[:, -1, :] #(B, T, E) -> (B, T-1, E), (B, 1, E)

        #memory module
        for block in self.mem.blocks:
            x = block(x)

        #lm module
        for block in self.lm.blocks:
            x = block(x)

        #get logits output
        x = self.lm.norm(x)
        logits = self.lm.out_emb(x) #(B, T-1, E) -> (B, T-1, V)

        #get final logit
        pred_logit = logits[:, -1, :] #(B, 1, V)
        stub_logit = self.lm.out_emb(stub_embed) #(B, 1, E) -> (B, 1, V)

        loss = None
        if target is not None:
            loss_pred = F.cross_entropy(pred_logit, target.squeeze(-1))
            loss_stub = F.cross_entropy(stub_logit, target.squeeze(-1))
            loss = loss_pred + loss_stub
            loss = loss / 2
            
        return pred_logit, stub_logit, loss
        
        