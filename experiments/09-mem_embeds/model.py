"""
@author: ethan-w-roland
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_float32_matmul_precision("medium")

@dataclass
class Config:
    vocab_size: int = 4096  # tiny bc simple stories tokenizer
    encode_dim: int = 4096
    embed_dim: int = 1024
    mlp_dim: int = 1024 * 4
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


class wRNN(nn.Module): #weird RNN

    def __init__(self, config: Config):

        super().__init__()
        self.config = config
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.norm = nn.RMSNorm(config.embed_dim)
        self.cur_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.nxt_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.inp_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.out_emb = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.inp_emb.weight = self.out_emb.weight #tie input and output embeds

        self.apply(self._init_weights)

        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params}")

    def _init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        tokens: torch.Tensor, #(B, T)
        targets: torch.Tensor, #(B, T)
    ) -> torch.Tensor:
        
        embeds = self.inp_emb(tokens) #(B, T) -> (B, T, E)
        B, T, E = embeds.shape

        dtype = embeds.dtype
        inp_emb = torch.zeros(B, E, device=tokens.device, dtype=dtype)
        logits = []
        ce_loss = 0.0
        aux_loss = 0.0

        for idx in range(T):

            tok_emb = embeds[:, idx, :] #(B, E)
            inp_emb = tok_emb + inp_emb #(B, E)

            x = inp_emb + self.blocks(inp_emb) #(B, E)
            x = self.norm(x) #(B, E)

            cur_emb = self.cur_proj(x) #(B, E)
            cur_logit = self.out_emb(cur_emb) #(B, V)

            nxt_emb = self.nxt_proj(x) #(B, E)
            nxt_logit = self.out_emb(nxt_emb) #(B, V)

            ce_loss_cur = F.cross_entropy(cur_logit, tokens[:, idx])
            ce_loss_nxt = F.cross_entropy(nxt_logit, targets[:, idx])
            ce_loss += (ce_loss_cur + ce_loss_nxt) / 2
            aux_loss += F.mse_loss(cur_emb, inp_emb)

            inp_emb = nxt_emb
            logits.append(nxt_logit)

        ce_loss = ce_loss / T
        aux_loss = aux_loss / T

        logits = torch.stack(logits, dim=1) #(B, T, V)
        return logits, ce_loss, aux_loss

class AUNN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        self.inp_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.in_proj = nn.Linear(config.encode_dim, config.embed_dim)
        self.blocks_a = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.blocks_b = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.norm = nn.RMSNorm(config.embed_dim)
        self.cur_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.nxt_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_emb = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.inp_emb.weight = self.out_emb.weight #tie input and output embeds

        #Random Fourier Features
        self.encode_dim = config.encode_dim
        self.block_size = config.block_size
        self.alpha = -np.log(0.1) / (self.block_size ** 2) #gaussian kernel decay factor      
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

    def get_fast_weights(self):
        yield from self.in_proj.parameters()
        yield from self.blocks_a.parameters()

    def get_slow_weights(self):
        yield from self.blocks_b.parameters()
        yield from self.norm.parameters()
        yield from self.cur_proj.parameters()
        yield from self.nxt_proj.parameters()
        yield from self.out_emb.parameters()

    def encode(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Generate RFF-based embeddings for integer positions.
        Parameters:
            positions (torch.Tensor): integer tensor of shape (...), e.g., (B, T)
        Returns:
            torch.Tensor: embeddings of shape (..., E) where E == embed_dim
        """

        E = self.encode_dim
        device = positions.device
        dtype = self.in_proj.weight.dtype
        w = self.rff_w.to(device=device, dtype=dtype)
        b = self.rff_b.to(device=device, dtype=dtype)

        # Compute embedding using cos(w * n + b) and sin(w * n + b)
        pos = positions.to(dtype)
        z = pos[..., None] * w + b  # (..., E)
        cos_z = torch.cos(z)
        sin_z = torch.sin(z)
        embedding = torch.cat([cos_z, sin_z], dim=-1)  # (..., E)
        embedding = embedding * (2.0 / float(E)) ** 0.5
        return embedding
    
    def forward(
        self,
        positions: torch.Tensor, #(B, T)
    ) -> torch.Tensor:

        x = self.encode(positions) #(B, T) -> (B, T, E)
        x = self.in_proj(x)
        h = self.blocks_a(x)
        x = h + self.blocks_b(h)
        x = self.norm(x)

        cur = self.cur_proj(x)
        nxt = self.nxt_proj(x)

        aux_loss = F.mse_loss(cur, h)

        return cur, nxt, aux_loss