'''
@author: ethan-w-roland
@desc: WIP test script for "true" AUNN architecture
'''

from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.set_float32_matmul_precision("medium")

@dataclass
class Config:
    vocab_size: int = 10
    encode_dim: int = 4096
    embed_dim: int = 256
    mlp_dim: int = 256 * 4
    n_layer: int = 4


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
        self.in_proj = nn.Linear(config.encode_dim, config.embed_dim)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.norm = nn.RMSNorm(config.embed_dim)
        self.out_emb_cur = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.out_emb_nxt = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        #RFF stuff
        self.encode_dim = config.encode_dim
        self.alpha = -np.log(0.1) / (10 ** 2) #gaussian kernel decay        
        M = config.encode_dim // 2
        rff_w = torch.randn(M, dtype=torch.float32) * (2.0 * self.alpha) ** 0.5
        rff_b = torch.rand(M, dtype=torch.float32) * (2.0 * torch.pi)
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
            torch.Tensor: embeddings of shape (..., E) where E == embed_dim
        """

        E = self.encode_dim
        w = self.rff_w
        b = self.rff_b

        # Compute embedding using cos(w * n + b) and sin(w * n + b)
        pos = positions.to(torch.float32)
        z = pos[..., None] * w + b
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
        x = self.blocks(x)
        x = self.norm(x)
        a = self.out_emb_cur(x)
        b = self.out_emb_nxt(x)

        return a, b

#------- Define Vocab -------

vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
tok2id = {tok: i for i, tok in enumerate(vocab)}
id2tok = {i: tok for i, tok in enumerate(vocab)}

def tokenize(text: str) -> torch.Tensor:
    return torch.tensor([tok2id[tok] for tok in text])

def detokenize(tokens: torch.Tensor) -> str:
    return ''.join([id2tok[tok] for tok in tokens])

device = "cuda"
batch_size = 1024*12
num_batches = 1

data = torch.arange(0, batch_size*num_batches + 1, device=device)
data = data % len(vocab)
print('data[-10:]', data[-10:])

#------- Define Model & Opt -------

assert torch.cuda.is_available()
config = Config(vocab_size=len(vocab))
model = AUNN(config).to(device)
opt = optim.AdamW(model.parameters(), lr=1e-3)

num_epochs = 100_000
pbar = tqdm(range(num_epochs), ncols=100)
for epoch in pbar:

    for batch_idx in range(num_batches):

        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch = data[start_idx:end_idx+1]

        x, y = batch[:-1], batch[1:]
        x = x.reshape(1, batch_size)
        y = y.reshape(1, batch_size)

        positions = torch.arange(start_idx, end_idx, device=device)
        positions = positions.view(1, batch_size)

        a, b = model(positions) #(1, T, V), (1, T, V)

        # --- Cross entropy loss ---
        ce_loss_a = F.cross_entropy(a.reshape(-1, a.size(-1)), x.reshape(-1))
        ce_loss_b = F.cross_entropy(b.reshape(-1, b.size(-1)), y.reshape(-1))

        # --- Causal loss ---
        a_stub = a[:, 1:] #(1, T-1, V)
        b_stub = b[:, :-1] #(1, T-1, V)
        kl_loss = F.kl_div(
            F.log_softmax(b_stub / 0.1, dim=-1),  # Sharpen B (becomes certain)
            F.softmax(a_stub, dim=-1),            # A's distribution (target)
            reduction='batchmean'
        )

        # --- Total loss ---
        loss = ce_loss_a + ce_loss_b + kl_loss * 10
            
        loss.backward()

        opt.step()
        opt.zero_grad(set_to_none=True)
        pbar.set_description(f"loss_a={ce_loss_a.item():.4f} loss_b={ce_loss_b.item():.4f} loss_kl={kl_loss.item():.4f}")

# begin experimental testing

end_idx = len(data) - 1
beg_idx = end_idx - 10
positions = torch.arange(beg_idx, end_idx, device=device)
positions = positions.view(1, -1)
print(positions)

a, b = model(positions)
a_argmax = a.argmax(dim=-1)
b_argmax = b.argmax(dim=-1)

print(a_argmax)
print(b_argmax)

print('---')

beg_idx = end_idx
end_idx = beg_idx + 10
positions = torch.arange(beg_idx, end_idx, device=device)
positions = positions.view(1, -1)
print(positions)

a, b = model(positions)
a_argmax = a.argmax(dim=-1)
b_argmax = b.argmax(dim=-1)

print(a_argmax)
print(b_argmax)