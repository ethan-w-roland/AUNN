"""
@author: ethan-w-roland
@date: 2025-07-20
@desc: Toeplitz Mixer Training Script
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import AUNN, Config as ModelConfig
from dataloader import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import numpy as np


def get_vocab_size(token_dir: str) -> int:
    with open(os.path.join(token_dir, "metadata.json")) as f:
        return json.load(f)["all"]["vocab_size"]


def run(
    data_dir: str,
    block_size: int,
    batch_size: int,
    resume: bool = False,
) -> None:

    assert torch.cuda.is_available()
    device = "cuda"

    # --- Model & Data ---

    tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")
    vocab_size = get_vocab_size(data_dir)
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size)
    model = AUNN(config).to(device)

    loader = DataLoader(
        filename=f"{data_dir}/train.bin",
        B=batch_size,
        T=block_size,
        device=device,
        pin_memory=True,
    )

    num_batches = 50
    tokens_per_batch = block_size * batch_size

    # --- Optimizers ---
    # Train different parameter groups with different learning rates
    b1_params = list(model.proj.parameters()) + list(model.b1.parameters()) + list(model.n1.parameters())
    b2_params = list(model.b2.parameters()) + list(model.n2.parameters()) + list(model.out_emb.parameters())
    b1_opt = optim.AdamW(b1_params, lr=1e-4, betas=(0.9, 0.95))
    b2_opt = optim.AdamW(b2_params, lr=1e-5, betas=(0.9, 0.95))

    # --- Resume from checkpoint if requested ---
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    if resume:

        candidates = sorted(
            checkpoint_dir.glob("aunn_*.bin"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        assert candidates, f"No checkpoints found in {checkpoint_dir}"
        latest_ckpt = candidates[0]
        print(f"Loading checkpoint from {latest_ckpt}")
        state = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        # if "b1_opt_state_dict" in state:
        #     b1_opt.load_state_dict(state["b1_opt_state_dict"]) 
        # if "b2_opt_state_dict" in state:
            # b2_opt.load_state_dict(state["b2_opt_state_dict"])

    else:

        tok_losses = []
        csl_losses = []
        model.train()
        loader.reset(0)
        pbar = tqdm(range(num_batches), ncols=100)

        for idx in pbar:

            data = loader.next_batch()
            positions = torch.arange(idx * tokens_per_batch, (idx + 1) * tokens_per_batch, device=device)
            positions = positions.reshape(batch_size, -1)
            # print(positions)

            for iter in range(1000):

                logits, x1, x2 = model(positions)

                token_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    data.reshape(-1),
                )

                csl_x1 = x1[:, 1:]
                csl_x2 = x2[:, :-1]
                causal_loss = F.mse_loss(csl_x1, csl_x2)

                loss = token_loss + causal_loss * 100

                tok_losses.append(token_loss.item())
                csl_losses.append(causal_loss.item())

                b1_opt.zero_grad(set_to_none=True)
                b2_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                b1_opt.step()
                b2_opt.step()

                pbar.set_description(f"iter={iter:03d}, tok_loss={token_loss.item():.3f}, csl_loss={causal_loss.item():.8f}")

        # --- Checkpoint Model & Optimizers ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"aunn_{timestamp}.bin"

        tok_losses = np.array(tok_losses)
        csl_losses = np.array(csl_losses)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "b1_opt_state_dict": b1_opt.state_dict(),
            "b2_opt_state_dict": b2_opt.state_dict(),
            "tok_losses": tok_losses,
            "csl_losses": csl_losses,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    #-------------

    print("Starting experimental phase...")

    prompt = "[EOS] the cat sat on the mat and"
    prompt = torch.tensor(tokenizer.encode(prompt), device=device)

    start_batch_idx = num_batches
    start_abs_idx = start_batch_idx * tokens_per_batch
    positions = torch.arange(start_abs_idx, start_abs_idx + prompt.size(0), device=device)
    print(positions)

    #train prompt into net
    print("Training prompt into net...")
    pbar = tqdm(range(1000), ncols=100)
    for _ in pbar:

        logits, x1, x2 = model(positions)

        token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            prompt.reshape(-1),
        )

        csl_x1 = x1[:, 1:]
        csl_x2 = x2[:, :-1]
        causal_loss = F.mse_loss(csl_x1, csl_x2)

        loss = token_loss + causal_loss * 100

        b1_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        b1_opt.step() #only lower portion of net

        pbar.set_description(f"tok_loss={token_loss.item():.3f}, csl_loss={causal_loss.item():.8f}")

    #now, autoregressively generate
    print("Autoregressively generating...")
    idx = positions[-1:].view(1,1)
    logits, x1, x2 = model(idx)
    next_token = logits.argmax(dim=-1)
    print(tokenizer.decode(next_token.squeeze(0)))




# --- CLI ---
if __name__ == "__main__":

    root_dir = Path(__file__).parent
    data_dir = root_dir.parent.parent / "data"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=data_dir / "simple_stories")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--resume", action="store_true", help="Load latest checkpoint before training")
    args = ap.parse_args()

    run(
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        resume=args.resume,
    )
