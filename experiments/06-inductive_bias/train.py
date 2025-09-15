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


def get_vocab_size(token_dir: str) -> int:
    with open(os.path.join(token_dir, "metadata.json")) as f:
        return json.load(f)["all"]["vocab_size"]


def run(
    data_dir: str,
    block_size: int,
    batch_size: int,
    lr: float,
    epochs: int,
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

    num_batches = len(loader)

    # --- Optimizers ---

    lm_params = list(model.lm.get_params(embeds=True))
    lm_opt = optim.AdamW(lm_params, lr=lr)

    mem_params = list(model.mem.get_params(embeds=True))
    mem_opt = optim.AdamW(mem_params, lr=lr*2)

    proj_params = list(model.proj.parameters())
    proj_opt = optim.AdamW(proj_params, lr=lr)

    # --- (0) Resume from checkpoint if requested ---
    checkpoint_dir = root_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if resume:
        try:
            candidates = sorted(
                checkpoint_dir.glob("aunn_*.bin"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                latest_ckpt = candidates[0]
                print(f"Loading checkpoint from {latest_ckpt}")
                state = torch.load(latest_ckpt, map_location=device)
                model.load_state_dict(state["model_state_dict"]) 
                if "mem_opt_state_dict" in state:
                    mem_opt.load_state_dict(state["mem_opt_state_dict"]) 
                if "lm_opt_state_dict" in state:
                    lm_opt.load_state_dict(state["lm_opt_state_dict"]) 
                if "proj_opt_state_dict" in state:
                    proj_opt.load_state_dict(state["proj_opt_state_dict"]) 
                print("Resume flag set: skipping steps (1)-(3).")
            else:
                print(f"No checkpoints found in {checkpoint_dir}; skipping steps (1)-(3) without loading.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}; skipping steps (1)-(3).")
    
    if not resume:
        # --- (1) Train Projector ---
        # position -> past positions

        print("Training Projector...")
        pbar = tqdm(range(num_batches), ncols=150)
        max_pos = num_batches * block_size * batch_size
        for _ in pbar:

            #randomly sample ints between 0 and max_pos
            positions = torch.randint(block_size, max_pos, (batch_size, 1)).to(device) #(B,1)

            #targets is size (B,T) where targets(b,n) is equal to the value of positions(b,1) and targets(b,n-i) is equal to positions(b,1) - i for i in range(T)
            offsets = torch.arange(-(block_size - 1), 1, device=device).view(1, block_size)  # (1, T)
            targets = positions + offsets  # (B, T)

            positions = model.encode(positions) #(B, 1, E)
            targets = model.encode(targets) #(B, T, E)
            
            pred = model.proj.forward(positions) #(B, T, E)
            loss = F.mse_loss(pred, targets)

            # Calculate average cosine similarity between pred and targets
            # pred: (B, T, E), targets: (B, T, E)
            pred_flat = pred.reshape(-1, pred.size(-1))
            targets_flat = targets.reshape(-1, targets.size(-1))
            cos_sim = F.cosine_similarity(pred_flat, targets_flat, dim=-1)
            avg_cos_sim = cos_sim.mean().item()

            loss.backward()
            nn.utils.clip_grad_norm_(proj_params, 1.0)
            proj_opt.step()
            proj_opt.zero_grad(set_to_none=True)

            pbar.set_description(f"loss={loss.item():.4f}, cos={avg_cos_sim:.4f}")

        # --- (2) Train Memory ---
        # position -> token values

        print("Training Memory...")
        for _ in range(epochs):

            loader.reset(0)
            pbar = tqdm(range(num_batches-10, num_batches), ncols=150)

            for batch_num in pbar:

                data = loader.next_batch()
                # Compute the starting position for this batch
                start_pos = batch_num * block_size
                # Create a (B, T) tensor where each row is a sequence of positions
                B, T = data.shape
                positions = (
                    torch.arange(start_pos, start_pos + B * T, device=data.device)
                    .view(B, T)
                )
                positions = model.encode(positions) #(B, T, E)

                N = 0
                while True:
                    N += 1

                    loss = model.mem.forward(embeds=positions, targets=data)[1]

                    loss.backward()
                    nn.utils.clip_grad_norm_(mem_params, 1.0)
                    mem_opt.step()
                    mem_opt.zero_grad(set_to_none=True)

                    pbar.set_description(f"loss={loss.item():.3f}")

                    if loss.item() < 0.1:
                        print(f"\nN={N}")
                        break


        # --- (3) Train LM ---

        print("Training LM...")
        for _ in range(epochs):

            loader.reset(0)
            pbar = tqdm(range(num_batches), ncols=150)

            for _ in pbar:

                data = loader.next_batch()
                x, y = data[:, :-1], data[:, 1:]
                x = model.lm.inp_emb(x)

                loss = model.lm.forward(embeds=x, targets=y)[1]

                loss.backward()
                nn.utils.clip_grad_norm_(lm_params, 1.0)
                lm_opt.step()
                lm_opt.zero_grad(set_to_none=True)

                pbar.set_description(f"loss={loss.item():.3f}")       


    # INSERT_YOUR_CODE

    # --- (3.5) Checkpoint Model ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"aunn_{timestamp}.bin"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mem_opt_state_dict": mem_opt.state_dict(),
        "lm_opt_state_dict": lm_opt.state_dict(),
        "proj_opt_state_dict": proj_opt.state_dict(),
        "epochs": epochs,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")


    #--- (4) Inference AUNN ---
    # a. Given a token prompt, condition memory module on the prompt (10-100ish steps)
    # b. input positions of token prompt to aunn.forward(), position -> embeds -> logits
    # c. get logits for next token
    # d. train memory module on the next token
    # e. construct new prompt with next token
    # f. repeat b-e until desired generation length is reached

    # now, once we prove the above works, the question becomes can we continue pretraining the LM using only position embeds?
    # the thing we'd really like to prove is "local continual learning" i.e. creating conherent responses to the prompt
    # where the completion clearly indicates a response conditioned on information not currently in context 
    # the hope is to solve long context coherence by treating LLM weights as a kind of additional hidden state

# --- CLI ---
if __name__ == "__main__":

    root_dir = Path(__file__).parent
    data_dir = root_dir.parent.parent / "data"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=data_dir / "simple_stories")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="Load latest checkpoint and skip steps 1-3")
    args = ap.parse_args()

    run(
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        resume=args.resume,
    )
