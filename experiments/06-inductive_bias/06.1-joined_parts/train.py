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

    lm_params = []
    for param_type in ["lm","norm","out_emb"]:
        lm_params.extend(list(getattr(model, param_type).parameters()))
    lm_opt = optim.AdamW(lm_params, lr=lr)

    mem_params = list(model.mem.parameters())
    mem_opt = optim.AdamW(mem_params, lr=lr*2)

    # --- Resume from checkpoint if requested ---
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
                print("Resume flag set: skipping steps (1)-(3).")
            else:
                print(f"No checkpoints found in {checkpoint_dir}; skipping steps (1)-(3) without loading.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}; skipping steps (1)-(3).")
    
    if not resume:

        print("Training LM...")
        for _ in range(epochs):

            loader.reset(0)
            pbar = tqdm(range(num_batches), ncols=150)

            for _ in pbar:

                data = loader.next_batch()
                x, y = data[:, :-1], data[:, 1:]

                x = model.inp_emb(x)
                x = model.lm(x)
                x = model.norm(x)
                x = model.out_emb(x)

                loss = F.cross_entropy(
                    x.view(-1, x.size(-1)),
                    y.reshape(-1),
                )

                loss.backward()
                nn.utils.clip_grad_norm_(lm_params, 1.0)
                lm_opt.step()
                lm_opt.zero_grad(set_to_none=True)

                pbar.set_description(f"loss={loss.item():.3f}")


        # --- Checkpoint Model ---
        print("Checkpoint Model...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"aunn_{timestamp}.bin"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "mem_opt_state_dict": mem_opt.state_dict(),
            "lm_opt_state_dict": lm_opt.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")


    #--- (3) Inference AUNN ---
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

    print("Starting experimental phase...")
    loader.reset(0)

    #autoregressively inference just the LM
    data = loader.next_batch()
    data = data[0,:20] #first batch, first 20 tokens
    print(tokenizer.decode(data))
    prompt = data[:10] #first 10 tokens
    prompt = prompt.unsqueeze(0) #(1, T)
    with torch.inference_mode():
        for i in range(10):
            #get prediction
            x = model.inp_emb(prompt)
            x = model.lm(x)
            x = model.norm(x)
            x = model.out_emb(x)
            next_logit = x[:, -1, :]
            next_token = next_logit.argmax(dim=-1).unsqueeze(0)
            prompt = torch.cat([prompt, next_token], dim=-1)
            print(tokenizer.decode(prompt.squeeze(0)))

    print('-'*30)

    #autoregressive inference via memory training

    #(a) train first 10 tokens into memory
    prompt = data[:10] #first 10 tokens
    prompt = prompt.unsqueeze(0) #(1, T)
    lm_emb = model.inp_emb(prompt).clone().detach()
    loss = float('inf')
    beg_pos = 0
    end_pos = 10
    positions = torch.arange(beg_pos, end_pos, device=data.device).view(1, -1)
    print(positions)
    pos_emb = model.encode(positions) #(1, T, E)
    while loss > 0.0001:
        x = pos_emb
        x = model.mem(x)
        loss = F.mse_loss(x, lm_emb)
        loss.backward()
        nn.utils.clip_grad_norm_(mem_params, 1.0)
        mem_opt.step()
        mem_opt.zero_grad(set_to_none=True)
        print(f"loss: {loss.item():.4f}", end="\n")

    #check that initial (a) training worked
    with torch.inference_mode():
        x = pos_emb
        x = model.mem(x)
        mem_logits = model.out_emb(x)
        mem_tokens = mem_logits.argmax(dim=-1)
        print(tokenizer.decode(mem_tokens.squeeze(0)))

    print('='*10)

    for _ in range(10):

        positions = torch.arange(beg_pos, end_pos, device=data.device).view(1, -1)
        print(positions)
        pos_emb = model.encode(positions) #(1, T, E)

        x = pos_emb
        x = model.mem(x)

        mem_logits = model.out_emb(x.clone().detach())
        mem_tokens = mem_logits.argmax(dim=-1)
        print(tokenizer.decode(mem_tokens.squeeze(0))) #DEBUG

        x = model.lm(x)
        x = model.norm(x)
        lm_logits = model.out_emb(x)
        lm_tokens = lm_logits.argmax(dim=-1)
        print(tokenizer.decode(lm_tokens.squeeze(0))) #DEBUG

        pred_token = lm_tokens[:, -1:]
        prompt = torch.cat([prompt, pred_token], dim=-1).detach()
        print(tokenizer.decode(prompt.squeeze(0)))

        print('~~~')

        # Train memory to fit the updated prompt; recompute graph each iteration
        lm_emb = model.inp_emb(prompt).clone().detach()
        loss = float('inf')
        end_pos += 1
        positions = torch.arange(beg_pos, end_pos, device=data.device).view(1, -1)
        print(positions)
        pos_emb = model.encode(positions) #(1, T, E)
        while loss > 0.0001:

            x = pos_emb
            x = model.mem(x)
            loss = F.mse_loss(x, lm_emb)
            loss.backward()
            nn.utils.clip_grad_norm_(mem_params, 1.0)
            mem_opt.step()
            mem_opt.zero_grad(set_to_none=True)
            print(f"loss: {loss.item():.4f}", end="\n")

        with torch.inference_mode():
            x = pos_emb
            x = model.mem(x)
            mem_logits = model.out_emb(x)
            mem_tokens = mem_logits.argmax(dim=-1)
            print(tokenizer.decode(mem_tokens.squeeze(0)))

        print('-'*10)

    exit()

# --- CLI ---
if __name__ == "__main__":

    root_dir = Path(__file__).parent.parent
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
