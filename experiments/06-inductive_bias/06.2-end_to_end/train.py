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
    for param_type in ["lm","norm","inp_emb","cur_emb","nxt_emb"]:
        lm_params.extend(list(getattr(model, param_type).parameters()))
    lm_opt = optim.AdamW(lm_params, lr=lr)

    mem_params = list(model.mem.parameters())
    mem_opt = optim.AdamW(mem_params, lr=lr)

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
                x_data, y_data = data[:, :-1], data[:, 1:]

                x = model.inp_emb(x_data)
                x = model.lm(x)
                x = model.norm(x)
                cur_logits = model.cur_emb(x)
                nxt_logits = model.nxt_emb(x)

                cur_loss = F.cross_entropy(
                    cur_logits.view(-1, cur_logits.size(-1)),
                    x_data.reshape(-1),
                )

                nxt_loss = F.cross_entropy(
                    nxt_logits.view(-1, nxt_logits.size(-1)),
                    y_data.reshape(-1),
                )

                loss = cur_loss + nxt_loss

                loss.backward()
                nn.utils.clip_grad_norm_(lm_params, 1.0)
                lm_opt.step()
                lm_opt.zero_grad(set_to_none=True)

                pbar.set_description(f"cur_loss={cur_loss.item():.3f}, nxt_loss={nxt_loss.item():.3f}")


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

    print("Starting experimental phase...")
    loader.reset(0)

   
    data = loader.next_batch()
    data = data[0,:] #first example
    prompt = data[:200] #first 200 tokens
    print(tokenizer.decode(prompt))
    prompt = prompt.unsqueeze(0) #(1, T)
    print('----')

    #autoregressively inference just the LM
    inp_tok = prompt
    with torch.inference_mode():
        for i in range(10):
            #get prediction
            x = model.inp_emb(inp_tok)
            x = model.lm(x)
            x = model.norm(x)
            x = model.nxt_emb(x)
            next_logit = x[:, -1, :]
            next_token = next_logit.argmax(dim=-1).unsqueeze(0)
            inp_tok = torch.cat([inp_tok, next_token], dim=-1)
            print(tokenizer.decode(inp_tok[:,-20:].squeeze(0)))

    print('-'*30)

    #autoregressive inference via memory training

    #(a) train first 10 tokens into memory
    lm_emb = model.inp_emb(prompt).clone().detach()
    loss = float('inf')
    beg_pos = 0
    end_pos = len(prompt.squeeze(0))
    positions = torch.arange(beg_pos, end_pos, device=data.device).view(1, -1)
    print(positions)
    pos_emb = model.encode(positions) #(1, T, E)
    while loss > 0.01:
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
        mem_logits = model.cur_emb(x)
        mem_tokens = mem_logits.argmax(dim=-1)
        print(tokenizer.decode(mem_tokens.squeeze(0)))

    print('='*10)

    #autoregressive inference loop
    tokens = prompt
    for _ in range(10):

        pos_cur = torch.arange(beg_pos, end_pos, device=data.device).view(1, -1)
        pos_nxt = torch.arange(beg_pos, end_pos+1, device=data.device).view(1, -1)
        print("pos_cur:", pos_cur)
        print("pos_nxt:", pos_nxt)

        cur_logits, nxt_logits = model(pos_cur)

        cur_tokens = cur_logits.argmax(dim=-1)
        print("cur_tokens:", tokenizer.decode(cur_tokens[:,-20:].squeeze(0)))

        nxt_tokens = nxt_logits.argmax(dim=-1)
        print("nxt_tokens:", tokenizer.decode(nxt_tokens[:,-20:].squeeze(0)))

        nxt_logit = nxt_logits[:, -1, :]
        nxt_token = nxt_logit.argmax(dim=-1).unsqueeze(0)
        tokens = torch.cat([tokens, nxt_token], dim=-1).detach()
        print("tokens:", tokenizer.decode(tokens[:,-20:].squeeze(0)))

        end_pos += 1

        print('~~~')

        # Train memory to fit the updated prompt; recompute graph each iteration
        cur_loss = float('inf')
        nxt_loss = float('inf')
        cur_target = torch.cat([cur_logits, nxt_logit.unsqueeze(1)], dim=1).detach()
        nxt_target = nxt_logits.detach()
        while cur_loss > 1 or nxt_loss > 1:

            # Single forward pass
            mem_opt.zero_grad(set_to_none=True)
            c, _ = model(pos_nxt)
            cur_loss = F.kl_div(
                F.log_softmax(c, dim=-1),
                F.log_softmax(cur_target, dim=-1),
                reduction='batchmean',
                log_target=True,
            )
            cur_loss.backward()
            nn.utils.clip_grad_norm_(mem_params, 1.0)
            mem_opt.step()

            lm_opt.zero_grad(set_to_none=True)
            _, n = model(pos_nxt)
            n_stub = n[:, :-1, :]
            nxt_loss = F.kl_div(
                F.log_softmax(n_stub, dim=-1),
                F.log_softmax(nxt_target, dim=-1),
                reduction='batchmean',
                log_target=True,
            )
            nxt_loss.backward()
            nn.utils.clip_grad_norm_(lm_params, 1.0)
            lm_opt.step()

            print(f"cur_loss: {cur_loss.item():.4f}, nxt_loss: {nxt_loss.item():.4f}", end="\n")

        with torch.inference_mode():
            cur_logits, nxt_logits = model(pos_nxt)
            cur_tokens = cur_logits.argmax(dim=-1)
            print(tokenizer.decode(cur_tokens[:,-20:].squeeze(0)))
            nxt_tokens = nxt_logits.argmax(dim=-1)
            print(tokenizer.decode(nxt_tokens[:,-20:].squeeze(0)))

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
