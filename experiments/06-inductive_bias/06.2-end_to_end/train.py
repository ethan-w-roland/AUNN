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

    other_params = []
    for param_type in ["lm","norm","inp_emb","nxt_emb", "cur_emb"]:
        other_params.extend(list(getattr(model, param_type).parameters()))
    other_opt = optim.AdamW(other_params, lr=lr)

    lm_params = list(model.lm.parameters())
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
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                other_opt.step()
                lm_opt.step()

                other_opt.zero_grad(set_to_none=True)
                lm_opt.zero_grad(set_to_none=True)

                pbar.set_description(f"cur_loss={cur_loss.item():.3f}, nxt_loss={nxt_loss.item():.3f}")


        # --- Checkpoint Model ---
        print("Checkpoint Model...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"aunn_{timestamp}.bin"

        checkpoint = {
            "model_state_dict": model.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    print("Starting experimental phase...")
    loader.reset(0)

    # --- Prepare a batched prompt ---
    data = loader.next_batch()  # (B, T)
    B, Tmax = data.shape
    init_len = min(200, Tmax - 1)
    prompt = data[:, :init_len]  # (B, T0)
    print(tokenizer.decode(prompt[0]))  # show first sample only
    print('----')

    # --- LM-only batched autoregression (no-grad) ---
    tokens = prompt.clone()
    with torch.no_grad():
        for _ in range(10):
            x = model.inp_emb(tokens)
            x = model.lm(x)
            x = model.norm(x)
            x = model.nxt_emb(x)
            next_logit = x[:, -1, :]
            next_token = next_logit.argmax(dim=-1).unsqueeze(1)  # (B, 1)
            tokens = torch.cat([tokens, next_token], dim=-1)
            print(tokenizer.decode(tokens[0, -20:]))

    print('-'*30)

    # --- Autoregressive generation via backprop, batched ---
    # (a) Condition memory on the initial prompt
    lm_emb = model.inp_emb(prompt).clone().detach()  # (B, T0, E)
    loss = float('inf')
    beg_pos = 0
    end_pos = prompt.size(1)
    offsets = (torch.arange(B, device=data.device).view(B, 1) * model.config.block_size)
    base_positions = torch.arange(beg_pos, end_pos, device=data.device).unsqueeze(0)
    positions = base_positions + offsets  # (B, T0)
    print(positions[0])
    print(positions[1])
    pos_emb = model.encode(positions)  # (B, T0, E)
    # Temporarily increase mem_opt LR by 10x for initial conditioning
    _orig_mem_lrs = [g['lr'] for g in mem_opt.param_groups]
    for g in mem_opt.param_groups:
        g['lr'] = g['lr'] * 10.0
    while loss > 0.02:
        x = pos_emb
        x = model.mem(x)
        loss = F.mse_loss(x, lm_emb)
        loss.backward()
        nn.utils.clip_grad_norm_(mem_params, 1.0)
        mem_opt.step()
        mem_opt.zero_grad(set_to_none=True)
        print(f"loss: {loss.item():.4f}", end="\n")
    # Restore original mem_opt LR
    for g, lr_val in zip(mem_opt.param_groups, _orig_mem_lrs):
        g['lr'] = lr_val

    # Check conditioning worked (first sample debug)
    with torch.no_grad():
        x = pos_emb
        x = model.mem(x)
        mem_logits = model.cur_emb(x)
        mem_tokens = mem_logits.argmax(dim=-1)
        print(tokenizer.decode(mem_tokens[0]))

    print('='*10)

    # (b) Batched autoregressive loop
    tokens = prompt.clone()  # (B, Tcur)
    steps = 10
    for _ in range(steps):

        offsets = (torch.arange(B, device=data.device).view(B, 1) * model.config.block_size)
        base_cur = torch.arange(beg_pos, end_pos, device=data.device).unsqueeze(0)
        base_nxt = torch.arange(beg_pos, end_pos + 1, device=data.device).unsqueeze(0)
        pos_cur = base_cur + offsets  # (B, Tcur)
        pos_nxt = base_nxt + offsets  # (B, Tcur+1)
        print("pos_cur:", pos_cur.shape, "pos_nxt:", pos_nxt.shape)

        # Teacher pass (no-grad)
        model.eval()
        with torch.no_grad():
            cur_logits, nxt_logits = model(pos_cur)  # (B, Tcur, V)
            cur_tokens = cur_logits.argmax(dim=-1)
            nxt_tokens = nxt_logits.argmax(dim=-1)
            nxt_token = nxt_tokens[:, -1:].contiguous()  # (B, 1)
            tokens = torch.cat([tokens, nxt_token], dim=-1).detach()  # (B, Tcur+1)
            target_log_probs = F.log_softmax(nxt_logits, dim=-1).detach()  # (B, Tcur, V)
            # Debug prints on first sample only
            print("cur_tokens[0]:", tokenizer.decode(cur_tokens[0, -20:]))
            print("nxt_tokens[0]:", tokenizer.decode(nxt_tokens[0, -20:]))
            print("tokens[0]:", tokenizer.decode(tokens[0, -20:]))

        end_pos += 1
        print('~~~')

        # Two-step optimization per iteration (batched)
        cur_loss = float('inf')
        nxt_loss = float('inf')
        model.train()
        # while cur_loss > 0.1 or nxt_loss > 3:
        while cur_loss > 0.01 or nxt_loss > 0.05:

            lm_opt.zero_grad(set_to_none=True)
            mem_opt.zero_grad(set_to_none=True)

            c, n = model(pos_nxt)  # (B, Tcur+1, V)

            # Cross-entropy on previous tokens and the most recent token (50/50 weighting)
            c_prev = c[:, :-1, :].reshape(-1, c.size(-1))   # (B*(Tcur), V)
            t_prev = tokens[:, :-1].reshape(-1)             # (B*(Tcur),)
            ce_prev = F.cross_entropy(c_prev, t_prev)

            c_last = c[:, -1, :]                             # (B, V)
            t_last = tokens[:, -1]                           # (B,)
            ce_last = F.cross_entropy(c_last, t_last)

            cur_loss = 0.5 * ce_prev + 0.5 * ce_last

            n_stub = n[:, :-1, :]
            nxt_loss = F.kl_div(
                F.log_softmax(n_stub, dim=-1),
                target_log_probs,
                reduction='batchmean',
                log_target=True,
            ) / n_stub.size(1)

            loss = cur_loss + nxt_loss
            loss.backward()

            lm_opt.step()
            mem_opt.step()

            # print(f"cur_loss: {cur_loss.item():.4f}")
            print(f"cur_loss: {cur_loss.item():.4f}, nxt_loss: {nxt_loss.item():.4f}")

        # Post-step summary (first sample)
        model.eval()
        with torch.no_grad():
            cur_logits, nxt_logits = model(pos_nxt)
            cur_tokens = cur_logits.argmax(dim=-1)
            nxt_tokens = nxt_logits.argmax(dim=-1)
            print("cur_tokens[0]:", tokenizer.decode(cur_tokens[0, -20:]))
            print("nxt_tokens[0]:", tokenizer.decode(nxt_tokens[0, -20:]))

        print('-'*10)

    exit()

# --- CLI ---
if __name__ == "__main__":

    root_dir = Path(__file__).parent.parent
    data_dir = root_dir.parent.parent / "data"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=data_dir / "simple_stories")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
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
