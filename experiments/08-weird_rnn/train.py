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
import torch.optim as optim
from model import wRNN, AUNN, Config as ModelConfig
from dataloader import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import torch.nn.functional as F


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
    model = wRNN(config).to(device)
    model = torch.compile(model)

    loader = DataLoader(
        filename=f"{data_dir}/train.bin",
        B=batch_size,
        T=block_size,
        device=device,
        pin_memory=True,
    )

    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    if resume:

        candidates = sorted(
            checkpoint_dir.glob("*.bin"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        assert candidates, f"No checkpoints found in {checkpoint_dir}"
        latest_ckpt = candidates[0]
        print(f"Loading checkpoint from {latest_ckpt}")
        state = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

    else:

        num_batches = len(loader)

        # --- Optimizers ---
        opt = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

        model.train()
        loader.reset(0)
        pbar = tqdm(range(num_batches), ncols=100)

        for _ in pbar:

            data = loader.next_batch()
            x, y = data[:, :-1], data[:, 1:]

            _, loss = model(x, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pbar.set_description(f"loss={loss.item():.3f}")

        # --- Checkpoint Model ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"{timestamp}.bin"

        checkpoint = {
            "model_state_dict": model.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    aunn = AUNN(config)
    aunn.wRNN = model
    aunn = aunn.to(device)
    aunn.train()
    loader.reset(0)

    #------------- normal LLM inference -------------
    
    print('start normal llm inference...')
    prompt = "one day a girl named"
    prompt = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False), device=device)
    gen_len = 10
    first_token = prompt[:1]
    x = aunn.wRNN.inp_emb(first_token) #hidden state
    preds = [first_token]
    with torch.inference_mode():
        for i in range(len(prompt) + gen_len):
            x = x + aunn.wRNN.blocks(x)
            x = aunn.wRNN.norm(x)
            if i < len(prompt) - 1:
                next_token = prompt[i+1].unsqueeze(0)
                print('[prompt] next_token:', tokenizer.decode(next_token.squeeze(0)))
                preds.append(next_token)
                next_emb = aunn.wRNN.inp_emb(next_token)
                x = x + next_emb
            else:
                logits = aunn.wRNN.out_emb(x)
                next_token = logits.argmax(dim=-1)
                print('[gen] next_token:', tokenizer.decode(next_token.squeeze(0)))
                preds.append(next_token)
                next_emb = aunn.wRNN.inp_emb(next_token)
                x = x + next_emb
            
    print(tokenizer.decode(torch.cat(preds, dim=0).squeeze(0)))

    #------------- memory conditioning -------------

    print('start memory conditioning...')

    mem_params = []
    for param_type in ["proj","mem"]:
        mem_params.extend(list(getattr(aunn, param_type).parameters()))
    mem_opt = optim.AdamW(mem_params, lr=1e-3)

    prompt = "one day a girl named"
    prompt = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False), device=device)
    gen_len = 10
    first_token = prompt[:1]
    x = aunn.wRNN.inp_emb(first_token).unsqueeze(0) #first hidden state
    x = x.clone().detach()
    preds = [first_token]
    for i in range(len(prompt) + gen_len):

        position = torch.arange(i, i+1, device=device)
        position = position.unsqueeze(0)

        loss = float('inf')
        while loss > 0.001:

            logits, x1, x2 = aunn(position)
            loss = F.mse_loss(x1, x)
            loss.backward()
            nn.utils.clip_grad_norm_(mem_params, 1.0)
            mem_opt.step()
            mem_opt.zero_grad(set_to_none=True)
            print(f"@{i} loss: {loss.item():.4f}", end="\r")

        if i < len(prompt) - 1:
            next_token = prompt[i+1].ravel()
            print('[prompt] next_token:', tokenizer.decode(next_token))
            preds.append(next_token)
            next_emb = aunn.wRNN.inp_emb(next_token)
            x = x2 + next_emb
            x = x.clone().detach()
        else:
            next_token = logits.argmax(dim=-1).ravel()
            print('[gen] next_token:', tokenizer.decode(next_token))
            preds.append(next_token)
            next_emb = aunn.wRNN.inp_emb(next_token)
            x = x2 + next_emb
            x = x.clone().detach()
            
    print(tokenizer.decode(torch.cat(preds, dim=0).squeeze(0)))

    #---------- werid parallel variant -------------
    
    print('start parallel variant...')
    
    #reinit aunn
    aunn = AUNN(config)
    aunn.wRNN = model
    aunn = aunn.to(device)
    aunn.train()

    #TODO ran out of time!!

# --- CLI ---
if __name__ == "__main__":

    root_dir = Path(__file__).parent
    data_dir = root_dir.parent.parent / "data"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=data_dir / "simple_stories")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    run(
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        resume=args.resume,
    )