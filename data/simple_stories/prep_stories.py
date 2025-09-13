#!/usr/bin/env python
"""
Prepare token-level binary shards and write a metadata.json file.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging

logging.set_verbosity(40)

def memmap_write(
    fname: Path,
    arr: List[List[int]],
    dtype: np.dtype = np.uint16,
) -> None:
    """
    Write array data to a memory-mapped file.

    Args:
        fname: Path to output file
        arr_iter: Iterable of arrays to write
        dtype: NumPy data type for the memory-mapped array
    """
    
    total = sum(len(a) for a in arr)
    mmap = np.memmap(fname, dtype=dtype, mode="w+", shape=(total,))
    idx = 0
    for a in tqdm(arr, desc="writing", total=len(arr)):
        mmap[idx : idx + len(a)] = a
        idx += len(a)
    mmap.flush()
    return


def prep(
    num_proc: int,
    tokenizer: AutoTokenizer,
    max_length: int,
    drop_over_limit: bool,
) -> tuple[Dict[str, Dataset], Dict[str, Dict[str, Any]]]:

    dset_name = "SimpleStories/SimpleStories"
    ds = load_dataset(dset_name, split="train")
    ds = ds.select_columns(["story"])

    print("Dataset columns:", ds.column_names)

    def tok_fn(ex: Dict[str, Any]) -> Dict[str, Any]:

        ids = tokenizer.encode(ex["story"], add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)

        # If not dropping, enforce truncation
        # and ensure the last token is EOS after truncation
        if not drop_over_limit and len(ids) > max_length:
            ids = ids[:max_length]
            ids[-1] = tokenizer.eos_token_id

        return {"ids": ids, "len": len(ids)}

    ds = ds.map(tok_fn, num_proc=num_proc)

    # If dropping is enabled, remove stories longer than max_length
    if drop_over_limit:
        ds = ds.filter(lambda ex: ex["len"] <= max_length, num_proc=num_proc)

    #split into train and test
    splits = ds.train_test_split(test_size=0.1, seed=42) #NOTE test size was 0.01 in OG version
    train, test = splits["train"], splits["test"]

    data = {"train": train, "test": test}

    return data


def write(
    data: Dict[str, Dataset],
    out_dir: Path,
    max_length: int,
    tokenizer: AutoTokenizer,
    drop_over_limit: bool,
) -> None:
    """Write datasets to binary files and collect metadata."""

    meta: dict[str, Any] = {}
    total_tokens_train = 0
    total_tokens_test = 0

    for split in ["train", "test"]:
        subset = data[split]
        out_path = out_dir / f"{split}.bin"
        if out_path.exists():
            os.remove(out_path)

        # write tokens
        memmap_write(
            out_path,
            subset["ids"],
            np.uint16,
        )

        # ---------- per‑split statistics ----------
        total_tokens = int(np.sum(subset["len"]))
        example_text = tokenizer.decode(subset[-1]["ids"], skip_special_tokens=False)

        meta[split] = {
            "total_tokens": total_tokens,
            "example": example_text,
        }

        if split == "train":
            total_tokens_train += total_tokens
        else:
            total_tokens_test += total_tokens

    # ---------- global statistics ----------
    meta["all"] = {
        "total_tokens_train": total_tokens_train,
        "total_tokens_test": total_tokens_test,
        "vocab_size": len(tokenizer),
        "max_length": max_length,
        "drop_over_limit": drop_over_limit,
    }

    # ---------------------------------------------------- #
    # dump metadata.json                                   #
    # ---------------------------------------------------- #
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- #
# main preparation sequence                                                   #
# --------------------------------------------------------------------------- #


def run(
        tokenizer: AutoTokenizer,
        out_dir: Path | None, 
        num_proc: int, 
        max_length: int, 
        drop_over_limit: bool, 
    ) -> None:

    if out_dir is None:
        dir_str = f"stories_{max_length}"
        cur_dir = Path(__file__).parent
        out_dir = cur_dir / dir_str
    
    print("out_dir:", out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    data = prep(
        num_proc=num_proc,
        tokenizer=tokenizer,
        max_length=max_length,
        drop_over_limit=drop_over_limit,
    )

    # Write datasets and metadata
    write(
        data=data,
        out_dir=out_dir,
        max_length=max_length,
        tokenizer=tokenizer,
        drop_over_limit=drop_over_limit,
    )

    print("Done - binary shards + metadata.json written to", out_dir)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    ap = argparse.ArgumentParser("Prepare simple stories")
    ap.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neo-125M")
    ap.add_argument("--out_dir", default=None, help="directory to write .bin files")
    ap.add_argument("--num_proc", type=int, default=19)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--drop_over_limit", action="store_true") #default truncates, set to True to drop
    args = ap.parse_args()

    run(
        tokenizer=args.tokenizer,
        out_dir=args.out_dir,
        num_proc=args.num_proc,
        max_length=args.max_length,
        drop_over_limit=args.drop_over_limit,
    )
