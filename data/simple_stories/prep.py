import argparse
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging

logging.set_verbosity(40)


def memmap_write(
    fname: Path,
    arr: List,
    dtype: np.dtype = np.uint16,
) -> None:

    total = sum(len(a) for a in arr)
    mmap = np.memmap(fname, dtype=dtype, mode="w+", shape=(total,))
    idx = 0
    for a in tqdm(arr, desc="writing", total=len(arr)):
        mmap[idx : idx + len(a)] = a
        idx += len(a)
    mmap.flush()


def prepare_and_tokenize_dataset(
    num_proc: int,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dict[str, Dataset]:

    dset_name = "SimpleStories/SimpleStories"
    train_ds = load_dataset(dset_name, split="train")
    test_ds = load_dataset(dset_name, split="test")

    print("Dataset columns:", train_ds.column_names)

    # --------------------------------------------------------- #
    # 1. tokenisation (no truncation)                           #
    # --------------------------------------------------------- #

    def tok_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        ids = tokenizer.encode(ex["story"], add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        return {"ids": ids, "len": len(ids)}

    train_ds = train_ds.map(tok_fn, num_proc=num_proc)
    test_ds = test_ds.map(tok_fn, num_proc=num_proc)

    # --------------------------------------------------------- #
    # 2. length filtering                                       #
    # --------------------------------------------------------- #

    train_ds = train_ds.filter(lambda ex: ex["len"] <= max_length, num_proc=num_proc)
    test_ds = test_ds.filter(lambda ex: ex["len"] <= max_length, num_proc=num_proc)

    # --------------------------------------------------------- #
    # 3. return datasets                                        #
    # --------------------------------------------------------- #

    out = {
        "train": train_ds,
        "test": test_ds,
    }

    return out


def write_datasets_and_metadata(
    datasets: Dict[str, Dataset],
    out_dir: Path,
    tokenizer: AutoTokenizer,
) -> None:
    """Write datasets to binary files and collect metadata."""

    total_tokens_train = 0
    total_tokens_test = 0
    meta = {}

    for split, data in datasets.items():

        meta[split] = {}

        subset = data
        out_path = out_dir / f"{split}.bin"

        # write tokens
        memmap_write(out_path, subset["ids"])

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
    }

    # ---------------------------------------------------- #
    # dump metadata.json                                   #
    # ---------------------------------------------------- #
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# main preparation sequence                                                   #
# --------------------------------------------------------------------------- #


def run(out_dir: Path, num_proc: int, max_length: int) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")

    datasets = prepare_and_tokenize_dataset(
        num_proc=num_proc,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # Write datasets and metadata
    write_datasets_and_metadata(
        datasets,
        out_dir,
        tokenizer,
    )

    print("Done - binary shards + metadata.json written to", out_dir)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    cur_dir = Path(__file__).parent
    ap = argparse.ArgumentParser("Prepare simple stories")
    ap.add_argument(
        "--out_dir", default=cur_dir, help="directory to write .bin files"
    )
    ap.add_argument("--num_proc", type=int, default=19)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    run(
        out_dir=Path(args.out_dir),
        num_proc=args.num_proc,
        max_length=args.max_length,
    )