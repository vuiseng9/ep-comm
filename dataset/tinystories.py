import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from itertools import chain
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import torch.distributed as dist

# Avoid tokenizer thread oversubscription when using datasets.map(num_proc > 1)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    is_dist = world_size > 1
    return is_dist, rank, world_size, local_rank, local_world_size

def get_tinystories_loaders(tokenizer, batch_size, ctx_len,
                            n_row=None, raw_dir="./raw_data"):

    is_dist, rank, world_size, local_rank, local_world_size = get_dist_info()

    REPO_ID = "roneneldan/TinyStories"
    NWORKERS = min(8, os.cpu_count())

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {REPO_ID}...")
    raw = load_dataset(REPO_ID, cache_dir=str(raw_dir))
    raw_train = raw["train"]
    raw_val   = raw["validation"]

    if n_row is not None:
        loaded_n_row = len(raw_train)
        n = min(n_row, loaded_n_row)
        print(f"Limiting train to {n:,} of {loaded_n_row:,} rows")
        raw_train = raw_train.select(range(n))

    print(f"Tokenizing with {tokenizer.__class__.__name__}")

    def tokenize_batch(batch, tokenizer, ctx_len: int):
        return tokenizer(
            batch["text"],
            truncation=False,   #NOTE: DO NOT truncate;packer handles length; don't discard tail of each sample
            padding=False,
        )

    def _tokenize(ds, desc):
        return ds.map(
            tokenize_batch,
            batched=True,
            num_proc=10,
            remove_columns=ds.column_names,
            fn_kwargs={"tokenizer": tokenizer, "ctx_len": ctx_len},
            load_from_cache_file=True,
            desc=desc,
        )

    def _pack(ds, desc):
        def concat_and_chunk(batch):
            ids = list(chain.from_iterable(batch["input_ids"]))  # flatten all stories into one stream
            block = ctx_len + 1                                   # +1 so after the x/y shift we get exactly ctx_len positions
            total = (len(ids) // block) * block                  # trim to a multiple of block
            return {"input_ids": [ids[i : i + block] for i in range(0, total, block)]}

        return ds.map(
            concat_and_chunk,
            batched=True,
            num_proc=NWORKERS,
            remove_columns=ds.column_names,  # drop attention_mask etc.; packer changes row count
            load_from_cache_file=True,
            desc=desc,
        )

    # Rank 0 preprocesses first so the HuggingFace cache is written before other
    # ranks try to read it.  Without this, concurrent writes to the same cache
    # file can corrupt or duplicate work.
    if is_dist and rank != 0:
        dist.barrier()  # non-zero ranks wait here while rank 0 builds the cache

    train_ds = _pack(_tokenize(raw_train, "Tokenizing train"), "Packing train")
    val_ds   = _pack(_tokenize(raw_val,   "Tokenizing val"),   "Packing val")

    if is_dist and rank == 0:
        dist.barrier()  # rank 0 signals: cache is ready, others may proceed

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
        return_tensors="pt",
    )

    def _make_loader(ds, shuffle, sampler=None):
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),  # DistributedSampler handles shuffling
            sampler=sampler,
            collate_fn=collator,
            num_workers=NWORKERS,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=(NWORKERS > 0),
            prefetch_factor=2,
        )

    if is_dist:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)
        train_loader  = _make_loader(train_ds, shuffle=False, sampler=train_sampler)
        val_loader    = _make_loader(val_ds,   shuffle=False, sampler=val_sampler)
    else:
        train_sampler = None
        train_loader  = _make_loader(train_ds, shuffle=True)
        val_loader    = _make_loader(val_ds,   shuffle=False)

    # Caller must call train_sampler.set_epoch(epoch) each epoch for proper shuffling in DDP.
    return train_loader, val_loader, train_sampler



if __name__ == "__main__":
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader, _ = get_tinystories_loader(
        tokenizer=tokenizer,
        batch_size=8,
        ctx_len=256,
        n_row=10_000,
    )

    batch = next(iter(train_loader))

    print("\n=== Batch sanity check ===")
    print("input_ids:     ", tuple(batch["input_ids"].shape))
    print("attention_mask:", tuple(batch["attention_mask"].shape))
    print("labels:        ", tuple(batch["labels"].shape))

    print("\nFirst sample:")
    print("input_ids[:16]:     ", batch["input_ids"][0, :16].tolist())
    print("attention_mask[:16]:", batch["attention_mask"][0, :16].tolist())
    print("labels[:16]:        ", batch["labels"][0, :16].tolist())

    print("\nFor your custom causal LM training:")
    print("x         = input_ids[:, :-1]")
    print("y         = labels[:, 1:]")
    print("loss_mask = (y != -100)")
