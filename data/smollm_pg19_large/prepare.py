# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.
"""Larger PG-19 book set (~100M tokens, SmolLM tokenizer) for the 5000-step adaptation. train.bin written in chunks;
val.bin and meta.pkl are the same as data/smollm_pg19 (WikiText-103 validation + test)."""
import os, shutil, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); SMALL = os.path.join(os.path.dirname(HERE), "smollm_pg19")
NAME = "HuggingFaceTB/SmolLM-135M"; TARGET = 100_000_000
tok = AutoTokenizer.from_pretrained(NAME)
out = open(os.path.join(HERE, "train.bin"), "wb"); total = 0; n_books = 0; buf = []
for ex in load_dataset("emozilla/pg19", split="train", streaming=True):
    t = tok.encode(ex["text"]); buf.extend(t); total += len(t); n_books += 1
    if len(buf) > 5_000_000:
        np.array(buf, dtype=np.uint16).tofile(out); buf = []; print(f"books {n_books}, tokens {total}", flush=True)
    if total >= TARGET: break
np.array(buf, dtype=np.uint16).tofile(out); out.close()
shutil.copy(os.path.join(SMALL, "val.bin"), HERE); shutil.copy(os.path.join(SMALL, "meta.pkl"), HERE)
print(f"done: {n_books} books, {total} tokens", flush=True)
