# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.
"""WikiText-103 TRAIN split tokenized with the SmolLM tokenizer (adaptation with no domain shift).
val.bin / meta.pkl are copied from data/smollm_pg19 (WikiText-103 validation + test; disjoint from train)."""
import os, shutil, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); SMALL = os.path.join(os.path.dirname(HERE), "smollm_pg19")
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
out = open(os.path.join(HERE, "train.bin"), "wb"); total = 0; buf = []
for i in range(0, len(wt), 20000):
    buf = tok.encode("".join(wt[i:i + 20000]["text"])); total += len(buf); np.array(buf, dtype=np.uint16).tofile(out)
    print(f"lines {i + 20000}, tokens {total}", flush=True)
out.close(); shutil.copy(os.path.join(SMALL, "val.bin"), HERE); shutil.copy(os.path.join(SMALL, "meta.pkl"), HERE)
print(f"done: {total} tokens", flush=True)
