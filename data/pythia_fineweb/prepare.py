# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.
"""Fine-tuning data for the Pythia adaptation, tokenized with the Pythia (GPT-NeoX) tokenizer.
train.bin = ~12M tokens of FineWeb-Edu (2024 web text, post-dates the Pile, so not in Pythia's pretraining data);
val.bin = WikiText-103 validation + test. Delimiters: '.' (15) and ' .' (964)."""
import os, pickle, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); NAME = "EleutherAI/pythia-160m"; TARGET = 12_000_000
tok = AutoTokenizer.from_pretrained(NAME)
out = open(os.path.join(HERE, "train.bin"), "wb"); total = 0; n = 0; buf = []
for ex in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True):
    t = tok.encode(ex["text"]) + [15]; buf.extend(t); total += len(t); n += 1   # documents end with '.'
    if len(buf) > 2_000_000:
        np.array(buf, dtype=np.uint16).tofile(out); buf = []; print(f"docs {n}, tokens {total}", flush=True)
    if total >= TARGET: break
np.array(buf, dtype=np.uint16).tofile(out); out.close()
wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
vids = []
for split in ("validation", "test"):
    vids.extend(tok.encode("".join(wt[split]["text"])))
val = np.array(vids, dtype=np.uint16); val.tofile(os.path.join(HERE, "val.bin"))
pickle.dump(dict(vocab_size=50304, tokenizer=f"hf:{NAME}", delimiter_ids=(15, 964)), open(os.path.join(HERE, "meta.pkl"), "wb"))
train = np.fromfile(os.path.join(HERE, "train.bin"), dtype=np.uint16)
for name, a in (("train (FineWeb-Edu)", train), ("val (WikiText-103 val+test)", val)):
    print(f"done: {name}: {len(a)} tokens, '.' {int((a == 15).sum())}, ' .' {int((a == 964).sum())}, max id {int(a.max())}", flush=True)
