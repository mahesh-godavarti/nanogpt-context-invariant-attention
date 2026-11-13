# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.
"""Fine-tuning data for the SmolLM adaptation: books (PG-19, public-domain Project Gutenberg) tokenized with the SmolLM tokenizer.
train.bin = ~12M tokens of PG-19 books; val.bin = WikiText-103 validation + test (raw), same tokenizer.
Delimiters for segmenting: '.' (id 30) and ' .' (id 1673); WikiText raw has the spaced form."""
import os, pickle, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "HuggingFaceTB/SmolLM-135M"
TARGET = 12_000_000
tok = AutoTokenizer.from_pretrained(NAME)
ids, n_books = [], 0
train_path = os.path.join(HERE, "train.bin")
if os.path.exists(train_path):
    train = np.fromfile(train_path, dtype=np.uint16); print("train.bin exists, skipping books", flush=True)
else:
# deepmind/pg19 needs a loading script (unsupported by datasets >= 4); emozilla/pg19 is a parquet mirror of the same data.
    for src in ("emozilla/pg19", "deepmind/pg19"):
        try:
            ds = load_dataset(src, split="train", streaming=True); it = iter(ds); ex = next(it); print("source", src, list(ex.keys()), flush=True); break
        except Exception as e:
            print("failed", src, repr(e)[:200], flush=True); ex = None
    assert ex is not None
    while True:
        t = tok.encode(ex["text"]); ids.extend(t); n_books += 1
        print(f"book {n_books}: {str(ex.get('short_book_title', ''))[:50]!r} {len(t)} tokens, total {len(ids)}", flush=True)
        if len(ids) >= TARGET: break
        ex = next(it)
    train = np.array(ids, dtype=np.uint16); train.tofile(os.path.join(HERE, "train.bin"))
wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
vids = []
for split in ("validation", "test"):
    text = "".join(wt[split]["text"]); vids.extend(tok.encode(text))
val = np.array(vids, dtype=np.uint16); val.tofile(os.path.join(HERE, "val.bin"))
pickle.dump(dict(vocab_size=49152, tokenizer=f"hf:{NAME}", delimiter_ids=(30, 1673)), open(os.path.join(HERE, "meta.pkl"), "wb"))
for name, a in (("train (PG-19)", train), ("val (WikiText-103 val+test)", val)):
    print(f"{name}: {len(a)} tokens, '.' {int((a == 30).sum())}, ' .' {int((a == 1673).sum())}, max id {int(a.max())}")
