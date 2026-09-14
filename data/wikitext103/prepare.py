# Copyright (c) 2026 Mahesh Godavarti
# Licensed under CC BY-NC-SA 4.0. See LICENSE file for details.

"""Prepare WikiText-103 for nanoGPT training.

Downloads WikiText-103-raw-v1 via HuggingFace datasets, tokenizes with
GPT-2 BPE (tiktoken), and writes train.bin / val.bin as uint16 memmap arrays.

~103M train tokens, ~250K val tokens.
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token  # 50256

def tokenize_doc(text):
    """Tokenize a single document, appending EOT."""
    ids = enc.encode_ordinary(text)
    ids.append(eot)
    return ids

if __name__ == '__main__':
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    # Splits: train, validation, test

    for split_name, hf_split in [('train', 'train'), ('val', 'validation')]:
        texts = dataset[hf_split]['text']
        # Filter empty lines and tokenize
        all_ids = []
        for text in texts:
            text = text.strip()
            if not text:
                continue
            all_ids.extend(tokenize_doc(text))

        all_ids = np.array(all_ids, dtype=np.uint16)
        out_path = os.path.join(os.path.dirname(__file__), f'{split_name}.bin')
        arr = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=all_ids.shape)
        arr[:] = all_ids
        arr.flush()
        print(f"{split_name}: {len(all_ids):,} tokens -> {out_path}")
