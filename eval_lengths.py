#!/usr/bin/env python3
# Copyright (c) 2026 Mahesh Godavarti
# Licensed under CC BY-NC-SA 4.0. See LICENSE file for details.

"""Evaluate a trained model's perplexity at multiple context lengths.

Usage:
    python eval_lengths.py --ckpt out-wikitext103-small-B/ckpt.pt --lengths 256 512 1024 2048 4096
"""

import argparse
import math
import os

import numpy as np
import torch

from model import GPTConfig, GPT


@torch.no_grad()
def eval_ppl(model, data, block_size, batch_size, device, n_batches=200):
    """Evaluate perplexity at a given context length."""
    model.eval()
    total_loss = 0.0
    n = 0
    for _ in range(n_batches):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    return math.exp(total_loss / n) if n > 0 else float('inf')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Path to checkpoint')
    p.add_argument('--data_dir', default=None)
    p.add_argument('--lengths', nargs='+', type=int,
                   default=[256, 512, 1024, 2048, 4096])
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--n_batches', type=int, default=200)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model_args = ckpt['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config)

    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model = model.to(args.device).eval()

    model_type = model_args.get('model_type', 'B')
    print(f"Model type: {model_type}")
    print(f"Params: {model.get_num_params():,}")
    print(f"Trained block_size: {model_args['block_size']}")
    print()

    # Load validation data
    if args.data_dir:
        data_dir = args.data_dir
    else:
        dataset = ckpt.get('config', {}).get('dataset', 'wikitext103')
        data_dir = os.path.join('data', dataset)
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    print(f"Val tokens: {len(val_data):,}")
    print()

    # Eval at each context length
    print(f"{'ctx':>6s}  {'PPL':>10s}  {'batch':>5s}")
    print("-" * 26)
    for L in args.lengths:
        if len(val_data) <= L + 1:
            print(f"{L:>6d}  {'N/A':>10s}  (data too short)")
            continue

        model.config.block_size = L

        # Scale batch size down for longer contexts to avoid OOM
        bs = max(1, args.batch_size * 1024 // L)

        ppl = eval_ppl(model, val_data, L, bs, args.device, args.n_batches)
        print(f"{L:>6d}  {ppl:>10.2f}  {bs:>5d}")

    print()


if __name__ == '__main__':
    main()
