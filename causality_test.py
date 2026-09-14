#!/usr/bin/env python3
# Copyright (c) 2026 Mahesh Godavarti
# Licensed under CC BY-NC-SA 4.0. See LICENSE file for details.

"""Causality test: no position's output may depend on tokens after it.

For each model (B, Jm, Jmr) and each perturbation below, one token at position
p is changed and the logits at positions < p are compared before and after.
A causal model gives an exact 0 difference on CPU.

The perturbations are chosen to exercise the segment machinery, not just the
attention mask:

  1. token -> other token inside a completed segment
       (a full-segment address must not reach the queries inside that segment)
  2. token -> other token inside the final segment
  3. token -> DELIMITER inside the final segment
       (positions before p stop being "final segment"; anything keyed on
        which segment is last would fail here)
  4. token -> DELIMITER inside a completed segment (splits the segment)
  5. DELIMITER -> other token (merges two segments)

Random-token perturbation alone (the old test) almost never creates or removes
a delimiter and therefore cannot detect a dependence on future segment
structure.

Runs on CPU (SDPA fallback path). For Jmr the RNG is seeded identically before
each forward pass so the random addresses match.
"""

import torch
from model import GPTConfig, GPT

TOL = 1e-5


def max_delta(model, x, p, new_token, seed=99):
    """Max |logit change| at positions < p after setting x[p] = new_token."""
    dummy_y = torch.zeros_like(x)
    with torch.no_grad():
        torch.manual_seed(seed)
        logits_orig, _ = model(x, dummy_y)
        x_edit = x.clone()
        x_edit[0, p] = new_token
        torch.manual_seed(seed)
        logits_edit, _ = model(x_edit, dummy_y)
    return (logits_orig[0, :p] - logits_edit[0, :p]).abs().max().item()


def main():
    T = 128
    vocab_size = 256
    delimiter_id = 200

    torch.manual_seed(42)
    x = torch.randint(0, vocab_size, (1, T))
    x[x == delimiter_id] = 1          # no accidental delimiters
    x[0, 20] = delimiter_id
    x[0, 50] = delimiter_id
    x[0, 80] = delimiter_id           # segments [0..20] [21..50] [51..80] [81..127]

    other = 3
    cases = [
        ("token -> other, completed segment (p=60)", 60, other),
        ("token -> other, final segment     (p=100)", 100, other),
        ("token -> DELIMITER, final segment (p=100)", 100, delimiter_id),
        ("token -> DELIMITER, completed seg (p=65)", 65, delimiter_id),
        ("DELIMITER -> other (p=50)", 50, other),
    ]

    print(f"T={T}, delimiters at 20, 50, 80. Tolerance {TOL:g}.\n")

    small_config = dict(
        block_size=T, vocab_size=vocab_size, n_layer=2, n_head=4,
        n_embd=64, dropout=0.0, bias=False, delimiter_id=delimiter_id,
    )
    all_pass = True
    for model_type in ['B', 'Jm', 'Jmr']:
        torch.manual_seed(42)
        model = GPT(GPTConfig(**small_config, model_type=model_type)).eval()
        print(f"Model {model_type}:")
        for name, p, tok in cases:
            d = max_delta(model, x, p, tok)
            ok = d < TOL
            all_pass &= ok
            print(f"  {'PASS' if ok else 'FAIL'}  {name:44s} max|dlogit| at pos<{p} = {d:.2e}")
        print()
        del model

    if all_pass:
        print("All models PASS causality test.")
    else:
        print("SOME MODELS FAILED causality test!")
        exit(1)


if __name__ == '__main__':
    main()
