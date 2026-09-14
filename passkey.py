#!/usr/bin/env python3
# Copyright (c) 2026 Mahesh Godavarti
# Licensed under CC BY-NC-SA 4.0. See LICENSE file for details.

"""Passkey retrieval: fine-tune a checkpoint on short examples, test at long contexts.

An example is WikiText filler with one sentence inserted at a random depth:

    <filler> The pass key is 73215 . Remember it . <filler> The pass key is 73215

Loss and accuracy are computed on the answer tokens only (" 73215" is three
GPT-2 BPE tokens; a trial counts as correct only if every answer token is the
argmax under teacher forcing). The key sentence is placed right after a
sentence-ending " ." token so that it forms its own segment.

Fine-tuning uses lengths up to --max_train_len (default 1024, the pretraining
context). Evaluation lengths beyond that are extrapolation. Evaluation examples
are generated from a fixed seed per (length, trial), so every model is scored
on identical examples.

Usage:
    python passkey.py --ckpt out-5k-Jm/ckpt.pt --out out-5k-Jm-passkey
    python passkey.py --ckpt out-5k-Jm-passkey/ckpt.pt --eval_only
    python passkey.py --smoke            # tiny random model on CPU, exercises all paths
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import GPTConfig, GPT


# ----------------------------------------------------------------------------
# Example construction
# ----------------------------------------------------------------------------

class PasskeyMaker:
    def __init__(self, data, delimiter_id, enc):
        self.data = data            # np.memmap / array of token ids (filler source)
        self.delim = delimiter_id
        self.enc = enc
        self.query = enc.encode(" The pass key is")

    def make(self, L, rng):
        """Return (x, y) of length L each: input tokens and targets (-1 = no loss)."""
        key = int(rng.integers(10000, 100000))
        key_sent = self.enc.encode(f" The pass key is {key} . Remember it .")
        answer = self.enc.encode(f" {key}")
        assert key_sent[-1] == self.delim
        n_fill = L + 1 - len(key_sent) - len(self.query) - len(answer)
        assert n_fill > 20, "length too short for a passkey example"

        start = int(rng.integers(0, len(self.data) - n_fill - 1))
        span = np.asarray(self.data[start:start + n_fill]).astype(np.int64)

        # Insert the key sentence after a sentence boundary near the chosen depth.
        depth = rng.uniform(0.05, 0.95)
        target = int(depth * n_fill)
        ends = np.nonzero(span[:target] == self.delim)[0]
        cut = int(ends[-1]) + 1 if len(ends) > 0 else target
        before, after = span[:cut], span[cut:]

        seq = np.concatenate([before, key_sent, after, self.query, answer])
        assert len(seq) == L + 1, (len(seq), L + 1)
        x = torch.from_numpy(seq[:L])
        y = torch.full((L,), -1, dtype=torch.long)
        n_ans = len(answer)
        y[L - n_ans:] = torch.from_numpy(seq[L + 1 - n_ans:])  # predict each answer token
        return x, y, n_ans


def make_batch(maker, L, bs, rng):
    xs, ys, ns = zip(*[maker.make(L, rng) for _ in range(bs)])
    return torch.stack(xs), torch.stack(ys), ns


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, maker, lengths, n_trials, device, seed=1000):
    """Exact-match accuracy over answer tokens at each length (identical examples per seed)."""
    model.eval()
    results = {}
    for L in lengths:
        model.config.block_size = max(model.config.block_size, L)
        correct = 0
        first_tok = 0
        for t in range(n_trials):
            rng = np.random.default_rng(seed + 100003 * L + t)
            x, y, n_ans = maker.make(L, rng)
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            torch.manual_seed(seed + t)  # Jmr: fixed random addresses per trial
            logits, _ = model(x, y)
            pred = logits[0, L - n_ans:].argmax(-1)
            tgt = y[0, L - n_ans:]
            correct += int(torch.equal(pred, tgt))
            first_tok += int(pred[0].item() == tgt[0].item())
        results[L] = dict(acc=correct / n_trials, first_token_acc=first_tok / n_trials)
        print(f"  ctx={L:6d}  exact={results[L]['acc']:.2f}  first_token={results[L]['first_token_acc']:.2f}",
              flush=True)
    return results


# ----------------------------------------------------------------------------
# Fine-tuning
# ----------------------------------------------------------------------------

def finetune(model, maker, args, device):
    model.train()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1},
                             {'params': nodecay, 'weight_decay': 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    ctx = torch.autocast('cuda', dtype=torch.float16) if use_amp else torch.autocast('cpu', enabled=False)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)  # dropout etc. during fine-tuning
    lengths = [l for l in args.train_lengths if l <= args.max_train_len]
    t0 = time.time()
    for it in range(1, args.finetune_iters + 1):
        L = int(rng.choice(lengths))
        bs = max(1, args.tokens_per_step // L)
        model.config.block_size = max(model.config.block_size, L)
        x, y, _ = make_batch(maker, L, bs, rng)
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if it % args.log_interval == 0 or it == 1:
            print(f"  iter {it:4d}  L={L:4d} bs={bs}  loss {loss.item():.4f}  "
                  f"({(time.time() - t0) / it:.2f}s/it)", flush=True)
    model.eval()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', help='checkpoint to fine-tune (or evaluate with --eval_only)')
    p.add_argument('--out', help='directory for the fine-tuned checkpoint and results')
    p.add_argument('--data_dir', default='data/wikitext103')
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--finetune_iters', type=int, default=300)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--train_lengths', nargs='+', type=int, default=[256, 512, 768, 1024])
    p.add_argument('--max_train_len', type=int, default=1024)
    p.add_argument('--tokens_per_step', type=int, default=4096,
                   help='batch size is tokens_per_step // length')
    p.add_argument('--eval_lengths', nargs='+', type=int, default=[512, 1024, 2048, 4096, 8192])
    p.add_argument('--n_eval', type=int, default=50)
    p.add_argument('--eval_before', action='store_true',
                   help='also evaluate the checkpoint before fine-tuning')
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--flex', action='store_true',
                   help='use FlexAttention on CUDA (default: SDPA with a boolean mask, '
                        'which needs far less memory at long contexts when uncompiled)')
    p.add_argument('--cross_shift_c', nargs='+', type=float, default=[0.0],
                   help='Jm/Jmr: evaluate with each of these cross-segment offset strengths '
                        '(fine-tuning uses the first value)')
    p.add_argument('--ntk', action='store_true',
                   help='B: also evaluate with dynamic NTK scaling beyond max_train_len')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    device = torch.device(args.device)

    if args.smoke:
        # Tiny random model on CPU; only checks that every code path runs.
        device = torch.device('cpu')
        model_args = dict(block_size=256, vocab_size=50304, n_layer=2, n_head=4, n_embd=64,
                          dropout=0.0, bias=False, model_type='Jm', delimiter_id=764)
        model = GPT(GPTConfig(**model_args)).to(device)
        args.finetune_iters, args.n_eval = 3, 2
        args.train_lengths, args.max_train_len = [128, 256], 256
        args.eval_lengths, args.tokens_per_step = [128, 512], 512
        args.out = None
    else:
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model_args = ckpt['model_args']
        model = GPT(GPTConfig(**model_args))
        sd = ckpt['model']
        for k in list(sd.keys()):
            if k.startswith('_orig_mod.'):
                sd[k[len('_orig_mod.'):]] = sd.pop(k)
        model.load_state_dict(sd)
        model = model.to(device)
    model.config.use_flex = bool(args.flex)
    print(f"Model type: {model.config.model_type} | params {model.get_num_params():,} | "
          f"delimiter {model.config.delimiter_id} | device {device}")

    train_data = np.memmap(os.path.join(args.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(args.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    train_maker = PasskeyMaker(train_data, model.config.delimiter_id, enc)
    val_maker = PasskeyMaker(val_data, model.config.delimiter_id, enc)

    is_seg_model = model.config.model_type != 'B'
    if is_seg_model:
        settings = [('cross_shift_c', c) for c in args.cross_shift_c]
    else:
        settings = [('rope_ntk', False)] + ([('rope_ntk', True)] if args.ntk else [])
    model.config.rope_ref_len = args.max_train_len

    def evaluate_all(tag):
        out = {}
        for name, val in settings:
            setattr(model.config, name, val)
            print(f"Passkey accuracy {tag} ({name}={val}):")
            out[f"{name}={val}"] = evaluate(model, val_maker, args.eval_lengths, args.n_eval, device,
                                            seed=1000 + args.seed)
        setattr(model.config, settings[0][0], settings[0][1])
        return out

    setattr(model.config, settings[0][0], settings[0][1])
    results = {}
    if args.eval_only or args.eval_before:
        results['before'] = evaluate_all("before fine-tuning")
    if not args.eval_only:
        print(f"Fine-tuning for {args.finetune_iters} iters on lengths {args.train_lengths}:")
        finetune(model, train_maker, args, device)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            torch.save({'model': model.state_dict(), 'model_args': model_args,
                        'config': dict(passkey_finetune=vars(args))},
                       os.path.join(args.out, 'ckpt.pt'))
        results['after'] = evaluate_all("after fine-tuning")
    if args.out:
        with open(os.path.join(args.out, 'passkey_results.json'), 'w') as f:
            json.dump(dict(model_type=model.config.model_type, results=results, args=vars(args)), f, indent=1)


if __name__ == '__main__':
    main()
