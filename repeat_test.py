# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.
"""Repeated-document test: [passage P][filler of `gap` tokens][P again]; score the second copy (token-level loss and argmax accuracy).
A model that can reach the first copy predicts the second almost perfectly; one that cannot scores it as unseen text.
Trials are seeded per gap, so every model/setting sees identical inputs.

    python repeat_test.py --ckpt out-smollm-B/ckpt.pt --data_dir data/smollm_pg19 --gaps 1024 4096 8192 14000 [--ntk] [--window 2048] [--no_flex]
"""
import argparse, json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True); p.add_argument('--data_dir', required=True)
p.add_argument('--gaps', nargs='+', type=int, default=[1024, 4096, 8192, 14000]); p.add_argument('--passage', type=int, default=300)
p.add_argument('--n', type=int, default=12); p.add_argument('--ntk', action='store_true'); p.add_argument('--window', type=int, default=None)
p.add_argument('--no_flex', action='store_true'); p.add_argument('--cross_shift_c', type=float, default=0.0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--tag', default=''); p.add_argument('--out', default=None)
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
c = torch.load(a.ckpt, map_location='cpu', weights_only=False); ma = dict(c['model_args']); ma['block_size'] = max(ma['block_size'], max(a.gaps) + 2 * a.passage + 8)
m = GPT(GPTConfig(**ma)); m.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in c['model'].items()}); m = m.to(dev).eval()
if a.no_flex: m.config.use_flex = False
if a.window is not None: m.config.window = a.window
m.config.cross_shift_c = a.cross_shift_c
if m.config.model_type == 'B':
    m.config.rope_ntk = a.ntk; m.config.rope_ref_len = c['model_args']['block_size']
val = np.memmap(os.path.join(a.data_dir, 'val.bin'), dtype=np.uint16, mode='r'); N = len(val)
res = {}
print(f"{os.path.basename(os.path.dirname(a.ckpt))} {a.tag} model_type={m.config.model_type} ntk={a.ntk} window={m.config.window}", flush=True)
print(f"{'gap':>6s} {'2nd-copy ppl':>13s} {'2nd-copy acc':>13s} {'unseen ppl':>11s}")
for gap in a.gaps:
    rng = np.random.default_rng(a.seed + 7919 * gap); L = a.passage
    losses, accs, losses_unseen = [], [], []
    for t in range(a.n):
        s_p = int(rng.integers(0, N - L - 1)); s_f = int(rng.integers(0, N - gap - 1))
        while abs(s_f - s_p) < L + gap: s_f = int(rng.integers(0, N - gap - 1))   # filler must not contain the passage
        P = torch.from_numpy(val[s_p:s_p + L].astype(np.int64)); F = torch.from_numpy(val[s_f:s_f + gap].astype(np.int64))
        x = torch.cat([P, F, P]).view(1, -1).to(dev); y = torch.full_like(x, -1); y[0, -L:-1] = x[0, -L + 1:]   # score the second copy (its own next tokens)
        xu = torch.cat([F[-min(gap, 1024):], P]).view(1, -1).to(dev); yu = torch.full_like(xu, -1); yu[0, -L:-1] = xu[0, -L + 1:]  # same passage, no first copy
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
            lg, loss = m(x, y); lgu, lossu = m(xu, yu)   # y[i] is already the token after x[i] (-1 elsewhere)
        pred = lg[0, -L:-1].argmax(-1); tgt = y[0, -L:-1]
        losses.append(loss.item()); accs.append((pred == tgt).float().mean().item()); losses_unseen.append(lossu.item())
    res[gap] = dict(ppl=math.exp(float(np.mean(losses))), acc=float(np.mean(accs)), unseen_ppl=math.exp(float(np.mean(losses_unseen))))
    print(f"{gap:6d} {res[gap]['ppl']:13.2f} {res[gap]['acc']:13.3f} {res[gap]['unseen_ppl']:11.2f}", flush=True)
if a.out:
    json.dump(dict(ckpt=a.ckpt, tag=a.tag, ntk=a.ntk, window=m.config.window, results=res), open(a.out, 'w'), indent=1)
