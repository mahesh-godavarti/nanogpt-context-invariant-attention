#!/usr/bin/env python3
# Copyright (c) 2026 Mahesh Godavarti
# Licensed under CC BY-NC-SA 4.0. See LICENSE file for details.

"""Rebuild the seed-averaged tables in the README from the logs in this directory.

    python results/aggregate.py

Perplexity: ppl_50k.log and ppl_50k_c1.log (training seed 0), ppl_50k_s1.log, ppl_50k_s2.log.
Passkey (1500 fine-tuning steps): passkey_50k_long.log (training seed 0, three
fine-tuning seeds), passkey_50k_long_s1.log, passkey_50k_long_s2.log (one each).
Passkey ceiling (5000 steps, lr 1e-4): passkey_50k_ceiling.log.
"""

import collections
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
LENGTHS = [1024, 2048, 4096, 8192, 16384]
PK_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]


def parse_ppl(path):
    """{(setting, ctx): ppl} where setting is like 'B plain', 'B --ntk', 'Jm c=1.5'."""
    out, setting = {}, None
    for line in open(path):
        m = re.match(r'##### (\S+)(?: s\d)? (.*)', line.strip())
        if m:
            setting = f"{m.group(1)} {m.group(2).strip()}"
            continue
        m = re.match(r'\s+(\d+)\s+([\d.]+)\s+\d+', line)
        if m and setting:
            out[(setting, int(m.group(1)))] = float(m.group(2))
    return out


def parse_passkey(path):
    """{(model, setting, ctx): [exact...]}, {(...): [first_token...]} over fine-tuning seeds; 'after' rows only."""
    acc, ft = collections.defaultdict(list), collections.defaultdict(list)
    model = setting = None
    for line in open(path):
        m = re.match(r'########## (\S+)', line)
        if m:
            model = m.group(1)
            continue
        m = re.match(r'Passkey accuracy (before|after) fine-tuning \((\w+)=([\w.]+)\)', line)
        if m:
            setting = None if m.group(1) == 'before' else f"{m.group(2)}={m.group(3)}"
            continue
        m = re.match(r'\s+ctx=\s*(\d+)\s+exact=([\d.]+)\s+first_token=([\d.]+)', line)
        if m and setting:
            key = (model, setting, int(m.group(1)))
            acc[key].append(float(m.group(2)))
            ft[key].append(float(m.group(3)))
    return acc, ft


def table(title, cols, rows, lengths):
    print(f"\n{title}")
    print("| ctx | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for L in lengths:
        cells = []
        for _, key in cols:
            vals = rows.get(key + (L,), [])
            if not vals:
                cells.append("")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.2f}")
            else:
                cells.append(f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}")
        print(f"| {L} | " + " | ".join(cells) + " |")


def main():
    # ---- perplexity: mean ± std over training seeds
    ppl = collections.defaultdict(list)
    for f in ['ppl_50k.log', 'ppl_50k_c1.log', 'ppl_50k_s1.log', 'ppl_50k_s2.log']:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            for (setting, L), v in parse_ppl(p).items():
                ppl[(setting, L)].append(v)
    cols = [("B", ("B plain",)), ("B + NTK", ("B --ntk",)),
            ("Jm", ("Jm c=0",)), ("Jm + offset (c=1)", ("Jm c=1.0",)), ("Jm + offset (c=1.5)", ("Jm c=1.5",)),
            ("Jmr", ("Jmr c=0",)), ("Jmr + offset (c=1.5)", ("Jmr c=1.5",))]
    n = max(len(v) for v in ppl.values()) if ppl else 0
    table(f"Perplexity, mean ± std over {n} training seed(s), 60 seeded windows each", cols, ppl, LENGTHS)

    # ---- passkey, 1500 steps: pool all fine-tuning runs (3 for seed 0, 1 each for seeds 1 and 2)
    acc, ft = collections.defaultdict(list), collections.defaultdict(list)
    for f in ['passkey_50k_long.log', 'passkey_50k_long_s1.log', 'passkey_50k_long_s2.log']:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            a, t = parse_passkey(p)
            for k, v in a.items():
                acc[k].extend(v)
            for k, v in t.items():
                ft[k].extend(v)
    cols = [("B", ("B", "rope_ntk=False")), ("B + NTK", ("B", "rope_ntk=True")),
            ("Jm", ("Jm", "cross_shift_c=0.0")), ("Jm + offset", ("Jm", "cross_shift_c=1.0")),
            ("Jmr", ("Jmr", "cross_shift_c=0.0")), ("Jmr + offset", ("Jmr", "cross_shift_c=1.0"))]
    n = max(len(v) for v in acc.values()) if acc else 0
    table(f"Passkey exact match, 1500 fine-tuning steps, mean ± std over {n} runs x 100 trials", cols, acc, PK_LENGTHS)
    table("Passkey first-token accuracy, same runs", cols, ft, PK_LENGTHS)

    # ---- passkey ceiling: 5000 steps, lr 1e-4, training seed 0, one fine-tuning seed
    p = os.path.join(HERE, 'passkey_50k_ceiling.log')
    if os.path.exists(p):
        a, t = parse_passkey(p)
        table("Passkey exact match, 5000 fine-tuning steps at lr 1e-4 (training seed 0, 100 trials)", cols, a, PK_LENGTHS)
        table("Passkey first-token accuracy, same run", cols, t, PK_LENGTHS)


if __name__ == '__main__':
    main()
