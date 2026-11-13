# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.
"""Save a public Hugging Face LLaMA-architecture model (default SmolLM-135M) as a model-B checkpoint of this repo,
so eval_lengths.py / passkey.py can evaluate the untouched model and train.py can adapt it (init_from='adapt').
    python make_hf_ckpt.py --name HuggingFaceTB/SmolLM-135M --out out-smollm-B --delimiter_ids 30 1673
"""
import argparse, dataclasses, os, torch
from model import GPT
p = argparse.ArgumentParser()
p.add_argument('--name', default='HuggingFaceTB/SmolLM-135M'); p.add_argument('--out', default='out-smollm-B')
p.add_argument('--delimiter_ids', nargs='+', type=int, default=[30, 1673]); p.add_argument('--dataset', default='smollm_pg19')
p.add_argument('--arch', default='llama', choices=['llama', 'neox'])
a = p.parse_args()
m = (GPT.from_hf_llama if a.arch == 'llama' else GPT.from_hf_neox)(a.name, model_type='B', delimiter_ids=tuple(a.delimiter_ids))
os.makedirs(a.out, exist_ok=True)
torch.save({'model': m.state_dict(), 'model_args': dataclasses.asdict(m.config), 'iter_num': 0, 'best_val_loss': 1e9,
            'config': dict(dataset=a.dataset, source=a.name)}, os.path.join(a.out, 'ckpt.pt'))
print("saved", os.path.join(a.out, 'ckpt.pt'), "block_size", m.config.block_size, "vocab", m.config.vocab_size)
