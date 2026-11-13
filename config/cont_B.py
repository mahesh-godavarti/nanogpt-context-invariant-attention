# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.

# Control: model B continued from the same 5K pilot checkpoint with the same schedule as the adaptation run (no addresses).
model_type = 'B'
delimiter_id = 764
init_from = 'adapt'
adapt = False
adapt_ckpt = 'out-wikitext103-small-B/ckpt.pt'
adapt_alpha0 = 0.1
adapt_anneal_iters = 2500
adapt_slow_frac = 1.0

out_dir = 'out-wikitext103-small-Bcont'
eval_interval = 500
eval_iters = 200
log_interval = 10
eval_only = False
always_save_checkpoint = True
wandb_log = False

dataset = 'wikitext103'
gradient_accumulation_steps = 8
batch_size = 8
block_size = 1024

n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.1
bias = False

learning_rate = 2e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 2e-5

compile = True
