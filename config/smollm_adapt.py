# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.

# Adapt the public pretrained SmolLM-135M (LLaMA architecture, RoPE, context 2048, 600B tokens) to content addresses
# using ~8M tokens of PG-19 books (not its pretraining data). alpha goes 0.1 -> 1 over all 500 steps; evaluation on WikiText-103.
model_type = 'Jm'
delimiter_ids = (30, 1673)          # SmolLM tokenizer: '.' and ' .'
init_from = 'adapt'
adapt = True
adapt_ckpt = 'out-smollm-B/ckpt.pt'  # the untouched SmolLM saved as a model-B checkpoint (make_hf_ckpt.py)
adapt_alpha0 = 0.1
adapt_anneal_iters = 500
adapt_slow_frac = 1.0
adapt_zero_addr = True   # measured before training: zero map shifts the loss less than a random map at every alpha

out_dir = 'out-smollm-adaptJm-500'
eval_interval = 100
eval_iters = 20
log_interval = 10
eval_only = False
always_save_checkpoint = True
wandb_log = False

dataset = 'smollm_pg19'
gradient_accumulation_steps = 4
batch_size = 2
block_size = 2048

dropout = 0.0
bias = False

learning_rate = 1e-4
max_iters = 500
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 50
lr_decay_iters = 500
min_lr = 1e-5

compile = False
