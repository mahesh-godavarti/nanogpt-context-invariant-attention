# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.

# Control for the 5000-step anneal: SmolLM-135M fine-tuned 5000 steps on the same 80M book tokens, no addresses.
model_type = 'B'
delimiter_ids = (30, 1673)          # SmolLM tokenizer: '.' and ' .'
init_from = 'adapt'
adapt = False
adapt_ckpt = 'out-smollm-B/ckpt.pt'  # the untouched SmolLM saved as a model-B checkpoint (make_hf_ckpt.py)
adapt_alpha0 = 0.1
adapt_anneal_iters = 5000
adapt_slow_frac = 1.0
adapt_zero_addr = True   # measured before training: zero map shifts the loss less than a random map at every alpha

out_dir = 'out-smollm-Bcont-5000'
eval_interval = 500
eval_iters = 20
log_interval = 10
eval_only = False
always_save_checkpoint = True
wandb_log = False

dataset = 'smollm_pg19_large'
gradient_accumulation_steps = 4
batch_size = 2
block_size = 2048

dropout = 0.0
bias = False

learning_rate = 1e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 1e-5

compile = False
