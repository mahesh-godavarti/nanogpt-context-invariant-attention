# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.

# Adapt the pretrained continuous-RoPE model B (5K pilot checkpoint) to content addresses.
# Start: alpha = 0.1, key/query angle = pos_in_seg*f + (1-alpha)*seg_start*f (alpha = 0 is B exactly),
# cross-segment keys get alpha * content address (content map at full gradient strength).
# Short budget: alpha goes linearly from 0.1 to 1 over all 500 iterations (a tenth of the pretraining), lr 2e-4 warmup 50 cosine to 2e-5.
model_type = 'Jm'
delimiter_id = 764
init_from = 'adapt'
adapt = True
adapt_ckpt = 'out-wikitext103-small-B/ckpt.pt'
adapt_alpha0 = 0.1
adapt_anneal_iters = 500
adapt_slow_frac = 1.0

out_dir = 'out-wikitext103-small-adaptJm-500'
eval_interval = 100
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
max_iters = 500
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 50
lr_decay_iters = 500
min_lr = 2e-5

compile = True
