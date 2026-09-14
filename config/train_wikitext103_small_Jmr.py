# Copyright (c) 2026 Mahesh Godavarti
# Licensed under CC BY-NC-SA 4.0. See LICENSE file for details.

# Train model Jmr (two-score attention + random addresses) on WikiText-103 -- small model
model_type = 'Jmr'
delimiter_id = 628

out_dir = 'out-wikitext103-small-Jmr'
eval_interval = 500
eval_iters = 200
log_interval = 10
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

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

learning_rate = 6e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 1000
lr_decay_iters = 5000
min_lr = 6e-5

compile = True
