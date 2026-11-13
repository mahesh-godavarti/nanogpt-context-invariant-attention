# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.

# Control for the Pythia window run: Pythia-160M fine-tuned 500 steps on the same FineWeb-Edu tokens, same schedule, no addresses.
# keys 2048+ tokens back scored by reset-RoPE + content address. Within 2048 tokens the model is the untouched model; no anneal.
# Far scores start with a per-head bias of -4 (learned). Trained with window 1024 on 2048-token FineWeb-Edu sequences so far keys exist
# (user 2026-10-28); evaluated with window 2048 so the deployed model is exact within its full pretrained context.
model_type = 'B'
delimiter_ids = (15, 964)           # Pythia tokenizer: '.' and ' .'
init_from = 'adapt'
adapt = False
adapt_ckpt = 'out-pythia-B/ckpt.pt'
adapt_zero_addr = True
window = 0
grad_checkpoint = False
far_bias_init = -6.0   # Pythia: its attention scores are larger, -4 let far keys disturb the start by +0.28 nats; -6 by +0.06

out_dir = 'out-pythia-Bcont-500-lr2e-5'
eval_interval = 100
eval_iters = 20
log_interval = 10
eval_only = False
always_save_checkpoint = True
wandb_log = False

dataset = 'pythia_fineweb'
gradient_accumulation_steps = 8
batch_size = 1
block_size = 2048

dropout = 0.0
bias = False

learning_rate = 2e-5   # 5x lower: at 1e-4 the control's val loss jumped 3.35 -> 3.79 during warmup
max_iters = 500
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 50
lr_decay_iters = 500
min_lr = 2e-6

compile = False
