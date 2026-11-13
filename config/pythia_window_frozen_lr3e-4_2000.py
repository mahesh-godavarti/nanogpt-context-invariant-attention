# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.

# Pythia-160M (GPT-NeoX, quarter rotary): window mode, separate far-path q/k projections, pretrained model FROZEN. Originally SmolLM-135M with keys closer than 2048 tokens scored exactly as pretrained (continuous RoPE) and
# keys 2048+ tokens back scored by reset-RoPE + content address. Within 2048 tokens the model is the untouched model; no anneal.
# Far scores start with a per-head bias of -4 (learned). Trained with window 1024 on 2048-token book sequences so far keys exist
# (user 2026-10-28); evaluated with window 2048 so the deployed model is exact within its full pretrained context.
model_type = 'Jm'
delimiter_ids = (15, 964)           # Pythia tokenizer
init_from = 'adapt'
adapt = False
adapt_ckpt = 'out-pythia-B/ckpt.pt'
adapt_zero_addr = True
window = 1024            # training window; evaluation uses --window 2048
far_bias_init = -4.0   # -6 may have starved the far path on 410M; -4 as on SmolLM

out_dir = 'out-pythia-window-2000-farproj-frozen-lr3e-4'
eval_interval = 250
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

learning_rate = 2e-5   # schedule base; pretrained weights are frozen
new_param_lr = 3e-4    # rate sweep for the frozen recipe on Pythia
far_proj = True
freeze_pretrained = True
grad_checkpoint = True
max_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 1e-5

compile = False
