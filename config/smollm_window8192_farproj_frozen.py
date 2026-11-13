# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.

# 8192-token adaptation sequences (natural long-range repeats for the far path to learn from). Window mode with SEPARATE far-path q/k projections (copied from the pretrained ones) and the pretrained model FROZEN (user 2026-10-30): SmolLM-135M with keys closer than 2048 tokens scored exactly as pretrained (continuous RoPE) and
# keys 2048+ tokens back scored by reset-RoPE + content address. Within 2048 tokens the model is the untouched model; no anneal.
# Far scores start with a per-head bias of -4 (learned). Trained with window 1024 on 2048-token book sequences so far keys exist
# (user 2026-10-28); evaluated with window 2048 so the deployed model is exact within its full pretrained context.
model_type = 'Jm'
delimiter_ids = (30, 1673)
init_from = 'adapt'
adapt = False
adapt_ckpt = 'out-smollm-B/ckpt.pt'
adapt_zero_addr = True
window = 2048            # training on 8192-token sequences: far keys 2048-8191 back exist naturally, so the window matches evaluation
far_bias_init = -4.0

out_dir = 'out-smollm-window8192-500-farproj-frozen'
eval_interval = 100
eval_iters = 20
log_interval = 10
eval_only = False
always_save_checkpoint = True
wandb_log = False

dataset = 'smollm_pg19_large'
gradient_accumulation_steps = 2   # 2 x 8192 = the same 16K tokens per step as before
batch_size = 1
block_size = 8192

dropout = 0.0
bias = False

learning_rate = 2e-5   # schedule base; pretrained weights are frozen
new_param_lr = 1e-3    # far-path q/k projections, address map, far biases
far_proj = True
freeze_pretrained = True
grad_checkpoint = True
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
