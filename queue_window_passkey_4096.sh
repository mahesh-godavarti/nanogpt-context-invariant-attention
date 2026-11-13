#!/bin/bash
# Window model passkey: fine-tune on sequences up to 4096 with window 2048 (far keys at their real distances; activation checkpointing), eval window 2048, offset 0.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
P="--data_dir data/smollm_pg19 --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
echo "##### window: passkey fine-tune @4096 $(date)"
python passkey.py --ckpt out-smollm-window-500/ckpt.pt --out out-smollm-window-500-passkey-ft4096 $P --cross_shift_c 0 > logs_smollm/passkey_window_ft4096_full.log 2>&1
grep -aE "ctx=|before|after|Error|Traceback|shift" logs_smollm/passkey_window_ft4096_full.log
echo "##### window passkey @4096 finished $(date)"
