#!/bin/bash
# Window model passkey: fine-tune with the TRAINING window 1024 (so far keys exist in the <=2048 fine-tune sequences), evaluate with window 2048.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
P="--data_dir data/smollm_pg19 --train_lengths 512 1024 1536 2048 --max_train_len 2048 --tokens_per_step 2048 --window 1024 --eval_window 2048 --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
echo "##### window: passkey rerun $(date)"
python passkey.py --ckpt out-smollm-window-500/ckpt.pt --out out-smollm-window-500-passkey-w1024 $P --cross_shift_c 0 1.5 > logs_smollm/passkey_window_w1024_full.log 2>&1
grep -aE "ctx=|before|after|Error|Traceback|shift" logs_smollm/passkey_window_w1024_full.log
echo "##### window passkey rerun finished $(date)"
