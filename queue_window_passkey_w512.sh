#!/bin/bash
# Window model passkey: fine-tune with window 512 on <=2048 sequences (3x the far keys, farther), evaluate with window 2048, offset 0 only.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
P="--data_dir data/smollm_pg19 --train_lengths 512 1024 1536 2048 --max_train_len 2048 --tokens_per_step 2048 --window 512 --eval_window 2048 --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
echo "##### window: passkey rerun $(date)"
python passkey.py --ckpt out-smollm-window-500/ckpt.pt --out out-smollm-window-500-passkey-w512 $P --cross_shift_c 0 > logs_smollm/passkey_window_w512_full.log 2>&1
grep -aE "ctx=|before|after|Error|Traceback|shift" logs_smollm/passkey_window_w512_full.log
echo "##### window passkey rerun finished $(date)"
