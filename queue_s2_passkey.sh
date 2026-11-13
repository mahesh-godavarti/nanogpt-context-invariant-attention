#!/bin/bash
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
O=out-pythia-window-500-farproj-frozen-lr1e-4-s2
PW="--data_dir data/pythia_fineweb --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --freeze_pretrained --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
echo "##### seed-2 passkey rerun: fine-tune seed 12, 600 iters $(date)"
python passkey.py --ckpt $O/ckpt.pt --out $O-passkey-ft4096-r2 $PW --cross_shift_c 0 --seed 12 --finetune_iters 600 > logs_smollm/passkey_pythia_frozen_lr1e-4_s2_r2_full.log 2>&1
grep -aE "iter +(100|300|600) |ctx=|after|Error|Traceback" logs_smollm/passkey_pythia_frozen_lr1e-4_s2_r2_full.log | cut -c1-100
echo "##### seed-2 passkey rerun finished $(date)"
