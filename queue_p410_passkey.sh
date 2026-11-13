#!/bin/bash
# Pythia-410M all-weights window model: passkey fine-tune rerun at lr 1e-5 (the default 5e-5 diverged at iter 300). Waits for the GPU.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
while ps -o args -C python | grep -q "eval_lengths.py"; do sleep 30; done
O=out-pythia410-window-500-splitlr
PW="--data_dir data/pythia_fineweb --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --lr 1e-5 --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
echo "##### 410M passkey lr 1e-5 $(date)"
python passkey.py --ckpt $O/ckpt.pt --out $O-passkey-ft4096-lr1e-5 $PW --cross_shift_c 0 > logs_smollm/passkey_pythia410_splitlr_lr1e-5_full.log 2>&1
grep -aE "iter +(100|200|300) |ctx=|after|Error|Traceback" logs_smollm/passkey_pythia410_splitlr_lr1e-5_full.log | cut -c1-120
echo "##### 410M passkey rerun finished $(date)"
