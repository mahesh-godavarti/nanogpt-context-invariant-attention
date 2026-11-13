#!/bin/bash
# Pythia-410M frozen recipe at new-param lr 1e-4, far bias -4 (the setting that works on Pythia-160M). Waits for the running sweep.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
until grep -q "frozen lr sweep 2 finished" logs_smollm/queue_pythia_frozen_lr2.log 2>/dev/null; do sleep 60; done
COMMON="--compile=False --use_flex=False --seed=1337"
L="--lengths 512 1024 2048 4096 8192 --n_batches 50 --data_dir data/pythia_fineweb"
PW="--data_dir data/pythia_fineweb --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --freeze_pretrained --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
O=out-pythia410-window-500-farproj-frozen-lr1e-4
echo "##### train 410M frozen lr1e-4 $(date)"
python train.py config/pythia410_window_frozen_lr1e-4.py $COMMON 2>&1 | tee logs_smollm/train_pythia410_frozen_lr1e-4.log | grep --line-buffered -aE "^step [0-9]+: train loss|Error|Traceback" | cut -c1-200
echo "##### ppl (window 2048, c=0) $(date)"; python eval_lengths.py --ckpt $O/ckpt.pt $L --no_flex --window 2048 --cross_shift_c 0 2>&1 | grep --line-buffered -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### ppl 16384 alone"; python eval_lengths.py --ckpt $O/ckpt.pt --lengths 16384 --n_batches 50 --data_dir data/pythia_fineweb --no_flex --window 2048 --cross_shift_c 0 2>&1 | grep --line-buffered -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### passkey $(date)"; python passkey.py --ckpt $O/ckpt.pt --out $O-passkey-ft4096 $PW --cross_shift_c 0 > logs_smollm/passkey_pythia410_frozen_lr1e-4_full.log 2>&1
grep -aE "ctx=|after|Error|Traceback" logs_smollm/passkey_pythia410_frozen_lr1e-4_full.log
echo "##### 410M frozen lr1e-4 finished $(date)"
