#!/bin/bash
# Short adaptation: B pilot checkpoint -> Jm in 500 steps, alpha 0.1 -> 1 over all 500. Then ppl sweep and passkey.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
COMMON="--compile=False --use_flex=False --seed=1337"
echo "##### train adapt_B_to_Jm_500 $(date)"
PYTHONUNBUFFERED=1 python train.py config/adapt_B_to_Jm_500.py $COMMON 2>&1 | tee logs_vt/train_adapt_B_to_Jm_500.log | grep -aE "^step [0-9]+: train loss|adapt: loaded|Error|Traceback"
echo "##### ppl evals $(date)"
L="--lengths 256 512 1024 2048 4096 8192 --n_batches 100"
for s in 0 1.5; do echo "##### adaptJm-500 c=$s"; python eval_lengths.py --ckpt out-wikitext103-small-adaptJm-500/ckpt.pt $L --no_flex --cross_shift_c $s 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"; done
echo "##### passkey $(date)"
P="--eval_lengths 512 1024 2048 4096 8192 --n_eval 50 --eval_before"
python passkey.py --ckpt out-wikitext103-small-adaptJm-500/ckpt.pt --out out-wikitext103-small-adaptJm-500-passkey $P --cross_shift_c 0 1.5 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|shift"
echo "##### adapt500 queue finished $(date)"
