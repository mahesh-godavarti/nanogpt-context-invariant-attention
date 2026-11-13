#!/bin/bash
# Long adaptation: SmolLM-135M, all bands, alpha 0.1 -> 1 over all 5000 steps (80M tokens of PG-19 books). Waits for the slow25 queue and the data.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
until grep -q "slow25 queue finished\|SMOKE FAILED" logs_smollm/queue_smollm_slow25.log 2>/dev/null; do sleep 120; done
until grep -q "^done:" data/smollm_pg19_large/prepare.log 2>/dev/null; do sleep 60; done
COMMON="--compile=False --use_flex=False --seed=1337"
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/smollm_pg19"
P="--data_dir data/smollm_pg19 --train_lengths 512 1024 1536 2048 --max_train_len 2048 --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50 --eval_before"
echo "##### train smollm_adapt_5000 $(date)"
python train.py config/smollm_adapt_5000.py $COMMON 2>&1 | tee logs_smollm/train_smollm_adapt_5000.log | grep -aE "^step [0-9]+: train loss|adapt:|Error|Traceback"
echo "##### 5000: ppl $(date)"
for s in 0 1.5; do echo "##### 5000 c=$s"; python eval_lengths.py --ckpt out-smollm-adaptJm-5000/ckpt.pt $L --no_flex --cross_shift_c $s 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"; done
echo "##### 5000: passkey $(date)"
python passkey.py --ckpt out-smollm-adaptJm-5000/ckpt.pt --out out-smollm-adaptJm-5000-passkey $P --cross_shift_c 0 1.5 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|shift"
echo "##### 5000 queue finished $(date)"
