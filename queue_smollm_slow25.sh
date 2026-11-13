#!/bin/bash
# SmolLM-135M adapted with the slow-quarter anneal (only the slowest 25% of the frequency bands move to reset-RoPE + content address;
# the fast bands keep continuous RoPE). 500 steps on PG-19 books, alpha 0.1 -> 1 over all 500. Evaluated on WikiText-103 as before.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
COMMON="--compile=False --use_flex=False --seed=1337"
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/smollm_pg19"
P="--data_dir data/smollm_pg19 --train_lengths 512 1024 1536 2048 --max_train_len 2048 --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50 --eval_before"
echo "##### smoke (10 iters slow25) $(date)"
python train.py config/smollm_adapt_slow25.py $COMMON --max_iters=10 --lr_decay_iters=10 --adapt_anneal_iters=10 --eval_interval=5 --eval_iters=2 --out_dir=out-smoke-smollm-slow25 > logs_smollm/smoke_slow25.log 2>&1
if grep -aqE "Traceback|Error:|loss nan|loss: nan" logs_smollm/smoke_slow25.log; then echo "##### SMOKE FAILED $(date)"; exit 1; fi
grep -a "adapt:\|^step" logs_smollm/smoke_slow25.log; echo "smoke clean"; rm -rf out-smoke-smollm-slow25
echo "##### train smollm_adapt_slow25 $(date)"
python train.py config/smollm_adapt_slow25.py $COMMON 2>&1 | tee logs_smollm/train_smollm_adapt_slow25.log | grep -aE "^step [0-9]+: train loss|adapt:|Error|Traceback"
echo "##### slow25: ppl $(date)"
for s in 0 1.5; do echo "##### slow25 c=$s"; python eval_lengths.py --ckpt out-smollm-adaptJm-slow25-500/ckpt.pt $L --no_flex --cross_shift_c $s 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"; done
echo "##### slow25: passkey $(date)"
python passkey.py --ckpt out-smollm-adaptJm-slow25-500/ckpt.pt --out out-smollm-adaptJm-slow25-500-passkey $P --cross_shift_c 0 1.5 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|shift"
echo "##### slow25 queue finished $(date)"
