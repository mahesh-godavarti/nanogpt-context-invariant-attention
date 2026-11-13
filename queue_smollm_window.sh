#!/bin/bash
# Window mode: train window 1024 on 2048-token book sequences (500 steps); evaluate with window 2048 (near keys as pretrained, far keys by address).
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
until grep -q "5000 queue finished" logs_smollm/queue_smollm_5000.log 2>/dev/null; do sleep 60; done
COMMON="--compile=False --use_flex=False --seed=1337"
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/smollm_pg19"
P="--data_dir data/smollm_pg19 --train_lengths 512 1024 1536 2048 --max_train_len 2048 --window 2048 --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50 --eval_before"
echo "##### smoke (10 iters window) $(date)"
python train.py config/smollm_window.py $COMMON --max_iters=10 --lr_decay_iters=10 --eval_interval=5 --eval_iters=2 --out_dir=out-smoke-window > logs_smollm/smoke_window.log 2>&1
if grep -aqE "Traceback|Error:|loss nan|loss: nan" logs_smollm/smoke_window.log; then echo "##### SMOKE FAILED $(date)"; tail -5 logs_smollm/smoke_window.log; exit 1; fi
grep -a "adapt:\|^step" logs_smollm/smoke_window.log; echo "smoke clean"; rm -rf out-smoke-window
echo "##### train smollm_window $(date)"
python train.py config/smollm_window.py $COMMON 2>&1 | tee logs_smollm/train_smollm_window.log | grep -aE "^step [0-9]+: train loss|adapt:|Error|Traceback"
echo "##### window: ppl $(date)"
for s in 0 1.5; do echo "##### window c=$s"; python eval_lengths.py --ckpt out-smollm-window-500/ckpt.pt $L --no_flex --window 2048 --cross_shift_c $s 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"; done
echo "##### window: passkey $(date)"
python passkey.py --ckpt out-smollm-window-500/ckpt.pt --out out-smollm-window-500-passkey $P --cross_shift_c 0 1.5 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|shift"
echo "##### window queue finished $(date)"
