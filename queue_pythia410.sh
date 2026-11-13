#!/bin/bash
# Pythia-410M: untouched baseline (ppl plain/NTK, passkey NTK), then frozen window adaptation with far projections (500 steps on
# FineWeb-Edu), ppl (window 2048, c=0), passkey (ft @4096, grad checkpointing). WikiText-103 evals.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
COMMON="--compile=False --use_flex=False --seed=1337"
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/pythia_fineweb"
PB="--data_dir data/pythia_fineweb --train_lengths 512 1024 1536 2048 --max_train_len 2048 --tokens_per_step 2048 --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
PW="--data_dir data/pythia_fineweb --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
O=out-pythia410-window-500-farproj-frozen
echo "##### smoke (10 iters) $(date)"
python train.py config/pythia410_window_frozen.py $COMMON --max_iters=10 --lr_decay_iters=10 --eval_interval=5 --eval_iters=2 --out_dir=out-smoke-p410 > logs_smollm/smoke_p410.log 2>&1
if grep -aqE "Traceback|Error:|loss nan|loss: nan" logs_smollm/smoke_p410.log; then echo "##### SMOKE FAILED $(date)"; tail -5 logs_smollm/smoke_p410.log; exit 1; fi
grep -a "adapt: far\|adapt: pretrained\|^step" logs_smollm/smoke_p410.log; echo "smoke clean"; rm -rf out-smoke-p410
echo "##### untouched 410M ppl plain"; python eval_lengths.py --ckpt out-pythia410-B/ckpt.pt $L 2>&1 | grep --line-buffered -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### untouched 410M ppl ntk";   python eval_lengths.py --ckpt out-pythia410-B/ckpt.pt $L --ntk 2>&1 | grep --line-buffered -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### untouched 410M passkey $(date)"; python passkey.py --ckpt out-pythia410-B/ckpt.pt --out out-pythia410-B-passkey $PB --ntk 2>&1 | grep --line-buffered -aE "ctx=|after|Error|Traceback|ntk"
echo "##### train 410M frozen window $(date)"
python train.py config/pythia410_window_frozen.py $COMMON 2>&1 | tee logs_smollm/train_pythia410_window_frozen.log | grep --line-buffered -aE "^step [0-9]+: train loss|adapt: far|adapt: pretrained|Error|Traceback" | cut -c1-200
echo "##### 410M window ppl (window 2048, c=0) $(date)"; python eval_lengths.py --ckpt $O/ckpt.pt $L --no_flex --window 2048 --cross_shift_c 0 2>&1 | grep --line-buffered -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### 410M window passkey $(date)"; python passkey.py --ckpt $O/ckpt.pt --out $O-passkey-ft4096 $PW --cross_shift_c 0 > logs_smollm/passkey_pythia410_frozen_full.log 2>&1
grep -aE "ctx=|after|Error|Traceback" logs_smollm/passkey_pythia410_frozen_full.log
echo "##### pythia410 queue finished $(date)"
