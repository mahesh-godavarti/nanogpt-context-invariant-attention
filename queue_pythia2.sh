#!/bin/bash
# Pythia-160M window run RERUN after the fp16 score-overflow fix (baseline evals from queue_pythia.sh stand).
# Pythia-160M (second public model): untouched baseline (ppl plain/NTK, passkey), then window mode (train window 1024 @2048, 500 steps on
# FineWeb-Edu; eval window 2048; passkey fine-tuned on <=4096 with grad checkpointing). Evaluated on WikiText-103. Waits for the SmolLM seeds queue.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
COMMON="--compile=False --use_flex=False --seed=1337"
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/pythia_fineweb"
PB="--data_dir data/pythia_fineweb --train_lengths 512 1024 1536 2048 --max_train_len 2048 --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
PW="--data_dir data/pythia_fineweb --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
echo "##### smoke (10 iters pythia window) $(date)"
python train.py config/pythia_window.py $COMMON --max_iters=10 --lr_decay_iters=10 --eval_interval=5 --eval_iters=2 --out_dir=out-smoke-pythia > logs_smollm/smoke_pythia.log 2>&1
if grep -aqE "Traceback|Error:|loss nan|loss: nan" logs_smollm/smoke_pythia.log; then echo "##### SMOKE FAILED $(date)"; tail -5 logs_smollm/smoke_pythia.log; exit 1; fi
grep -a "^step" logs_smollm/smoke_pythia.log; echo "smoke clean"; rm -rf out-smoke-pythia
echo "##### train pythia window $(date)"
python train.py config/pythia_window.py $COMMON 2>&1 | tee logs_smollm/train_pythia_window.log | grep -aE "^step [0-9]+: train loss|adapt: content|Error|Traceback"
echo "##### pythia window: ppl (window 2048, c=0) $(date)"; python eval_lengths.py --ckpt out-pythia-window-500/ckpt.pt $L --no_flex --window 2048 --cross_shift_c 0 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### pythia window: passkey $(date)"
python passkey.py --ckpt out-pythia-window-500/ckpt.pt --out out-pythia-window-500-passkey-ft4096 $PW --cross_shift_c 0 > logs_smollm/passkey_pythia_window_full.log 2>&1
grep -aE "ctx=|after|Error|Traceback" logs_smollm/passkey_pythia_window_full.log
echo "##### pythia queue finished $(date)"
