#!/bin/bash
# Pythia at lr 2e-5 (5x lower): window model and no-address control, 500 steps on FineWeb-Edu, WikiText evals.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
COMMON="--compile=False --use_flex=False --seed=1337"
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/pythia_fineweb"
PB="--data_dir data/pythia_fineweb --train_lengths 512 1024 1536 2048 --max_train_len 2048 --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
PW="--data_dir data/pythia_fineweb --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
echo "##### train pythia control lr2e-5 $(date)"
python train.py config/pythia_cont_lr2e-5.py $COMMON 2>&1 | tee logs_smollm/train_pythia_cont_lr2e-5.log | grep -aE "^step [0-9]+: train loss|Error|Traceback" | cut -c1-200
echo "##### control ppl plain"; python eval_lengths.py --ckpt out-pythia-Bcont-500-lr2e-5/ckpt.pt $L 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### control ppl ntk";   python eval_lengths.py --ckpt out-pythia-Bcont-500-lr2e-5/ckpt.pt $L --ntk 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### control passkey $(date)"; python passkey.py --ckpt out-pythia-Bcont-500-lr2e-5/ckpt.pt --out out-pythia-Bcont-500-lr2e-5-passkey $PB --ntk 2>&1 | grep -aE "ctx=|after|Error|Traceback|ntk"
echo "##### train pythia window lr2e-5 $(date)"
python train.py config/pythia_window_lr2e-5.py $COMMON 2>&1 | tee logs_smollm/train_pythia_window_lr2e-5.log | grep -aE "^step [0-9]+: train loss|Error|Traceback" | cut -c1-200
echo "##### window ppl (window 2048, c=0)"; python eval_lengths.py --ckpt out-pythia-window-500-lr2e-5/ckpt.pt $L --no_flex --window 2048 --cross_shift_c 0 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### window passkey $(date)"; python passkey.py --ckpt out-pythia-window-500-lr2e-5/ckpt.pt --out out-pythia-window-500-lr2e-5-passkey-ft4096 $PW --cross_shift_c 0 > logs_smollm/passkey_pythia_window_lr2e-5_full.log 2>&1
grep -aE "ctx=|after|Error|Traceback" logs_smollm/passkey_pythia_window_lr2e-5_full.log
echo "##### pythia lr queue finished $(date)"
