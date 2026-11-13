#!/bin/bash
# Frozen recipe on Pythia-160M: new-parameter learning-rate sweep (3e-4, 3e-3; 1e-3 already run). Passkey fine-tune with the base frozen.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
COMMON="--compile=False --use_flex=False --seed=1337"
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/pythia_fineweb"
PW="--data_dir data/pythia_fineweb --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --freeze_pretrained --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
for lr in 3e-4 3e-3; do
  O=out-pythia-window-500-farproj-frozen-lr$lr
  echo "##### train frozen lr $lr $(date)"
  python train.py config/pythia_window_frozen_lr$lr.py $COMMON 2>&1 | tee logs_smollm/train_pythia_window_frozen_lr$lr.log | grep --line-buffered -aE "^step [0-9]+: train loss|Error|Traceback" | cut -c1-200
  echo "##### ppl lr $lr (window 2048, c=0)"; python eval_lengths.py --ckpt $O/ckpt.pt $L --no_flex --window 2048 --cross_shift_c 0 2>&1 | grep --line-buffered -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
  echo "##### passkey lr $lr $(date)"; python passkey.py --ckpt $O/ckpt.pt --out $O-passkey-ft4096 $PW --cross_shift_c 0 > logs_smollm/passkey_pythia_frozen_lr${lr}_full.log 2>&1
  grep -aE "ctx=|after|Error|Traceback" logs_smollm/passkey_pythia_frozen_lr${lr}_full.log
done
echo "##### frozen lr sweep finished $(date)"
