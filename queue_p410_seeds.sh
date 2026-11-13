#!/bin/bash
# Pythia-410M seeds 1 and 2: frozen recipe (lr 1e-4) first, then the all-weights two-rate recipe. 16K ppl evaluated alone (memory).
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
L="--lengths 512 1024 2048 4096 8192 --n_batches 50 --data_dir data/pythia_fineweb"
L16="--lengths 16384 --n_batches 50 --data_dir data/pythia_fineweb"
PWF="--data_dir data/pythia_fineweb --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --freeze_pretrained --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
PWA="--data_dir data/pythia_fineweb --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --lr 1e-5 --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
run () {  # cfg outdir seed passkey-args
  echo "##### train $1 seed $3 $(date)"
  python train.py config/$1.py --compile=False --use_flex=False --seed=$3 2>&1 | tee logs_smollm/train_$1.log | grep --line-buffered -aE "^step [0-9]+: train loss|Error|Traceback" | cut -c1-200
  echo "##### ppl $1"; python eval_lengths.py --ckpt $2/ckpt.pt $L --no_flex --window 2048 --cross_shift_c 0 2>&1 | grep --line-buffered -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
  python eval_lengths.py --ckpt $2/ckpt.pt $L16 --no_flex --window 2048 --cross_shift_c 0 2>&1 | grep --line-buffered -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
  echo "##### passkey $1 $(date)"; python passkey.py --ckpt $2/ckpt.pt --out $2-passkey-ft4096 $4 --cross_shift_c 0 --seed $3 > logs_smollm/passkey_$1_full.log 2>&1
  grep -aE "restored|ctx=|Error|Traceback" logs_smollm/passkey_$1_full.log | cut -c1-100
}
for s in 1 2; do run pythia410_window_frozen_lr1e-4_s$s out-pythia410-window-500-farproj-frozen-lr1e-4-s$s $s "$PWF"; done
for s in 1 2; do run pythia410_window_splitlr_s$s out-pythia410-window-500-splitlr-s$s $s "$PWA"; done
echo "##### 410M seeds finished $(date)"
