#!/bin/bash
# Frozen recipe on Pythia-160M: (a) new-param lr 1e-4, 500 steps; (b) lr 3e-4, 2000 steps. Frozen-base passkey fine-tune.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
COMMON="--compile=False --use_flex=False --seed=1337"
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/pythia_fineweb"
PW="--data_dir data/pythia_fineweb --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --freeze_pretrained --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
for c in pythia_window_frozen_lr1e-4:out-pythia-window-500-farproj-frozen-lr1e-4 pythia_window_frozen_lr3e-4_2000:out-pythia-window-2000-farproj-frozen-lr3e-4; do
  cfg=${c%%:*}; O=${c##*:}
  echo "##### train $cfg $(date)"
  python train.py config/$cfg.py $COMMON 2>&1 | tee logs_smollm/train_$cfg.log | grep --line-buffered -aE "^step [0-9]+: train loss|Error|Traceback" | cut -c1-200
  echo "##### ppl $cfg (window 2048, c=0)"; python eval_lengths.py --ckpt $O/ckpt.pt $L --no_flex --window 2048 --cross_shift_c 0 2>&1 | grep --line-buffered -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
  echo "##### passkey $cfg $(date)"; python passkey.py --ckpt $O/ckpt.pt --out $O-passkey-ft4096 $PW --cross_shift_c 0 > logs_smollm/passkey_${cfg}_full.log 2>&1
  grep -aE "ctx=|after|Error|Traceback" logs_smollm/passkey_${cfg}_full.log
done
echo "##### frozen lr sweep 2 finished $(date)"
