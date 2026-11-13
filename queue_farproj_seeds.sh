#!/bin/bash
# Frozen + far-projection window model, SmolLM, seeds 1 and 2.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/smollm_pg19"
PW="--data_dir data/smollm_pg19 --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
for s in 1 2; do
  O=out-smollm-window-500-farproj-frozen-s$s
  echo "##### train farproj frozen seed $s $(date)"
  python train.py config/smollm_window_farproj_frozen_s$s.py --compile=False --use_flex=False --seed=$s 2>&1 | tee logs_smollm/train_farproj_frozen_s$s.log | grep -aE "^step [0-9]+: train loss|Error|Traceback" | cut -c1-200
  echo "##### ppl seed $s (window 2048, c=0)"; python eval_lengths.py --ckpt $O/ckpt.pt $L --no_flex --window 2048 --cross_shift_c 0 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
  echo "##### passkey seed $s $(date)"; python passkey.py --ckpt $O/ckpt.pt --out $O-passkey-ft4096 $PW --cross_shift_c 0 --seed $s > logs_smollm/passkey_farproj_frozen_s${s}_full.log 2>&1
  grep -aE "ctx=|after|Error|Traceback" logs_smollm/passkey_farproj_frozen_s${s}_full.log
done
echo "##### farproj seeds finished $(date)"
