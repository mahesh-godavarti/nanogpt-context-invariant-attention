#!/bin/bash
# Pythia window model, seeds 1 and 2, two-learning-rate recipe (pretrained 2e-5, new params 1e-3); ppl (window 2048, c=0) and passkey (ft @4096).
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/pythia_fineweb"
PW="--data_dir data/pythia_fineweb --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
for s in 1 2; do
  O=out-pythia-window-500-splitlr-s$s
  echo "##### train pythia window splitlr seed $s $(date)"
  python train.py config/pythia_window_splitlr_s$s.py --compile=False --use_flex=False --seed=$s 2>&1 | tee logs_smollm/train_pythia_window_splitlr_s$s.log | grep -aE "^step [0-9]+: train loss|Error|Traceback" | cut -c1-200
  echo "##### ppl seed $s (window 2048, c=0)"; python eval_lengths.py --ckpt $O/ckpt.pt $L --no_flex --window 2048 --cross_shift_c 0 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
  echo "##### passkey seed $s $(date)"; python passkey.py --ckpt $O/ckpt.pt --out $O-passkey-ft4096 $PW --cross_shift_c 0 --seed $s > logs_smollm/passkey_pythia_splitlr_s${s}_full.log 2>&1
  grep -aE "ctx=|after|Error|Traceback" logs_smollm/passkey_pythia_splitlr_s${s}_full.log
done
echo "##### pythia seeds finished $(date)"
