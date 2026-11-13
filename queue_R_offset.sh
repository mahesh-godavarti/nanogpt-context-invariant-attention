#!/bin/bash
# Model R with the cross-segment offset (evaluation only; same checkpoints as before).
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
for s in 0 1 2; do
  sfx=$([ $s -eq 0 ] && echo "" || echo "-s$s")
  { for c in 1.0 1.5; do echo "##### R s$s c=$c"; python eval_lengths.py --ckpt out-50k-R$sfx/ckpt.pt --lengths 1024 2048 4096 8192 16384 --no_flex --n_batches 60 --cross_shift_c $c 2>&1 | grep -E "^\s+[0-9]+\s+[0-9.]+|Error"; done; } 2>&1 | tee ppl_50k_R_offset_s$s.log
done
{
  for fs in 0 1 2; do
    echo "########## R seed=$fs"
    python passkey.py --ckpt out-50k-R-passkey-long-s$fs/ckpt.pt --eval_only --n_eval 100 --seed $fs --cross_shift_c 0 1.0 --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn
  done
  for s in 1 2; do
    echo "########## R s$s"
    python passkey.py --ckpt out-50k-R-s$s-passkey-long/ckpt.pt --eval_only --n_eval 100 --seed 0 --cross_shift_c 0 1.0 --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn
  done
} 2>&1 | tee passkey_50k_long_R_offset.log
{
  echo "########## R ceiling"
  python passkey.py --ckpt out-50k-R-passkey-5000/ckpt.pt --eval_only --n_eval 100 --seed 0 --cross_shift_c 0 1.0 --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn
} 2>&1 | tee passkey_50k_ceiling_R_offset.log
echo "##### R offset finished $(date)"
