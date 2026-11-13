#!/bin/bash
# Passkey by fine-tuning budget: R and Jm, three training seeds, 2500 and 5000 steps at lr 1e-4,
# each scored with cross_shift_c = 0 and 1.0. Seed 0 at 5000 steps already exists
# (out-50k-{R,Jm}-passkey-5000, scored in passkey_50k_ceiling.log / passkey_50k_ceiling_R_offset.log).
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
EVAL="--n_eval 100 --seed 0 --cross_shift_c 0 1.0 --eval_lengths 512 1024 2048 4096 8192 16384"
{
  for s in 0 1 2; do
    sfx=$([ $s -eq 0 ] && echo "" || echo "-s$s")
    for m in R Jm; do
      echo "########## $m s$s"
      python passkey.py --ckpt out-50k-$m$sfx/ckpt.pt --out out-50k-$m$sfx-passkey-2500 --finetune_iters 2500 --lr 1e-4 $EVAL 2>&1 | grep -vi warn
    done
  done
} 2>&1 | tee passkey_50k_budget2500.log
{
  for s in 1 2; do
    for m in R Jm; do
      echo "########## $m s$s"
      python passkey.py --ckpt out-50k-$m-s$s/ckpt.pt --out out-50k-$m-s$s-passkey-5000 --finetune_iters 5000 --lr 1e-4 $EVAL 2>&1 | grep -vi warn
    done
  done
} 2>&1 | tee passkey_50k_budget5000_s12.log
echo "##### budget finished $(date)"
