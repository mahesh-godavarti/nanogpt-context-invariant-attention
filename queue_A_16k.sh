#!/bin/bash
# Seed-0 ALiBi cells at 16384 that OOM'd before the chunked attention path existed:
# the perplexity sweep point and the five fine-tuned passkey checkpoints (eval only).
# Waits for queue_A.sh (seed 1) to finish.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
until ! ps aux | grep "queue_A.sh" | grep -v grep | grep -v 16k > /dev/null; do sleep 60; done
echo "##### A seed0 16384 cells started $(date)"
{ echo "##### A s0 plain"; python eval_lengths.py --ckpt out-50k-A/ckpt.pt --lengths 16384 --n_batches 60 2>&1 | grep -E "^\s+[0-9]+\s+[0-9.]+|Error"; } 2>&1 | tee ppl_50k_A_s0_16k.log
PK="--eval_only --n_eval 100 --eval_lengths 16384"
{
  for fs in 0 1 2; do echo "########## A seed=$fs"; python passkey.py --ckpt out-50k-A-passkey-long-s$fs/ckpt.pt --seed $fs $PK 2>&1 | grep -vi warn; done
} 2>&1 | tee passkey_50k_long_A_16k.log
{ echo "########## A ceiling"; python passkey.py --ckpt out-50k-A-passkey-5000/ckpt.pt --seed 0 $PK 2>&1 | grep -vi warn; } 2>&1 | tee passkey_50k_ceiling_A_16k.log
{ echo "########## A s0"; python passkey.py --ckpt out-50k-A-passkey-2500/ckpt.pt --seed 0 $PK 2>&1 | grep -vi warn; } 2>&1 | tee passkey_50k_budget2500_A_16k.log
echo "##### A 16384 cells finished $(date)"
