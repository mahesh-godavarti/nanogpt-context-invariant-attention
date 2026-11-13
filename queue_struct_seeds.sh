#!/bin/bash
# Structured-data adaptation (rotate-then-pool address), seeds 1 and 2, 10000 steps at K=16.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
for s in 1 2; do
  echo "##### struct seed $s $(date)"
  python struct_adapt.py --steps 10000 --batch 32 --K 16 --eval_every 2000 --seed $s --out out-struct-smollm-10k-rtp-s$s 2>&1 | grep --line-buffered -vE "Warning|Loading|number of parameters"
done
echo "##### struct seeds finished $(date)"
