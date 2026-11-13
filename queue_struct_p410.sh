#!/bin/bash
# Structured-data adaptation on Pythia-410M (frozen base, far projections, rotate-then-pool), new-param lr 1e-4, 10000 steps at K=16, seeds 0 1 2.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
for s in 0 1 2; do
  echo "##### struct pythia410 seed $s $(date)"
  python struct_adapt.py --ckpt out-pythia410-B/ckpt.pt --hf_name EleutherAI/pythia-410m --steps 10000 --batch 32 --K 16 --lr 1e-4 --eval_every 2000 --seed $s --out out-struct-pythia410-10k-s$s 2>&1 | grep --line-buffered -vE "Warning|Loading|number of parameters"
done
echo "##### struct pythia410 finished $(date)"
