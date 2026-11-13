#!/bin/bash
# Structured-data adaptation on Pythia-160M (frozen base, far projections, rotate-then-pool address), new-param lr 1e-4, 10000 steps at K=16.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
echo "##### struct pythia $(date)"
python struct_adapt.py --ckpt out-pythia-B/ckpt.pt --hf_name EleutherAI/pythia-160m --steps 10000 --batch 32 --K 16 --lr 1e-4 --eval_every 1000 --out out-struct-pythia-10k 2>&1 | grep --line-buffered -vE "Warning|Loading|number of parameters"
echo "##### struct pythia finished $(date)"
