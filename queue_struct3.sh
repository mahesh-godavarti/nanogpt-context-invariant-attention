#!/bin/bash
# Structured-data adaptation, rotate-then-pool address (slot+relation angles before pooling), 10000 steps at K=16.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
echo "##### struct adapt $(date)"
python struct_adapt.py --steps 10000 --batch 32 --K 16 --eval_every 1000 --out out-struct-smollm-10k-rtp 2>&1 | grep --line-buffered -vE "Warning|Loading|number of parameters"
echo "##### struct queue finished $(date)"
