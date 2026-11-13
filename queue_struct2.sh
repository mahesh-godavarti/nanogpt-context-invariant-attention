#!/bin/bash
# Structured-data adaptation, 5000 steps, training contexts of 16-64 facts.: SmolLM frozen, facts as two-token units via the far path; arms A/B/C.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
echo "##### struct adapt $(date)"
python struct_adapt.py --steps 5000 --batch 32 --K 16 --K_max 64 --eval_every 500 --out out-struct-smollm-5k-mixK 2>&1 | grep --line-buffered -vE "Warning|Loading|number of parameters"
echo "##### struct queue finished $(date)"
