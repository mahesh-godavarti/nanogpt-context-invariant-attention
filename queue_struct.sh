#!/bin/bash
# Structured-data adaptation (STRUCT_DESIGN.md): SmolLM frozen, facts as two-token units via the far path; arms A/B/C.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
echo "##### struct adapt $(date)"
python struct_adapt.py --steps 2000 --batch 32 --K 16 --out out-struct-smollm 2>&1 | grep --line-buffered -vE "Warning|Loading|number of parameters"
echo "##### struct queue finished $(date)"
