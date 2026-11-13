#!/bin/bash
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
for s in 1 2; do
  echo "##### struct pythia seed $s $(date)"
  python struct_adapt.py --ckpt out-pythia-B/ckpt.pt --hf_name EleutherAI/pythia-160m --steps 10000 --batch 32 --K 16 --lr 1e-4 --eval_every 2000 --seed $s --out out-struct-pythia-10k-s$s 2>&1 | grep --line-buffered -vE "Warning|Loading|number of parameters"
done
echo "##### struct pythia seeds finished $(date)"
