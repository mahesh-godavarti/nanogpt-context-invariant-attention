#!/bin/bash
# Passkey for B and B' (pilot-1 checkpoints carried delimiter_id 628, which passkey.py uses to place the key sentence; copies with 764 are used, B ignores the delimiter).
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
L="--eval_lengths 512 1024 2048 4096 8192 --n_eval 50 --eval_before"
for c in B Bp; do echo "########## $c $(date)"; python passkey.py --ckpt out-wikitext103-small-$c-d764/ckpt.pt --out out-wikitext103-small-$c-passkey $L --ntk 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|ntk"; done
echo "##### passkey B finished $(date)"
