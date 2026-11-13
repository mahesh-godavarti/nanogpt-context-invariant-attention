#!/bin/bash
# Passkey retrieval on the 5K pilot checkpoints: B B' (plain and --ntk), Jm Jm' Jmr Jmr' (offset c = 0 and 1.5). Fine-tune 300 steps on <=1024, test 512..8192, 50 trials.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
echo "##### smoke $(date)"
if ! python passkey.py --smoke 2>&1 | grep -aE "Traceback|Error:" ; then echo "smoke clean"; else echo "##### SMOKE FAILED $(date)"; exit 1; fi
L="--eval_lengths 512 1024 2048 4096 8192 --n_eval 50 --eval_before"
for c in B Bp; do echo "########## $c $(date)"; python passkey.py --ckpt out-wikitext103-small-$c/ckpt.pt --out out-wikitext103-small-$c-passkey $L --ntk 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|ntk"; done
for c in Jm Jmp Jmr Jmrp; do echo "########## $c $(date)"; python passkey.py --ckpt out-wikitext103-small-$c/ckpt.pt --out out-wikitext103-small-$c-passkey $L --cross_shift_c 0 1.5 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|shift"; done
echo "##### passkey finished $(date)"
