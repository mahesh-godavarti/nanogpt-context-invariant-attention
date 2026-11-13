#!/bin/bash
# Adaptation of pretrained B to content addresses (anneal alpha 0.1 -> 1 over 2500 iters, 5000 total) vs B continued (control).
# Then: ppl at 256..8192 (adapt: offset c = 0 and 1.5; B: plain and NTK) and passkey (fine-tune 300 steps, 50 trials).
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
COMMON="--compile=False --use_flex=False --seed=1337"
echo "##### GPU smoke (20 iters adapt) $(date)"
python train.py config/adapt_B_to_Jm.py $COMMON --max_iters=20 --lr_decay_iters=20 --adapt_anneal_iters=10 --eval_interval=10 --eval_iters=2 --out_dir=out-smoke-adapt > logs_vt/smoke_adapt.log 2>&1
if grep -aqE "Traceback|Error:|loss nan|loss: nan" logs_vt/smoke_adapt.log; then echo "##### SMOKE FAILED $(date)"; exit 1; fi
grep -a "adapt: loaded" logs_vt/smoke_adapt.log; echo "smoke clean"; rm -rf out-smoke-adapt
for c in adapt_B_to_Jm cont_B; do
  echo "##### train $c $(date)"
  python train.py config/$c.py $COMMON 2>&1 | tee logs_vt/train_$c.log | grep -aE "^iter (0|1000|2500|4990):|step (0|2500|5000): train loss|adapt: loaded|Error|Traceback"
done
echo "##### ppl evals $(date)"
L="--lengths 256 512 1024 2048 4096 8192 --n_batches 100"
for s in 0 1.5; do echo "##### adaptJm c=$s"; python eval_lengths.py --ckpt out-wikitext103-small-adaptJm/ckpt.pt $L --no_flex --cross_shift_c $s 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"; done
echo "##### Bcont plain"; python eval_lengths.py --ckpt out-wikitext103-small-Bcont/ckpt.pt $L 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### Bcont ntk";   python eval_lengths.py --ckpt out-wikitext103-small-Bcont/ckpt.pt $L --ntk 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### passkey $(date)"
P="--eval_lengths 512 1024 2048 4096 8192 --n_eval 50 --eval_before"
echo "########## adaptJm"; python passkey.py --ckpt out-wikitext103-small-adaptJm/ckpt.pt --out out-wikitext103-small-adaptJm-passkey $P --cross_shift_c 0 1.5 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|shift"
echo "########## Bcont";   python passkey.py --ckpt out-wikitext103-small-Bcont/ckpt.pt --out out-wikitext103-small-Bcont-passkey $P --ntk 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|ntk"
echo "##### adapt queue finished $(date)"
