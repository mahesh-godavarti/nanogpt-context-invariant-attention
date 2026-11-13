#!/bin/bash
# nanoGPT value-transport pilot, Jm family RERUN with the real delimiter (764 = " ."; the 5K configs carried 628 = "\n\n", which never occurs in the data,
# so the first pilot's Jm/Jmr runs had a single segment per sequence). B and B' from the first pilot stand (they do not use the delimiter).
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
COMMON="--compile=False --use_flex=False --seed=1337"
echo "##### GPU smoke (20 iters Jmp) $(date)"
if ! python train.py config/train_wikitext103_small_Jmp.py $COMMON --max_iters=20 --lr_decay_iters=20 --eval_interval=10 --eval_iters=2 --out_dir=out-smoke-vt-Jmp 2>&1 | grep -aE "Traceback|Error:|loss nan|loss: nan" ; then echo "smoke clean"; else echo "##### SMOKE FAILED $(date)"; exit 1; fi
rm -rf out-smoke-vt-*
for c in Jm Jmp Jmr Jmrp; do
  echo "##### train $c (delimiter 764) $(date)"
  python train.py config/train_wikitext103_small_$c.py $COMMON 2>&1 | tee logs_vt/train_5k_$c.log | grep -aE "^iter (0|2500|4990):|step 5000: train loss|Error|Traceback"
done
echo "##### evals $(date)"
L="--lengths 256 512 1024 2048 4096 8192 --n_batches 100"
for c in Jm Jmp Jmr Jmrp; do for s in 0 1.5; do echo "##### $c c=$s"; python eval_lengths.py --ckpt out-wikitext103-small-$c/ckpt.pt $L --no_flex --cross_shift_c $s 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|cross_shift"; done; done
echo "##### pilot2 finished $(date)"
