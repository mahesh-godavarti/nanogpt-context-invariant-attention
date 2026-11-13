#!/bin/bash
# nanoGPT value-transport PILOT (user 2026-10-24): 5K-iter configs (6L, 256d, batch 8x8, block 1024, WikiText-103), one seed each:
# B B' Jm Jm' Jmr Jmr' (the unprimed ones retrained so all six share code and seed). Then perplexity at 256..8192:
# B/B' plain and --ntk; Jm-family with cross_shift_c 0 and 1.5. T4: --compile=False, --use_flex=False (SDPA + mask).
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
COMMON="--compile=False --use_flex=False --seed=1337"
echo "##### GPU smoke (20 iters Bp, Jmp) $(date)"
for c in Bp Jmp; do
  if ! python train.py config/train_wikitext103_small_$c.py $COMMON --max_iters=20 --lr_decay_iters=20 --eval_interval=10 --eval_iters=2 --out_dir=out-smoke-vt-$c 2>&1 | grep -aE "Traceback|Error:|loss nan|loss: nan" ; then echo "smoke $c clean"; else echo "##### SMOKE FAILED ($c) $(date)"; exit 1; fi
done
rm -rf out-smoke-vt-*
for c in B Bp Jm Jmp Jmr Jmrp; do
  echo "##### train $c $(date)"
  python train.py config/train_wikitext103_small_$c.py $COMMON 2>&1 | tee logs_vt/train_5k_$c.log | grep -aE "^iter (0|1000|2000|3000|4000|4990):|step [0-9]+: train loss|Error|Traceback"
done
echo "##### evals $(date)"
L="--lengths 256 512 1024 2048 4096 8192 --n_batches 100"
for c in B Bp; do for flag in "" "--ntk"; do echo "##### $c ${flag:-plain}"; python eval_lengths.py --ckpt out-wikitext103-small-$c/ckpt.pt $L $flag 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error"; done; done
for c in Jm Jmp Jmr Jmrp; do for s in 0 1.5; do echo "##### $c c=$s"; python eval_lengths.py --ckpt out-wikitext103-small-$c/ckpt.pt $L --no_flex --cross_shift_c $s 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error"; done; done
echo "##### pilot finished $(date)"
