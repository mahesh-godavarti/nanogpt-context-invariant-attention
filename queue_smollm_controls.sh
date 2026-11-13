#!/bin/bash
# Controls (user 2026-10-27): (2) adapt on WikiText train 500 steps, (3) fine-tune on WikiText train 500 steps no addresses,
# (1) fine-tune on 80M book tokens 5000 steps no addresses. Waits for the 5000-step adaptation queue and the WikiText data.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
until grep -q "5000 queue finished" logs_smollm/queue_smollm_5000.log 2>/dev/null; do sleep 120; done
until grep -q "^done:" data/smollm_wikitext/prepare.log 2>/dev/null; do sleep 60; done
COMMON="--compile=False --use_flex=False --seed=1337"
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/smollm_pg19"
P="--train_lengths 512 1024 1536 2048 --max_train_len 2048 --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50 --eval_before"
run_jm () {  # name data_dir
  echo "##### train $1 $(date)"; python train.py config/$1.py $COMMON 2>&1 | tee logs_smollm/train_$1.log | grep -aE "^step [0-9]+: train loss|adapt:|Error|Traceback"
  for s in 0 1.5; do echo "##### $1 ppl c=$s"; python eval_lengths.py --ckpt $2/ckpt.pt $L --no_flex --cross_shift_c $s 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"; done
  echo "##### $1 passkey $(date)"; python passkey.py --ckpt $2/ckpt.pt --out $2-passkey --data_dir $3 $P --cross_shift_c 0 1.5 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|shift"
}
run_b () {
  echo "##### train $1 $(date)"; python train.py config/$1.py $COMMON 2>&1 | tee logs_smollm/train_$1.log | grep -aE "^step [0-9]+: train loss|adapt:|Error|Traceback"
  echo "##### $1 ppl plain"; python eval_lengths.py --ckpt $2/ckpt.pt $L 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
  echo "##### $1 ppl ntk";   python eval_lengths.py --ckpt $2/ckpt.pt $L --ntk 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
  echo "##### $1 passkey $(date)"; python passkey.py --ckpt $2/ckpt.pt --out $2-passkey --data_dir $3 $P --ntk 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|ntk"
}
run_jm smollm_adapt_wt out-smollm-adaptJm-wt-500 data/smollm_wikitext
run_b  smollm_cont_wt  out-smollm-Bcont-wt-500  data/smollm_wikitext
run_b  smollm_cont_5000 out-smollm-Bcont-5000   data/smollm_pg19_large
echo "##### controls queue finished $(date)"
