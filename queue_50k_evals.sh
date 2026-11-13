#!/bin/bash
# Waits for the Jmr 50K run to finish, then runs the paired length sweeps and passkey for the 50K models.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
until tr '\r' '\n' < run_50k_Jmr.log | grep -aq "^iter 50000:"; do sleep 60; done
until ! pgrep -f "train.py config/train_wikitext103_small_Jmr" > /dev/null; do sleep 10; done
sleep 20
echo "##### evals started $(date)"
{
  for flag in "" "--ntk"; do echo "##### B ${flag:-plain}"; python eval_lengths.py --ckpt out-50k-B/ckpt.pt --lengths 1024 2048 4096 8192 16384 --n_batches 60 $flag 2>&1 | grep -E "^\s+[0-9]+\s+[0-9.]+|Error|rope_ntk"; done
  for m in Jm Jmr; do for c in 0 1.5; do echo "##### $m c=$c"; python eval_lengths.py --ckpt out-50k-$m/ckpt.pt --lengths 1024 2048 4096 8192 16384 --no_flex --n_batches 60 --cross_shift_c $c 2>&1 | grep -E "^\s+[0-9]+\s+[0-9.]+|Error|cross_shift"; done; done
} 2>&1 | tee ppl_50k.log
{
  echo "########## B"; python passkey.py --ckpt out-50k-B/ckpt.pt --out out-50k-B-passkey --eval_before --ntk --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn
  for m in Jm Jmr; do echo "########## $m"; python passkey.py --ckpt out-50k-$m/ckpt.pt --out out-50k-$m-passkey --eval_before --cross_shift_c 0 1.5 --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn; done
} 2>&1 | tee passkey_50k.log
echo "##### evals finished $(date)"
