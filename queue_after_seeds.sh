#!/bin/bash
# Waits for queue_seeds.sh to finish, then:
#  1. passkey ceiling check: 5000 fine-tuning steps at lr 1e-4, seed 0, 100 trials, all models (seed-0 checkpoints)
#  2. for training seeds 1 and 2: perplexity sweeps (paired, seeded) and passkey (1500 steps, seed 0)
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
until ! pgrep -f "queue_seeds.sh" > /dev/null; do sleep 120; done
sleep 30
echo "##### ceiling started $(date)"
{
  echo "########## B ceiling"
  python passkey.py --ckpt out-50k-B/ckpt.pt --out out-50k-B-passkey-5000 --finetune_iters 5000 --lr 1e-4 --n_eval 100 --seed 0 --ntk --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn
  for m in Jm Jmr; do
    echo "########## $m ceiling"
    python passkey.py --ckpt out-50k-$m/ckpt.pt --out out-50k-$m-passkey-5000 --finetune_iters 5000 --lr 1e-4 --n_eval 100 --seed 0 --cross_shift_c 0 1.0 --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn
  done
} 2>&1 | tee passkey_50k_ceiling.log
echo "##### ceiling finished $(date)"
for s in 1 2; do
  {
    for flag in "" "--ntk"; do echo "##### B s$s ${flag:-plain}"; python eval_lengths.py --ckpt out-50k-B-s$s/ckpt.pt --lengths 1024 2048 4096 8192 16384 --n_batches 60 $flag 2>&1 | grep -E "^\s+[0-9]+\s+[0-9.]+|Error|rope_ntk"; done
    for m in Jm Jmr; do for c in 0 1.0 1.5; do echo "##### $m s$s c=$c"; python eval_lengths.py --ckpt out-50k-$m-s$s/ckpt.pt --lengths 1024 2048 4096 8192 16384 --no_flex --n_batches 60 --cross_shift_c $c 2>&1 | grep -E "^\s+[0-9]+\s+[0-9.]+|Error|cross_shift"; done; done
  } 2>&1 | tee ppl_50k_s$s.log
  {
    echo "########## B s$s"
    python passkey.py --ckpt out-50k-B-s$s/ckpt.pt --out out-50k-B-s$s-passkey-long --finetune_iters 1500 --n_eval 100 --seed 0 --ntk --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn
    for m in Jm Jmr; do
      echo "########## $m s$s"
      python passkey.py --ckpt out-50k-$m-s$s/ckpt.pt --out out-50k-$m-s$s-passkey-long --finetune_iters 1500 --n_eval 100 --seed 0 --cross_shift_c 0 1.0 1.5 --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn
    done
  } 2>&1 | tee passkey_50k_long_s$s.log
done
echo "##### after-seeds finished $(date)"
