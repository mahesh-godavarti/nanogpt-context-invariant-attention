#!/bin/bash
# Waits for queue_50k_evals.sh to finish, then:
#  1. passkey with 1500 fine-tuning steps, 100 trials, 3 seeds, all models (c = 0, 1, 1.5; B with NTK)
#  2. two more 50K training seeds for B, Jm, Jmr
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
until ! pgrep -f "queue_50k_evals.sh" > /dev/null; do sleep 30; done
sleep 20
echo "##### passkey-long started $(date)"
{
  for s in 0 1 2; do
    echo "########## B seed=$s"
    python passkey.py --ckpt out-50k-B/ckpt.pt --out out-50k-B-passkey-long-s$s --finetune_iters 1500 --n_eval 100 --seed $s --ntk --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn
    for m in Jm Jmr; do
      echo "########## $m seed=$s"
      python passkey.py --ckpt out-50k-$m/ckpt.pt --out out-50k-$m-passkey-long-s$s --finetune_iters 1500 --n_eval 100 --seed $s --cross_shift_c 0 1.0 1.5 --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn
    done
  done
} 2>&1 | tee passkey_50k_long.log
echo "##### passkey-long finished $(date)"
for s in 1 2; do
  for m in B Jm Jmr; do
    echo "##### train $m seed=$s $(date)"
    python train.py config/train_wikitext103_small_$m.py --n_layer=12 --n_head=8 --n_embd=512 --max_iters=50000 --lr_decay_iters=50000 --compile=False --out_dir=out-50k-$m-s$s --delimiter_id=764 --block_size=1024 --batch_size=2 --gradient_accumulation_steps=8 --use_flex=False --seed=$((1337 + s)) 2>&1 | tee run_50k_${m}_s$s.log
  done
done
echo "##### seeds finished $(date)"
