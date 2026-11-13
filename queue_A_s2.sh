#!/bin/bash
# Model A (ALiBi): seed 2 at 50K, perplexity sweep, passkey 1500 / 2500 / 5000.

cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
COMMON="--n_layer=12 --n_head=8 --n_embd=512 --max_iters=50000 --lr_decay_iters=50000 --compile=False --delimiter_id=764 --block_size=1024 --batch_size=2 --gradient_accumulation_steps=8"
PK="--n_eval 100 --eval_lengths 512 1024 2048 4096 8192 16384"

for s in 2; do  echo "##### train A seed=$s $(date)"
  python train.py config/train_wikitext103_small_A.py $COMMON --out_dir=out-50k-A-s$s --seed=$((1337 + s)) 2>&1 | tee run_50k_A-s$s.log
  { echo "##### A s$s plain"; python eval_lengths.py --ckpt out-50k-A-s$s/ckpt.pt --lengths 1024 2048 4096 8192 16384 --n_batches 60 2>&1 | grep -E "^\s+[0-9]+\s+[0-9.]+|Error"; } 2>&1 | tee ppl_50k_A_s$s.log
  { echo "########## A s$s"; python passkey.py --ckpt out-50k-A-s$s/ckpt.pt --out out-50k-A-s$s-passkey-long --finetune_iters 1500 --seed 0 $PK 2>&1 | grep -vi warn; } 2>&1 | tee -a passkey_50k_long_A.log
  { echo "########## A s$s"; python passkey.py --ckpt out-50k-A-s$s/ckpt.pt --out out-50k-A-s$s-passkey-2500 --finetune_iters 2500 --lr 1e-4 --seed 0 $PK 2>&1 | grep -vi warn; } 2>&1 | tee -a passkey_50k_budget2500_A.log
  { echo "########## A s$s"; python passkey.py --ckpt out-50k-A-s$s/ckpt.pt --out out-50k-A-s$s-passkey-5000 --finetune_iters 5000 --lr 1e-4 --seed 0 $PK 2>&1 | grep -vi warn; } 2>&1 | tee -a passkey_50k_budget5000_A.log
done
echo "##### A seed 2 finished $(date)"
