#!/bin/bash
# Model A (ALiBi): seed 0 at 50K with the full evaluation set first (so one complete column
# exists early), then seeds 1 and 2 with their perplexity sweeps and one passkey run each.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
COMMON="--n_layer=12 --n_head=8 --n_embd=512 --max_iters=50000 --lr_decay_iters=50000 --compile=False --delimiter_id=764 --block_size=1024 --batch_size=2 --gradient_accumulation_steps=8"
PK="--n_eval 100 --eval_lengths 512 1024 2048 4096 8192 16384"

echo "##### train A seed=0 $(date)"
python train.py config/train_wikitext103_small_A.py $COMMON --out_dir=out-50k-A --init_from=resume --seed=1337 2>&1 | tee -a run_50k_A.log
echo "##### evals A seed=0 $(date)"
{ echo "##### A s0 plain"; python eval_lengths.py --ckpt out-50k-A/ckpt.pt --lengths 1024 2048 4096 8192 16384 --n_batches 60 2>&1 | grep -E "^\s+[0-9]+\s+[0-9.]+|Error"; } 2>&1 | tee ppl_50k_A_s0.log
{
  for fs in 0 1 2; do echo "########## A seed=$fs"; python passkey.py --ckpt out-50k-A/ckpt.pt --out out-50k-A-passkey-long-s$fs --finetune_iters 1500 --seed $fs $PK 2>&1 | grep -vi warn; done
} 2>&1 | tee passkey_50k_long_A.log
{ echo "########## A ceiling"; python passkey.py --ckpt out-50k-A/ckpt.pt --out out-50k-A-passkey-5000 --finetune_iters 5000 --lr 1e-4 --seed 0 $PK 2>&1 | grep -vi warn; } 2>&1 | tee passkey_50k_ceiling_A.log
{ echo "########## A s0"; python passkey.py --ckpt out-50k-A/ckpt.pt --out out-50k-A-passkey-2500 --finetune_iters 2500 --lr 1e-4 --seed 0 $PK 2>&1 | grep -vi warn; } 2>&1 | tee passkey_50k_budget2500_A.log
echo "##### seed 0 complete $(date)"

for s in 1; do   # seed 2 does not fit the 32-hour budget at 0.84 s/step
  echo "##### train A seed=$s $(date)"
  python train.py config/train_wikitext103_small_A.py $COMMON --out_dir=out-50k-A-s$s --seed=$((1337 + s)) 2>&1 | tee run_50k_A-s$s.log
  { echo "##### A s$s plain"; python eval_lengths.py --ckpt out-50k-A-s$s/ckpt.pt --lengths 1024 2048 4096 8192 16384 --n_batches 60 2>&1 | grep -E "^\s+[0-9]+\s+[0-9.]+|Error"; } 2>&1 | tee ppl_50k_A_s$s.log
  { echo "########## A s$s"; python passkey.py --ckpt out-50k-A-s$s/ckpt.pt --out out-50k-A-s$s-passkey-long --finetune_iters 1500 --seed 0 $PK 2>&1 | grep -vi warn; } 2>&1 | tee -a passkey_50k_long_A.log
  { echo "########## A s$s"; python passkey.py --ckpt out-50k-A-s$s/ckpt.pt --out out-50k-A-s$s-passkey-2500 --finetune_iters 2500 --lr 1e-4 --seed 0 $PK 2>&1 | grep -vi warn; } 2>&1 | tee -a passkey_50k_budget2500_A.log
  { echo "########## A s$s"; python passkey.py --ckpt out-50k-A-s$s/ckpt.pt --out out-50k-A-s$s-passkey-5000 --finetune_iters 5000 --lr 1e-4 --seed 0 $PK 2>&1 | grep -vi warn; } 2>&1 | tee -a passkey_50k_budget5000_A.log
done
echo "##### A finished $(date)"
