#!/bin/bash
# Model R (reset-RoPE, no addresses): three 50K training seeds, then the same evaluations
# as B/Jm/Jmr: seeded perplexity sweeps (R has no cross-segment offset; NTK is a RoPE-base
# scaling and does not apply to reset positions, so R is evaluated plain), passkey with
# 1500 steps (seeds 0,1,2 on training seed 0; seed 0 on training seeds 1,2), and the
# 5000-step ceiling run on training seed 0.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
for s in 0 1 2; do
  sfx=$([ $s -eq 0 ] && echo "" || echo "-s$s")
  echo "##### train R seed=$s $(date)"
  python train.py config/train_wikitext103_small_R.py --n_layer=12 --n_head=8 --n_embd=512 --max_iters=50000 --lr_decay_iters=50000 --compile=False --out_dir=out-50k-R$sfx --delimiter_id=764 --block_size=1024 --batch_size=2 --gradient_accumulation_steps=8 --use_flex=False --seed=$((1337 + s)) 2>&1 | tee run_50k_R$sfx.log
done
echo "##### evals R $(date)"
for s in 0 1 2; do
  sfx=$([ $s -eq 0 ] && echo "" || echo "-s$s")
  { echo "##### R s$s plain"; python eval_lengths.py --ckpt out-50k-R$sfx/ckpt.pt --lengths 1024 2048 4096 8192 16384 --no_flex --n_batches 60 2>&1 | grep -E "^\s+[0-9]+\s+[0-9.]+|Error"; } 2>&1 | tee ppl_50k_R_s$s.log
done
{
  for fs in 0 1 2; do
    echo "########## R seed=$fs"
    python passkey.py --ckpt out-50k-R/ckpt.pt --out out-50k-R-passkey-long-s$fs --finetune_iters 1500 --n_eval 100 --seed $fs --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn
  done
  for s in 1 2; do
    echo "########## R s$s"
    python passkey.py --ckpt out-50k-R-s$s/ckpt.pt --out out-50k-R-s$s-passkey-long --finetune_iters 1500 --n_eval 100 --seed 0 --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn
  done
} 2>&1 | tee passkey_50k_long_R.log
{
  echo "########## R ceiling"
  python passkey.py --ckpt out-50k-R/ckpt.pt --out out-50k-R-passkey-5000 --finetune_iters 5000 --lr 1e-4 --n_eval 100 --seed 0 --eval_lengths 512 1024 2048 4096 8192 16384 2>&1 | grep -vi warn
} 2>&1 | tee passkey_50k_ceiling_R.log
echo "##### R finished $(date)"
