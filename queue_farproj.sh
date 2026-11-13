#!/bin/bash
# SmolLM window model with separate far-path projections, pretrained weights frozen. 500 steps on books; ppl (window 2048, c=0); passkey ft @4096.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
COMMON="--compile=False --use_flex=False --seed=1337"
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/smollm_pg19"
PW="--data_dir data/smollm_pg19 --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
O=out-smollm-window-500-farproj-frozen
echo "##### smoke (10 iters) $(date)"
python train.py config/smollm_window_farproj_frozen.py $COMMON --max_iters=10 --lr_decay_iters=10 --eval_interval=5 --eval_iters=2 --out_dir=out-smoke-farproj > logs_smollm/smoke_farproj.log 2>&1
if grep -aqE "Traceback|Error:|loss nan|loss: nan" logs_smollm/smoke_farproj.log; then echo "##### SMOKE FAILED $(date)"; tail -5 logs_smollm/smoke_farproj.log; exit 1; fi
grep -a "adapt: far\|adapt: pretrained\|^step\|new-parameter" logs_smollm/smoke_farproj.log; echo "smoke clean"; rm -rf out-smoke-farproj
echo "##### train farproj frozen $(date)"
python train.py config/smollm_window_farproj_frozen.py $COMMON 2>&1 | tee logs_smollm/train_smollm_window_farproj_frozen.log | grep -aE "^step [0-9]+: train loss|adapt: far|adapt: pretrained|Error|Traceback" | cut -c1-200
echo "##### ppl (window 2048, c=0) $(date)"; python eval_lengths.py --ckpt $O/ckpt.pt $L --no_flex --window 2048 --cross_shift_c 0 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### passkey $(date)"; python passkey.py --ckpt $O/ckpt.pt --out $O-passkey-ft4096 $PW --cross_shift_c 0 > logs_smollm/passkey_farproj_frozen_full.log 2>&1
grep -aE "ctx=|after|Error|Traceback" logs_smollm/passkey_farproj_frozen_full.log
echo "##### farproj queue finished $(date)"
