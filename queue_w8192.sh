#!/bin/bash
# SmolLM frozen recipe adapted on 8192-token book sequences (window 2048). Evals: ppl, repeated-document test, passkey. Compare with out-smollm-window-500-farproj-frozen.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
COMMON="--compile=False --use_flex=False --seed=1337"
O=out-smollm-window8192-500-farproj-frozen
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/smollm_pg19"
PW="--data_dir data/smollm_pg19 --train_lengths 1024 2048 3072 4096 --max_train_len 4096 --tokens_per_step 4096 --window 2048 --eval_window 2048 --grad_checkpoint --freeze_pretrained --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50"
echo "##### smoke (5 iters) $(date)"
python train.py config/smollm_window8192_farproj_frozen.py $COMMON --max_iters=5 --lr_decay_iters=5 --eval_interval=5 --eval_iters=1 --out_dir=out-smoke-w8192 > logs_smollm/smoke_w8192.log 2>&1
if grep -aqE "Traceback|Error:|loss nan|loss: nan" logs_smollm/smoke_w8192.log; then echo "##### SMOKE FAILED $(date)"; tail -5 logs_smollm/smoke_w8192.log; exit 1; fi
grep -a "trainable\|^step\|^iter 5" logs_smollm/smoke_w8192.log | cut -c1-120; echo "smoke clean"; rm -rf out-smoke-w8192
echo "##### train $(date)"
python train.py config/smollm_window8192_farproj_frozen.py $COMMON 2>&1 | tee logs_smollm/train_smollm_window8192_frozen.log | grep --line-buffered -aE "^step [0-9]+: train loss|Error|Traceback" | cut -c1-200
echo "##### repeat test $(date)"; python repeat_test.py --ckpt $O/ckpt.pt --data_dir data/smollm_pg19 --gaps 1024 4096 8192 14000 --n 12 --no_flex --window 2048 --tag "SmolLM frozen, 8192-token adaptation" --out results_repeat/smollm_frozen_w8192.json 2>&1 | grep --line-buffered -vE "Warning|number of parameters|NVML"
echo "##### ppl (window 2048, c=0)"; python eval_lengths.py --ckpt $O/ckpt.pt $L --no_flex --window 2048 --cross_shift_c 0 2>&1 | grep --line-buffered -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### passkey $(date)"; python passkey.py --ckpt $O/ckpt.pt --out $O-passkey-ft4096 $PW --cross_shift_c 0 > logs_smollm/passkey_w8192_full.log 2>&1
grep -aE "restored|ctx=|Error|Traceback" logs_smollm/passkey_w8192_full.log | cut -c1-100
echo "##### w8192 queue finished $(date)"
