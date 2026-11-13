#!/bin/bash
# SmolLM-135M (public, RoPE, ctx 2048) adapted to content addresses with 500 steps of PG-19 books; evaluated on WikiText-103.
# Arms: untouched SmolLM (plain, NTK), adapted (offset c = 0, 1.5), B continued on the same books (plain, NTK).
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
until [ -f data/smollm_pg19/meta.pkl ]; do sleep 30; done
COMMON="--compile=False --use_flex=False --seed=1337"
L="--lengths 512 1024 2048 4096 8192 16384 --n_batches 50 --data_dir data/smollm_pg19"
P="--data_dir data/smollm_pg19 --train_lengths 512 1024 1536 2048 --max_train_len 2048 --eval_lengths 512 1024 2048 4096 8192 16384 --n_eval 50 --eval_before"
echo "##### smoke (10 iters adapt) $(date)"
python train.py config/smollm_adapt.py $COMMON --max_iters=10 --lr_decay_iters=10 --adapt_anneal_iters=10 --eval_interval=5 --eval_iters=2 --out_dir=out-smoke-smollm > logs_smollm/smoke.log 2>&1
if grep -aqE "Traceback|Error:|loss nan|loss: nan" logs_smollm/smoke.log; then echo "##### SMOKE FAILED $(date)"; exit 1; fi
grep -a "adapt: loaded\|^step" logs_smollm/smoke.log; echo "smoke clean"; rm -rf out-smoke-smollm
echo "##### untouched SmolLM: ppl $(date)"
echo "##### smollm-B plain"; python eval_lengths.py --ckpt out-smollm-B/ckpt.pt $L 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### smollm-B ntk";   python eval_lengths.py --ckpt out-smollm-B/ckpt.pt $L --ntk 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### untouched SmolLM: passkey $(date)"
python passkey.py --ckpt out-smollm-B/ckpt.pt --out out-smollm-B-passkey $P --ntk 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|ntk"
for c in smollm_adapt smollm_cont; do
  echo "##### train $c $(date)"
  python train.py config/$c.py $COMMON 2>&1 | tee logs_smollm/train_$c.log | grep -aE "^step [0-9]+: train loss|adapt: loaded|Error|Traceback"
done
echo "##### adapted: ppl $(date)"
for s in 0 1.5; do echo "##### adaptJm c=$s"; python eval_lengths.py --ckpt out-smollm-adaptJm-500/ckpt.pt $L --no_flex --cross_shift_c $s 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"; done
echo "##### control: ppl $(date)"
echo "##### Bcont plain"; python eval_lengths.py --ckpt out-smollm-Bcont-500/ckpt.pt $L 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### Bcont ntk";   python eval_lengths.py --ckpt out-smollm-Bcont-500/ckpt.pt $L --ntk 2>&1 | grep -aE "^\s+[0-9]+\s+[0-9.]+|Error|Traceback"
echo "##### adapted: passkey $(date)"
python passkey.py --ckpt out-smollm-adaptJm-500/ckpt.pt --out out-smollm-adaptJm-500-passkey $P --cross_shift_c 0 1.5 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|shift"
echo "##### control: passkey $(date)"
python passkey.py --ckpt out-smollm-Bcont-500/ckpt.pt --out out-smollm-Bcont-500-passkey $P --ntk 2>&1 | grep -aE "ctx=|before|after|Error|Traceback|ntk"
echo "##### smollm queue finished $(date)"
