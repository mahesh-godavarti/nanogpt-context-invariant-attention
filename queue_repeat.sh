#!/bin/bash
# Repeated-document test on all three models: untouched (plain, NTK), window all-weights (seed 0), window frozen (seed 0); gaps 1024..14000, 12 trials.
cd /home/ubuntu/nanoGPT-unitmask
source /home/ubuntu/experiments/venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p results_repeat
G="--gaps 1024 4096 8192 14000 --n 12"
run () { python repeat_test.py --ckpt $1 --data_dir $2 $G $3 --tag "$4" --out results_repeat/$5.json 2>&1 | grep --line-buffered -vE "Warning|number of parameters|NVML"; }
echo "##### repeat test $(date)"
run out-smollm-B/ckpt.pt data/smollm_pg19 "" "SmolLM untouched" smollm_untouched
run out-smollm-B/ckpt.pt data/smollm_pg19 "--ntk" "SmolLM untouched+NTK" smollm_ntk
run out-smollm-window-500/ckpt.pt data/smollm_pg19 "--no_flex --window 2048" "SmolLM window all-weights" smollm_allw
run out-smollm-window-500-farproj-frozen/ckpt.pt data/smollm_pg19 "--no_flex --window 2048" "SmolLM window frozen" smollm_frozen
run out-pythia-B/ckpt.pt data/pythia_fineweb "" "Pythia-160M untouched" p160_untouched
run out-pythia-B/ckpt.pt data/pythia_fineweb "--ntk" "Pythia-160M untouched+NTK" p160_ntk
run out-pythia-window-500-splitlr/ckpt.pt data/pythia_fineweb "--no_flex --window 2048" "Pythia-160M window all-weights" p160_allw
run out-pythia-window-500-farproj-frozen-lr1e-4/ckpt.pt data/pythia_fineweb "--no_flex --window 2048" "Pythia-160M window frozen" p160_frozen
run out-pythia410-B/ckpt.pt data/pythia_fineweb "" "Pythia-410M untouched" p410_untouched
run out-pythia410-B/ckpt.pt data/pythia_fineweb "--ntk" "Pythia-410M untouched+NTK" p410_ntk
run out-pythia410-window-500-splitlr/ckpt.pt data/pythia_fineweb "--no_flex --window 2048" "Pythia-410M window all-weights" p410_allw
run out-pythia410-window-500-farproj-frozen-lr1e-4/ckpt.pt data/pythia_fineweb "--no_flex --window 2048" "Pythia-410M window frozen" p410_frozen
echo "##### repeat test finished $(date)"
