# Adapting a pretrained RoPE model (B) to content addresses, 2026-10-26

Start: the 5K-iteration pilot B checkpoint (6L/256d, ctx 1024, WikiText-103, delimiter 764 = " ."). Both runs below train 5000 more
iterations from that checkpoint with the same schedule (lr 2e-4, warmup 200, cosine to 2e-5, batch 8 x 8 accum), seed 1337.

- **adapted**: model Jm (two-score attention + content addresses). Query/key angle = pos_in_seg*f + (1-alpha)*seg_start*f; cross-segment
  keys get alpha * content address. alpha = 0 is B exactly (verified on CPU to 2e-7); alpha = 1 is Jm exactly. alpha starts at 0.1 and
  rises linearly to 1 by iteration 2500; the content map gets full-strength gradient throughout (straight-through on alpha).
  Config `config/adapt_B_to_Jm.py`, log `logs_vt/train_adapt_B_to_Jm.log`, checkpoint `out-wikitext103-small-adaptJm`.
- **B continued** (control): same checkpoint, same schedule, no addresses. `config/cont_B.py`, `out-wikitext103-small-Bcont`.

## Validation loss at the training length (1024)

| step | B continued | adapted (alpha) |
|---|---|---|
| 0 | 3.706 | 3.754 (0.10) |
| 1000 | 3.692 | 3.689 (0.46) |
| 2500 | 3.653 | 3.658 (1.00) |
| 4000 | 3.604 | 3.614 |
| 5000 | 3.619 | 3.600 |

Pilot models trained from scratch for 5K: B 3.721, Jm 3.708. The 0.03 bump at step 0 is the cost of alpha0 = 0.1; it is gone by step 500.
No bump anywhere during the anneal. At training length the adapted model and the control are the same within noise.

## Perplexity vs context length (val, 100 batches)

| length | B continued | B continued + NTK | adapted, c=0 | adapted, c=1.5 | pilot Jm (scratch), c=1.5 |
|---|---|---|---|---|---|
| 256 | 42.33 | 42.33 | 42.25 | 42.25 | 47.01 |
| 512 | 39.71 | 39.71 | 39.59 | 39.59 | 44.12 |
| 1024 | 36.58 | 36.58 | 36.47 | 36.47 | 40.60 |
| 2048 | 52.62 | 41.16 | 37.18 | 37.28 | 41.40 |
| 4096 | 82.73 | 49.44 | 36.86 | 35.97 | 39.71 |
| 8192 | 130.18 | 71.83 | 45.51 | 39.65 | 43.70 |

c = cross-segment offset (c * log(n_segments / 32), the standard Jm inference fix). The adapted model holds its 1024 perplexity out to
8x the training length; the continued B collapses past 1024 even with NTK. The adapted model is 4 points better than the scratch Jm at
every length (it has 10K total steps vs 5K).

## Passkey retrieval (fine-tune 300 steps on <= 1024, exact match, 50 trials)

| context | B continued | B continued + NTK | adapted (c=0 and c=1.5 identical) | pilot Jm (scratch), c=1.5 |
|---|---|---|---|---|
| 512 | .64 | .64 | .68 | .68 |
| 1024 | .62 | .62 | .60 | .60 |
| 2048 | .54 | .72 | .72 | .70 |
| 4096 | .00 | .38 | .54 | .60 |
| 8192 | .00 | .04 | .60 | .58 |

The adapted model retrieves at 8192 as well as at 512. The continued B cannot past 2048 (NTK gives a little at 4096, nothing at 8192).
The adapted model matches the scratch Jm on passkey. The offset changes the adapted model's logits (checked: up to 1.0 at the answer
positions) but not the argmax, so c=0 and c=1.5 give the same accuracy.

## Verdict

Annealing from the exact continuous-RoPE start works: a pretrained RoPE model gains content addressing in 5K steps with no loss at the
training length and reaches the from-scratch Jm's length behaviour. Not yet run: slow-band-only anneal (`adapt_slow_frac < 1`), more seeds,
a larger/longer pretrained model, the ALiBi-style recency bias.

# Short adaptation: 500 steps, alpha 0.1 -> 1 over all 500 (2026-10-27)

Same B checkpoint, config `config/adapt_B_to_Jm_500.py` (lr 2e-4, warmup 50, cosine to 2e-5 at 500), `out-wikitext103-small-adaptJm-500`.
Val@1024: 3.754 (step 0) -> 3.710 (400, alpha .82) -> 3.735 (500, alpha 1). B itself: 3.721.

| length | pilot B + NTK | adapted 500 steps, c=1.5 | adapted 5000 steps, c=1.5 | scratch Jm 5K, c=1.5 |
|---|---|---|---|---|
| 1024 | 40.49 | 41.34 | 36.47 | 40.60 |
| 2048 | 45.06 | 42.88 | 37.28 | 41.40 |
| 4096 | 51.90 | 41.28 | 35.97 | 39.71 |
| 8192 | 73.42 | 43.98 | 39.65 | 43.70 |

Passkey after the 300-step fine-tune (exact match, 50 trials):

| context | pilot B + NTK | adapted 500 steps | adapted 5000 steps | scratch Jm 5K |
|---|---|---|---|---|
| 512 | .62 | .64 | .68 | .68 |
| 1024 | .60 | .62 | .60 | .60 |
| 2048 | .64 | .70 | .72 | .70 |
| 4096 | .44 | .58 | .54 | .60 |
| 8192 | .30 | .62 | .60 | .58 |

500 steps (a tenth of the pretraining) gives the full length behaviour: flat perplexity to 8192 and passkey at 8192 equal to passkey at 512.
Cost at training length: +0.014 val loss over B (+0.85 ppl at 1024). The 5000-step run's lower numbers at every length come from the extra training.

# Public pretrained model: SmolLM-135M adapted on PG-19 books, evaluated on WikiText-103 (2026-10-27)

Setup: HuggingFaceTB/SmolLM-135M (LLaMA architecture, full per-head RoPE, pretrained context 2048, 600B tokens) loaded into this repo
(`GPT.from_hf_llama`, logits match Hugging Face to 8e-5). Fine-tuning text: 113 PG-19 books, 12.1M tokens (SmolLM tokenizer), not the
pretraining data. Evaluation: WikiText-103 validation + test (568K tokens). Delimiters: '.' (30) and ' .' (1673). Queue: queue_smollm.sh.

Loss shift BEFORE any training (4 WikiText windows of 2048, CPU), untouched model 2.903:

| alpha | all bands, random map | all bands, zero map | slowest half, zero map | slowest quarter, zero map |
|---|---|---|---|---|
| 0.1 | 3.039 | 3.024 | | |
| 0.3 | 3.477 | 3.371 | 2.916 | 2.907 |
| 1.0 | 7.599 | 5.546 | 4.576 | 3.104 |

The pretrained model is far more sensitive to the angle interpolation than our own 5K-step B was (+0.12 at alpha 0.1 vs +0.03). Annealing
only the slowest quarter of the bands costs just +0.20 at alpha 1 before training. The queued run anneals all bands with the zero map
(adapt_zero_addr) over 500 steps; the slow-quarter variant is the natural next run.

## SmolLM results, all-bands anneal, zero map, 500 steps (2026-10-27)

Val loss at 2048 (WikiText) during the runs: untouched 2.903. Adapted: 2.968 (step 0, alpha .1), 2.997 (200, .46), 3.117 (300, .64),
3.101 (400, .82), 3.345 (500, 1.0). Control (books, no addresses): 2.868 (0) -> 2.994 (500). So of the adapted model's +0.44 over the
untouched model, +0.13 is the books domain shift (the control pays it too) and +0.31 is the anneal, most of it in the last 100 steps.

Perplexity on WikiText-103 (50 batches):

| length | untouched | untouched + NTK | control (books) | control + NTK | adapted, c=0 | adapted, c=1.5 |
|---|---|---|---|---|---|---|
| 512 | 23.90 | 23.90 | 27.87 | 27.87 | 31.90 | 31.90 |
| 1024 | 20.57 | 20.57 | 23.51 | 23.51 | 28.96 | 28.87 |
| 2048 | 17.74 | 17.74 | 20.13 | 20.13 | 29.17 | 27.49 |
| 4096 | 116.54 | 22.86 | 133.58 | 25.94 | 38.22 | 30.84 |
| 8192 | 474.02 | 42.07 | 534.53 | 47.44 | 49.89 | 30.88 |
| 16384 | 1969.89 | 98.37 | 2344.51 | 111.00 | 85.90 | 35.49 |

The adapted model (offset 1.5) rises only 8 points from 2048 to 16384 (8x its pretrained context); NTK on the control rises 91. Crossover
between 4096 and 8192: NTK wins at 4096 (25.9 vs 30.8), the adapted model wins at 8192 (30.9 vs 47.4) and 16384 (35.5 vs 111). Cost at
the training length: 27.5 vs the control's 20.1. Passkey: below.

Passkey after the 300-step passkey fine-tune (exact match, 50 trials). Queue finished 11:49 UTC.

| context | untouched | untouched + NTK | control (books) | control + NTK | adapted, c=0 | adapted, c=1.5 |
|---|---|---|---|---|---|---|
| 512 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1024 | 1.00 | 1.00 | 1.00 | 1.00 | .98 | .98 |
| 2048 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4096 | .00 | .02 | .00 | .00 | .98 | 1.00 |
| 8192 | .00 | .00 | .00 | .00 | .90 | .96 |
| 16384 | .00 | .00 | .00 | .00 | .20 | .98 |

Before the passkey fine-tune the adapted model (c=1.5) already retrieved .34 at 2048 and .08 at 4096; after it, .98 at 16384 (8x the
pretrained context) against 0 for every RoPE arm, NTK included. The offset matters at 16384 (.20 -> .98): without it the cross-segment
mass is spread over too many sentences.

Verdict: a public pretrained model gains content addressing from 500 steps on 8M tokens of unrelated text. Retrieval at 8x context goes
from 0 to .98 and perplexity at 16384 from 98 (NTK) to 35. Open cost: +7 ppl at the training length from the all-bands anneal; the
slow-quarter anneal (config/smollm_adapt_slow25.py, not run) is the proposed fix.

## SmolLM, slow-quarter anneal (adapt_slow_frac 0.25, zero map, 500 steps), 2026-10-27

Val loss at 2048 at step 500: 3.046 (control 2.994, all bands 3.345) -- the training-length cost is nearly gone.
But perplexity collapses past 2048 (c=0 / c=1.5): 2048 21.1 / 21.3; 4096 102 / 102; 8192 287 / 301; 16384 565 / 652.
Reason: SmolLM's 32 planes per head have wavelengths 6 .. 63,000 tokens; 11 of them never complete a cycle in 2048 and 23 complete fewer
than 32. The slowest quarter (8 planes, wavelength > 6,300) was converted; planes with wavelengths 630 .. 4,700 stayed on continuous RoPE
and go out of range past 2048 on their own. Pre-training shift at alpha 1 by fraction converted (zero map): 8 planes 3.10, 12 planes 4.53,
16 planes 4.58, 20 planes 4.93, 24 planes 5.26, 32 planes 5.55 (untouched 2.90). The jump from 8 to 12 planes shows the model leans
heavily on the planes with wavelengths around 1,500 .. 4,700 -- the very ones that break past 2048.
Options: (a) NTK-scale the unconverted planes at inference on the slow-quarter checkpoint (no training; code added to the adapt path);
(b) convert ~12 planes (frac .375) and let the anneal repair the 4.5 shift; (c) the queued 5000-step all-bands anneal.

## SmolLM, 5000-step all-bands anneal (80M tokens of books, alpha .1 -> 1 over all 5000), 2026-10-27

Val loss at 2048 (WikiText): 2.968 (0) 2.987 (500) 3.130 (1000) 3.160 (1500) 3.225 (2000) 3.294 (2500) 3.260 (3000) 3.277 (3500)
3.358 (4000) 3.399 (4500) 3.500 (5000). Train loss on books fell 3.43 -> ~3.0. The slow anneal did not reduce the training-length cost:
it ended 0.15 ABOVE the 500-step all-bands run (3.345). Part is the books domain shift accumulating over 80M tokens (no 5000-step
books-only control was run, so the split is unknown); the steady rise with alpha says the anneal itself is the larger part. Evals pending.
5000-step run passkey (after fine-tune, c=1.5): 1.00 / 1.00 / 1.00 / 1.00 / .98 / .96 at 512..16384 (c=0: .84 at 16384). Before the
passkey fine-tune it already retrieved .84-.94 inside 2048 and .50 at 4096. Retrieval is as good as the 500-step run; perplexity is worse.

## Window mode (user 2026-10-28): near keys (< window) scored as pretrained, far keys by reset-RoPE + address, learned far bias (init -4)
Train: window 1024 on 2048-token book sequences, 500 steps (so far keys exist in training); evaluate with window 2048 (exact within the
pretrained context). CPU checks: equals the untouched model within the window (7e-5 on real SmolLM at 2048); equals sliding-window
attention when far keys are suppressed (1-layer, 1e-7); causal; far bias and address map trained; window override at eval works.
Queue: queue_smollm_window.sh -> out-smollm-window-500.

### Window mode results (train window 1024 @ 2048, 500 steps on books; eval window 2048), 2026-10-28

Perplexity on WikiText-103 (window 2048, c=0 / c=1.5): 512 27.2; 1024 23.1; 2048 19.80; 4096 20.79 / 21.58; 8192 19.28 / 20.34;
16384 19.24 / 20.47. Flat from 2048 to 16384 with NO offset needed. At 2048 it equals the books-only control (20.13): the whole cost
is the books domain shift, none from the attention change. Every other arm at 16384: untouched+NTK 98, control+NTK 111, all-bands
anneal 35.5, slow-quarter 652.
Passkey before the passkey fine-tune (c=0): 1.00 / .98 / .92 / .32 / .08 / .00 -- already the best "before" retrieval at 4096 of any arm.
The after-fine-tune passkey did not run: the passkey fine-tune at tokens_per_step 4096 (batch 2 at 2048) OOMs in the window path (T4,
batch 2 needs > 15 GB) and the NVML-mismatch assert hid the error. Rerun with batch 1 queued (queue_window_passkey.sh).
Window model passkey (fine-tune with window 2048 on <=2048 sequences, batch 1; eval window 2048), exact / first token, c=0:
512 1.00/1.00; 1024 1.00/1.00; 2048 1.00/1.00; 4096 .52/1.00; 8192 .18/1.00; 16384 .08/.96. The key sentence is FOUND every time
(first digit right at 16K in 96-100% of trials) but the remaining digits are not copied. Likely cause: with window 2048 the passkey
fine-tune (sequences <= 2048) never has a far key, so the far path never sees the copying task; the all-bands model has no such split.
Rerun queued: fine-tune with window 1024 (the training recipe), evaluate with window 2048 (queue_window_passkey.sh, -w1024 outputs).
Window model passkey, fine-tune with window 1024 / eval window 2048 (c=0), exact / first token: 512 1.00; 1024 1.00; 2048 1.00;
4096 .92/.98; 8192 .94/1.00; 16384 .70/.96. With c=1.5 it is worse (4096 .72, 8192 .18): the learned far bias already sets the far
mass, and the offset on top starves it -- use c=0 in window mode. Verdict: window mode = untouched model inside 2048, flat ppl 19-21
to 16384 with no offset, passkey .70-.94 at 4-8x context. The remaining gap to the all-bands model's .98 at 16384 is the one open item.
Window model passkey, fine-tune with window 512 / eval window 2048 (c=0), exact: 512 1.00; 1024 1.00; 2048 1.00; 4096 .92; 8192 .96;
16384 .82 (first token 1.00 everywhere). Window 1024 fine-tune gave .70 at 16K, window 2048 gave .08: more and farther far keys in the
fine-tune -> better far copying. Supports the theory; the proper test (fine-tune on 4096-token sequences, needs activation checkpointing)
is the next step if .82 is not enough.
Window model passkey, fine-tune on sequences up to 4096 with window 2048 (activation checkpointing; far keys at real distances),
eval window 2048, c=0, exact: 512 1.00; 1024 1.00; 2048 1.00; 4096 1.00; 8192 1.00; 16384 .90 (first token 1.00 everywhere).
Progression at 16K by fine-tune setting: window 2048 @2048 .08 -> window 1024 @2048 .70 -> window 512 @2048 .82 -> window 2048 @4096 .90.
Window model final (1 seed): untouched inside 2048; ppl 19.8 / 20.8 / 19.3 / 19.2 at 2048..16384; passkey 1.00 / 1.00 / .90 at 4K/8K/16K.

### Window model, seeds (same recipe; passkey fine-tune @4096 window 2048, c=0)
seed 1: ppl 27.17 / 23.03 / 19.76 / 20.73 / 19.24 / 19.31 at 512..16384 (seed 0: 27.23 / 23.07 / 19.80 / 20.79 / 19.28 / 19.24);
        passkey 1.00 / 1.00 / 1.00 / .94 / .94 / .48 (seed 0: 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / .90). 16K retrieval varies by seed.
seed 2: ppl 27.19 / 23.02 / 19.74 / 20.78 / 19.36 / 19.53; passkey 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / .98.
Three seeds, window model: ppl at 2048 19.74-19.80, at 16384 19.24-19.53; passkey 4K .94-1.00, 8K .94-1.00, 16K .48 / .90 / .98 (mean .79).

## Pythia-160M (second public model; GPT-NeoX, partial rotary 8/32 planes; fine-tune on FineWeb-Edu, eval WikiText-103), 2026-10-29
Untouched: ppl plain 36.5 / 30.2 / 27.6 / 105.7 / 764.9 / 1585.7 at 512..16384; +NTK 36.5 / 30.2 / 27.6 / 25.8 / 33.5 / 73.8.
Passkey after fine-tune: plain 1.00 to 2048 then 0; +NTK .96 at 4096, .68 at 8192, 0 at 16384 (NTK works much better for Pythia than
for SmolLM; its passkey is 2 tokens under its tokenizer).
First window run INVALID: step-0 loss 10.9 (CPU fp32 gives 3.58) -> fp16 overflow of the raw q.k product in the chunked window path
(Pythia's attention scores are large; HF scales inside the matmul). Fixed by scaling q before the product. Rerun queued.
Pythia window run (valid; after the fp16-score and trainer-architecture fixes), train window 1024 @2048 500 steps on FineWeb-Edu, eval
window 2048, c=0: ppl 46.6 / 38.8 / 35.6 / 31.8 / 31.7 / 30.5 at 512..16384 (untouched 36.5 / 30.2 / 27.6 / 105.7 / 764.9 / 1585.7;
untouched+NTK 25.8 / 33.5 / 73.8 at 4096+). Passkey (ft @4096, c=0): .98 / .92 / .88 / .90 / .92 / .92 (untouched+NTK .96 / .68 / .00 at 4K+).
Flat to 16K and retrieval .92 at 8x context with only 8 address planes per head. Cost at 2048: +8 ppl (27.6 -> 35.6); within-window
passkey also fell to .88-.92. Inside the window the model is the fine-tuned Pythia, so this is the FineWeb-Edu fine-tune itself (or lr
1e-4 too high for Pythia), not the attention change: needs the no-address control (queued: queue_pythia_control.sh).
Pythia control (500 steps FineWeb-Edu, no addresses): ppl plain 45.5 / 37.7 / 34.5 / 129 / 961 / 1900; +NTK 31.2 / 38.0 / 76.0 at 4096+.
Passkey +NTK: .94 / .94 / .94 / .98 / .98 / .38. So the window model's 35.6 at 2048 is the fine-tune (control 34.5), not the attention
change; both lose ~.06 of within-window passkey from the fine-tune too. Window vs control+NTK: ppl 8K 31.7 vs 38.0, 16K 30.5 vs 76.0;
passkey 16K .92 vs .38. NTK on fine-tuned Pythia retrieves at 8K (.98) -- Pythia's 2-token key and its stronger NTK behaviour.
Pythia at lr 2e-5 (5x lower). Control: ppl plain 39.1 / 32.5 / 29.5 / 121.7 / 978.7 / 2026.5; +NTK 27.0 / 33.3 / 69.4 at 4096+;
passkey +NTK .96 / .92 / .92 / .98 / .88 / .00. Window: ppl 40.1 / 33.3 / 30.4 / 27.5 / 27.6 / 27.5; passkey .96 / .98 / .94 / .94 / .88 / .56.
Lower lr cut the fine-tune cost at 2048 from 7 to 2 ppl (control 29.5 vs untouched 27.6; window 30.4), kept flat ppl to 16K (27.5 vs
control+NTK 69.4), but the far path learned less: 16K passkey .56 (vs .92 at lr 1e-4). The pretrained weights want a low lr, the new
address map and far biases want a high one -> next: two learning rates (pretrained 2e-5, new params 1e-3).
Pythia window, two learning rates (pretrained 2e-5, new params 1e-3), seed 0: ppl 40.4 / 33.5 / 30.6 / 27.4 / 27.3 / 26.4 at
512..16384 (control lr 2e-5: 29.5 at 2048, +NTK 69.4 at 16384); passkey .96 / .84 / .94 / .86 / .82 / .84 (first token .96-1.00).
Keeps the low-rate base (30.6 vs 30.4) and recovers far retrieval (16K .84 vs .56 at one low rate, .92 at one high rate with a 7-ppl
base cost). This is the Pythia recipe; seeds 1 and 2 queued (queue_pythia_seeds.sh).
Pythia window, two rates, seed 1: ppl 39.1 / 32.5 / 29.7 / 26.6 / 26.4 / 25.5; passkey .92 / .86 / .78 / .88 / .94 / .86.
                            seed 2: ppl 39.6 / 32.9 / 30.1 / 27.0 / 26.9 / 25.9; passkey .98 / .92 / .98 / .92 / .78 / .86.
Three seeds: ppl at 2048 29.7-30.6 (control 29.5, untouched 27.6), at 16384 25.5-26.4 (control+NTK 69.4); passkey 4K .86-.92,
8K .78-.94, 16K .84-.86 (control+NTK 16K .00).

## Window model with separate far-path q/k projections, pretrained weights FROZEN (SmolLM, 2026-10-30)
Trainable: far q/k projections (copies of the pretrained ones at init), address map, far biases = 13.4M of 148M; lr 1e-3 for them.
ppl (window 2048, c=0): 23.90 / 20.57 / 17.74 / 18.47 / 17.02 / 16.63 at 512..16384 -- inside 2048 identical to the untouched
model to the last digit (23.90 / 20.57 / 17.74), beyond it BELOW the 2048 number (the far context helps). Passkey (ft @4096):
1.00 / 1.00 / 1.00 / .98 / 1.00 / .94. Best arm so far: no fine-tune cost at all, best long-context ppl, retrieval .94 at 16K.
Seeds 1 and 2 queued (queue_farproj_seeds.sh).
Frozen + far projections, seed 1: ppl 23.90 / 20.57 / 17.74 / 18.43 / 17.00 / 16.57; passkey 1.00 / 1.00 / 1.00 / .96 / 1.00 / .96.
                            seed 2: ppl 23.90 / 20.57 / 17.74 / 18.47 / 17.03 / 16.73; passkey 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / .96.
Three seeds: ppl 17.74 at 2048 (= untouched) in all; 16.57-16.73 at 16384; passkey 16K .94 / .96 / .96, 4K-8K .96-1.00.

## Structured-data adaptation (STRUCT_DESIGN.md), SmolLM frozen, 2026-10-31, seed 0
16 facts per context (3 share the target's head name), 8 relations, answer = single-token name; hit@1 over 500 test contexts.
Untouched SmolLM, facts as sentences in context (arm A): new triples .21, new names .29. Bare name pairs (arm C): .00 / .00.
Adapted (arm B, facts as two-token UNITS through the far path, no relation words; 2000 steps, batch 32, 9.4 min): .68 / .55.
Learning curve (new triples / new names): 250: .38/.37; 500: .40/.41; 1000: .49/.48; 1500: .58/.47; 2000: .62/.52 (final 500-context
eval .68/.55) -- still rising. At K=64 facts (trained at 16): B .15, A .08.
Verdict: the frozen model reads facts supplied only as structured units, 3x better than it reads the same facts as sentences.
Open: more steps (curve not saturated), train on mixed K for large contexts, seeds.
Structured, 5000 steps with training contexts of 16-64 facts (mixed K): hit@1 at K=16 .51 / .44 (new triples / new names), at K=64
.144 -- WORSE than the 2000-step K=16-only run (.68 / .55; K=64 .146). Mixed sizes slow the learning and do not help K=64.
Curve at K=16: 500 .24, 2000 .36, 4000 .43, 5000 .46 (still rising). Diagnosis: the content address is pooled from the unit's token
embeddings rotated only by the in-slot offset (0 for both tokens), so it is symmetric in the two names and carries neither the
relation nor which name is the head; selection among many units then rests on the key rotation alone. Fix to try: rotate by the
slot and relation angles BEFORE pooling (rotate-then-pool, as in the LIS J models), so the address encodes role and relation.
Structured, ROTATE-THEN-POOL address (slot + relation angles applied before pooling), 10000 steps at K=16, seed 0 (44 min):
hit@1 new triples .994, new names .946, K=64 (trained at 16) .898. Untouched model reading the same facts as sentences: .21 / .29,
K=64 .076. Curve: 1000 .50/.43/.06; 2000 .69/.69/.16; 3000 .93/.87/.50; 4000 .98/.93/.69; 7000 .99/.93/.94; 10000 .996/.94/.85.
The address fix was the whole story: the role- and relation-aware address lets the query pick one unit out of 64 at 90%.
Structured rotate-then-pool, seeds 1 / 2: new triples .990 / .994, new names .942 / .952, K=64 .870 / .890.
Three seeds: new triples .990-.994, new names .942-.952, K=64 (trained at 16) .870-.898; sentences-in-context baseline .21 / .29 / .08.

## Pythia-410M, frozen + far projections, 500 steps on FineWeb-Edu (2026-11-01) -- NEGATIVE at this budget
Untouched: ppl 22.0 / 18.0 / 16.3 / 98.4 / 1365.9 / 7263.3 at 512..16384; +NTK 15.1 / 18.1 / 45.5 at 4096+.
Untouched passkey: plain .92 / .96 / .80 then 0; +NTK .74 / .26 / .02 at 4096+.
Frozen window (50.9M trainable, far bias -6, 36 s/iter, 5 h): ppl 22.0 / 18.0 / 16.3 (= untouched, as designed) then 21.9 / 26.1 / 28.2
-- not flat; worse than NTK at 4096 and 8192, better only at 16384 (28 vs 45). Passkey .78 / .80 / .80 / .24 / .10 / .08 -- the far
path barely learned (train val loss 3.18 -> 3.13 over 500 steps) and the all-weights passkey fine-tune lowered even the within-window
score (.80-.96 -> .78-.80). Open questions: frozen recipe on Pythia at all (160M frozen not yet run; 160M all-weights worked), far bias
-6 starving the far path's gradient, 500 steps too few for 51M new parameters, passkey fine-tune should keep the pretrained weights frozen.

## Pythia-160M, frozen + far projections, far bias -4, 500 steps; passkey fine-tune with pretrained weights frozen (2026-11-02) -- NEGATIVE
ppl 36.5 / 30.2 / 27.6 (= untouched) / 37.5 / 52.9 / 67.7 (untouched+NTK 25.8 / 33.5 / 73.8; all-weights two-rate window model 27 / 27 / 26).
Passkey .86 / .90 / .74 / .00 / .00 / .00. The frozen recipe does not learn the far path on Pythia-160M either; the all-weights
recipe did (.84-.86 at 16K). So "frozen on Pythia" fails, independent of size.
Check (2026-11-03): the frozen Pythia-160M window model's logits inside 2048 equal the untouched model's (and ours equal Hugging Face's
token for token on the passkey trials), so its within-window passkey (.86 / .90 / .74) IS the untouched model's pre-fine-tune score
under the current evaluation. The untouched run of 10-29 logged .98 / 1.00 / 1.00 "before fine-tuning" for the same trials; the cause
of that discrepancy was not found (same checkpoint, data, seed, options). Tables use after-fine-tune numbers, which are unaffected.
Conclusion on the frozen recipe: works on SmolLM (3 seeds), fails to train the far path on Pythia-160M and -410M at 500 steps.
The all-weights two-rate recipe worked on Pythia-160M (3 seeds) -> use it for Pythia-410M.

RESOLVED (2026-11-03): the "discrepancy" above was a misreading. The untouched Pythia-160M passkey run of 10-29 had no --eval_before;
its two blocks were AFTER fine-tuning (plain, NTK). Re-evaluating the saved fine-tuned checkpoint (out-pythia-B-passkey) on CPU today
gives .98 / 1.00 at 512 / 2048, as logged. The untouched model's PRE-fine-tune score inside 2048 is .86 / .90 / .74, which is what the
frozen-base passkey fine-tune (near path unchangeable) reproduces. All tables stand.

## Pythia-410M, all-weights two-rate window recipe (pretrained 2e-5, new params 1e-3), 500 steps on FineWeb-Edu (2026-11-03)
Training val (window 1024): 3.26 -> 2.92. ppl (window 2048, c=0): 24.2 / 19.9 / 18.1 / 16.45 / 16.39 at 512..8192 (untouched 22.0 / 18.0 /
16.3 / 98.4 / 1365.9; +NTK 15.1 / 18.1 at 4096 / 8192). Flat and falling with length; +1.8 at 2048 = fine-tune cost. 16384 eval crashed
(NVML-masked, likely OOM) -> rerun alone. Passkey after the all-weights fine-tune: ZERO everywhere incl. 512 (first token .04-.14) ->
the 405M all-weights passkey fine-tune broke the model (see passkey log); needs the frozen-base passkey fine-tune or a lower lr.
410M window 16384 (rerun alone, same seed/windows): 16.08 (untouched+NTK 45.5). Full row: 24.2 / 19.9 / 18.1 / 16.45 / 16.39 / 16.08.
The in-queue crash at 16384 was transient memory (the eval process had just run the shorter lengths); alone it fits.
410M passkey, fine-tune at lr 1e-5 (stable; loss 0.0002-0.01): 1.00 / 1.00 / .98 / 1.00 / .92 / .86 at 512..16384
(untouched+NTK after fine-tune .92 / .96 / .80 / .74 / .26 / .02). Pythia-410M all-weights: DONE, 1 seed.

## Frozen recipe on Pythia-160M: new-parameter learning-rate sweep (2026-11-03), 500 steps, far bias -4, frozen-base passkey fine-tune
| new-param lr | train val (w1024) 0 -> 500 | ppl 4096 / 8192 / 16384 (window 2048) | passkey 4K / 8K / 16K |
| 3e-3 | 3.86 -> 5.55 (diverged) | 340 / 1101 / 2039 | 0 / 0 / 0 |
| 1e-3 (earlier) | 3.86 -> 3.67 | 37.5 / 52.9 / 67.7 | 0 / 0 / 0 |
| 3e-4 | 3.86 -> 3.45 | 28.5 / 30.6 / 30.7 | .36 / .16 / .08 |
Reference: untouched+NTK 25.8 / 33.5 / 73.8; all-weights two-rate window model 27 / 27 / 26, passkey .86-.92 / .78-.94 / .84-.86.
The frozen far path on Pythia is rate-sensitive: 1e-3 (fine on SmolLM) is already too high here; 3e-4 trains and is flat-ish to 16K
but retrieval is weak at 500 steps. Next: 1e-4, and 3e-4 for 2000 steps.
| 1e-4 | 3.86 -> 3.35 | 25.9 / 26.4 / 26.3 | .98 / .94 / .96 |   <-- WORKS (2026-11-03): = SmolLM's frozen result; below untouched 27.6 at 2048
| 3e-4, 2000 steps | 3.86 -> 3.45 | 28.1 / 29.8 / 29.9 | (passkey pending) |
Frozen recipe on Pythia-160M at new-param lr 1e-4: untouched inside 2048 (27.63 exact), 25.9 / 26.4 / 26.3 beyond, passkey .98 / .94 / .96
with the base frozen even in the passkey fine-tune. The failure at 1e-3 was the learning rate. SmolLM tolerated 1e-3; Pythia needs 1e-4.
3e-4 x 2000 steps passkey: .24 / .14 / .08 (worse than 1e-4 x 500). Rate, not duration, is the knob.

## Pythia-410M, FROZEN recipe at new-param lr 1e-4, far bias -4, 500 steps (2026-11-04) -- WORKS
ppl (window 2048, c=0): 22.0 / 18.0 / 16.32 (= untouched exactly) / 15.45 / 15.86 / 16.60 at 512..16384 (untouched+NTK 15.1 / 18.1 / 45.5
at 4096+). Passkey (base frozen in the fine-tune too): .84 / .86 / .76 inside (= untouched pre-fine-tune behaviour) / 1.00 / 1.00 / .94.
Train val (w1024) 3.38 -> 2.79. 50.9M new parameters, 2.3 h training. Frozen recipe now shown on SmolLM-135M (3 seeds), Pythia-160M (1),
Pythia-410M (1): untouched inside the context, below its own 2048 perplexity to 8x, retrieval .94-.96 at 8x.
Frozen Pythia-160M lr 1e-4, seeds 1 / 2 (2026-11-04): ppl 25.95 / 26.36 / 26.25 and 26.12 / 26.52 / 26.28 at 4K/8K/16K (seed 0: 25.9 / 26.4 / 26.3)
-- identical across seeds. Passkey seed 1: 1.00 / .92 / .96; seed 2: .42 / .20 / .12 (first token .46 / .26 / .20). Seed 2's LM far path
is as good as the others (ppl), but its passkey fine-tune (frozen base, 300 iters, lr 5e-5, far params only) did not produce copying.
Rerunning seed 2's passkey fine-tune with a different fine-tune seed and 600 iters to separate "adaptation seed" from "fine-tune luck".
Seed 2 passkey rerun (fine-tune seed 12, 600 iters, loss ends 0.95 with no spike): 1.00 / .98 / .96 at 4K/8K/16K.
Frozen Pythia-160M, 3 seeds: ppl 25.9-26.1 / 26.4-26.5 / 26.25-26.3; passkey .98-1.00 / .92-.98 / .96. The seed-2 failure was the
passkey fine-tune's last-step spike, not the adaptation.

## Structured-data adaptation on Pythia-160M (frozen base, far projections, rotate-then-pool, new-param lr 1e-4, 10000 steps, 2026-11-04), seed 0
238 single-token names (46 held out). Untouched Pythia reading the facts as sentences: .126 (new triples) / .22 (new names) / .044 at K=64;
bare pairs 0. Adapted, units: .98 / .71 / .91 (K=64, trained at 16). Curve (triples/names/K64): 1000 .21/.17/.05; 3000 .63/.47/.34;
5000 .92/.59/.78; 7000 .97/.67/.89; 10000 .98/.72/.93. New-name generalisation weaker than SmolLM's (.71 vs .95) and still rising.
Structured Pythia-160M seeds 1 / 2: .984 / .748 / .928 and .966 / .700 / .886. Three seeds: new triples .97-.98, new names .70-.75,
K=64 .89-.93; untouched reading sentences .13 / .22 / .04. Done.
Pythia-410M frozen lr 1e-4, seeds 1 / 2 (2026-11-05): ppl 15.51 / 15.90 / 16.29 and 15.54 / 15.94 / 16.46 at 4K/8K/16K (seed 0 15.45 / 15.86 /
16.60); passkey 1.00 / 1.00 / 1.00 both (seed 0 1.00 / 1.00 / .94). Three seeds: exact inside 2048 (16.32), 15.5 / 15.9 / 16.3-16.6
beyond, passkey .94-1.00 at 16K. Frozen recipe: 3 seeds on all three models.
Pythia-410M all-weights seed 1: ppl 24.4 / 20.1 / 18.4 / 16.6 / 16.6 / 16.2 (seed 0 18.1 / 16.5 / 16.4 / 16.1 at 2048+); passkey pending.
Pythia-410M all-weights two-rate, seeds 1 / 2 (2026-11-05): ppl 18.36 / 16.58 / 16.61 / 16.21 and 18.11 / 16.39 / 16.33 / 16.00 at 2048..16384
(seed 0 18.10 / 16.45 / 16.39 / 16.08); passkey 1.00 at every length 512..16384 in both seeds (seed 0 1.00 / .92 / .86 at 4K/8K/16K with
the old last-step checkpoint; the lowest-loss restoration was not in place then). Three seeds on every recipe and model. ALL DONE.

## Structured-data adaptation on Pythia-410M (frozen base, lr 1e-4, 10000 steps, 2026-11-05), 3 seeds
Untouched 410M reading the facts as sentences: .448 / .47 / .164 (new triples / new names / K=64); bare pairs 0.
Adapted, units: seed 0 1.000 / .976 / .986; seed 1 1.000 / .978 / 1.000; seed 2 1.000 / .980 / .994. Reached .99 / .95 / .92 by step 2000.
Structured task: 3 models x 3 seeds. Program complete.

## Repeated-document test (repeat_test.py, 2026-11-06): [300-token WikiText passage][gap filler][same passage]; score the 2nd copy
12 trials per gap, paired across models. Columns: 2nd-copy ppl / token accuracy; "unseen" = same passage with no first copy (ppl 20-45).
| model, setting | 1024 | 4096 | 8192 | 14000 |
| SmolLM untouched | 1.19/.97 | 1788/.07 | 2562/.06 | 5325/.06 |
| SmolLM +NTK | 1.19/.97 | 191/.18 | 822/.08 | 1246/.08 |
| SmolLM window all-weights | 1.32/.95 | 8.65/.55 | 9.55/.55 | 11.8/.52 |
| SmolLM window frozen | 1.19/.97 | 10.2/.53 | 10.3/.54 | 12.2/.50 |
| Pythia-160M untouched | 1.29/.95 | 1809/.03 | 1894/.04 | 1871/.03 |
| Pythia-160M +NTK | 1.29/.95 | 19.6/.45 | 46.4/.34 | 291/.17 |
| Pythia-160M window all-weights | 1.56/.92 | 19.6/.46 | 24.4/.44 | 21.4/.45 |
| Pythia-160M window frozen | 1.30/.95 | 24.0/.40 | 29.2/.40 | 28.3/.40 |
| Pythia-410M untouched | 1.12/.98 | 7263/.02 | 45306/.00 | 41606/.00 |
| Pythia-410M +NTK | 1.12/.98 | 7.36/.60 | 151/.21 | 477/.12 |
| Pythia-410M window all-weights | 1.22/.97 | 27.3/.39 | 28.9/.39 | 26.0/.41 |
| Pythia-410M window frozen | 1.12/.98 | 3.94/.71 | 6.16/.63 | 6.44/.62 |
Reading: inside the window (1024) every model copies (.95-.98). Beyond it, the window models copy PARTIALLY -- 40-71% of tokens, ppl
well below the unseen level -- and the rate is flat from 4K to 14K; untouched collapses; NTK copies at 4K (esp. 410M: .60) and decays
with distance. Best: frozen 410M at .62-.71 to 14K. The far path retrieves real prose at a distance but copies it approximately; the
near path copies exactly. (Perplexities above ~20 for Pythia-160M all-weights/frozen at 4K reflect the unseen level of 35-45 there.)

## SmolLM frozen, adapted on 8192-token sequences with window 2048 (2026-11-10) -- the far-copy gap is NOT exposure
500 steps x 2 x 8192 tokens (same 16K tokens/step), 13.4M trainable, 5.7 h (265 s/iter: the chunked window path at 8192 is slow).
Repeated-document token accuracy (2nd copy) at 4096 / 8192 / 14000: .48 / .50 / .49 (2048-trained frozen: .53 / .54 / .50). ppl: 17.74 (exact)
/ 18.37 / 16.62 / 15.70 at 2048..16384 (2048-trained: 17.74 / 18.47 / 17.02 / 16.63). Passkey .86 / .92 / .80 (2048-trained .98 / 1.00 / .94).
Training on natural long-range repeats did not raise far copying; slightly better long-range ppl, slightly worse passkey. The 40-70%
far-copy ceiling is in the mechanism (reading off a far unit token by token), not in training exposure.
