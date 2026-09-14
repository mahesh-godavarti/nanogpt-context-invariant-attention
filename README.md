# nanogpt-context-invariant-attention

A fork of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy that replaces learned position embeddings with **segment-reset RoPE plus content-addressed cross-segment attention**, implemented with PyTorch's [FlexAttention](https://pytorch.org/blog/flexattention/) API.

Paper: Mahesh Godavarti, *Content-based addressing for long context* — [arXiv:2609.07314](https://arxiv.org/abs/2609.07314).

Standard transformers assign every token a position from 0 to T-1, so a model trained at length 1024 meets positions it has never seen when evaluated at 4096. Here the position restarts at 0 after every delimiter token (a sentence or paragraph boundary), so no position ever exceeds the longest segment seen in training, whatever the context length. Attention across segments is routed by a per-segment content address on the keys instead of by absolute position. The result is perplexity that degrades slowly beyond the training context instead of collapsing.

## What's different from nanoGPT

| | nanoGPT | This repo |
|---|---------|-----------|
| **Positional encoding** | Learned absolute position embeddings | RoPE on Q and K. Model B: continuous positions. Models Jm/Jmr: position resets to 0 after every delimiter |
| **Attention** | `F.scaled_dot_product_attention(is_causal=True)` | Model B: same. Models Jm/Jmr: two-score segment attention via FlexAttention |
| **Segment awareness** | None | Segments detected from a configurable delimiter token |
| **Context extrapolation** | Degrades beyond training length | Jm/Jmr hold up at several times the training length |
| **Everything else** | — | Same (MLP, LayerNorm, weight tying, training loop, data loading) |

## Models

| Model | Positions | Cross-segment attention | Description |
|-------|-----------|-------------------------|-------------|
| **B** | Continuous RoPE | Standard causal | Baseline. nanoGPT with RoPE instead of learned position embeddings. |
| **Jm** | Reset at each delimiter | Two-score, content addresses | Each segment's keys carry an address computed from the segment's content (mean of rotated embeddings, LayerNorm, linear projection). |
| **Jmr** | Reset at each delimiter | Two-score, random addresses | Same as Jm with an i.i.d. random address per segment. Ablation: isolates the effect of the reset and the routing from the content signal. |

## How two-score attention works

Every query attends causally to every key. The score used for a (query, key) pair depends on whether the two tokens are in the same segment:

```
same segment:    score = q_rot . k_rot        (pure reset-RoPE, position within segment)
other segment:   score = q_rot . k_addr       (k_addr = k_rot rotated by the key's segment address)
```

- **Q never carries an address.** Everything a query computes depends only on tokens at or before it.
- **K carries its own segment's address**, and that copy is only scored by queries in later segments, so the address of a segment is consumed only after the segment is complete.
- Within a segment, attention is purely positional, and the positions are always in the trained range.
- Across segments, attention is by content address, with no dependence on how far apart the segments are.

The two scores are realised as one softmax over a doubled key set `[k_rot ; k_addr]` with a `T x 2T` mask that admits exactly one copy per causal pair: the plain copy when the pair is in the same segment, the addressed copy otherwise. This is algebraically the same as selecting between two score matrices, and FlexAttention compiles the mask into a block-sparse fused kernel. On CPU the model falls back to SDPA with an explicit boolean mask.

### Why not a unit mask

An earlier version of this repo restricted non-final segments to attend only within themselves and let only the final segment attend across segments. That rule needs to know which segment is the final one, and that depends on whether a delimiter appears *later* in the window. Changing a later token to the delimiter moved earlier logits, so the model could tell it was in the final segment and that the next token was therefore not a delimiter. The two-score formulation treats every segment the same way and has no such dependence. `causality_test.py` now perturbs tokens to and from the delimiter specifically to catch this class of problem; the old random-token test could not.

## Causality test

```bash
CUDA_VISIBLE_DEVICES="" python causality_test.py
```

Perturbs one token and checks that no logit before it changes, for five perturbations: a token change inside a completed segment, a token change inside the final segment, a token turned into a delimiter inside the final segment, a token turned into a delimiter inside a completed segment, and a delimiter removed. All three models give an exact 0.0 on CPU.

## Results

Setup: 12 layers, 8 heads, 512 dim (~63M params), trained on WikiText-103 at context 1024, delimiter ` .` (GPT-2 token 764), batch 2 x 8 accumulation, cosine schedule. Two training lengths are reported: 50,000 iterations (main results) and 5,000 iterations (earlier, weaker models).

Two inference-time length corrections are used, one per model family, and reported separately from the raw numbers:

- **B + NTK**: dynamic NTK scaling of the RoPE base beyond the training length, the standard correction for RoPE models (`--ntk`).
- **Jm + offset**: subtract `c * log(n_i / n_ref)` from every cross-segment score, where `n_i` is the number of segments before the query's segment and `n_ref` = 32 is the count at training length (`--cross_shift_c`). Without it, the total attention mass on cross-segment keys grows with their count and crowds out the query's own segment. `c = 1` holds that mass at its training-length level; `c = 1.5` was marginally better. The term is the same for every cross-segment key, so it introduces no distance preference, and it depends only on past tokens, so causality is unchanged. It is zero at and below the training length.

### 50,000 iterations

Three training seeds per model. Validation loss at the training length, mean over seeds: B 3.176, Jm 3.147, Jmr 3.232. All tables below are means ± standard deviation over the three seeds and can be regenerated with `python results/aggregate.py`.

#### Perplexity by context length

Same 60 validation windows for every model and setting (`--seed 0`).

| ctx | B | B + NTK | Jm | Jm + offset (c=1) | Jm + offset (c=1.5) | Jmr | Jmr + offset (c=1.5) |
|-----|---|---|---|---|---|---|---|
| 1024 | 23.00 ± 0.08 | 23.00 ± 0.08 | 22.82 ± 0.03 | 22.83 ± 0.03 | 22.83 ± 0.04 | 24.46 ± 0.26 | 24.46 ± 0.26 |
| 2048 | 30.99 ± 3.54 | 24.82 ± 0.28 | 23.60 ± 0.07 | 23.59 ± 0.06 | 23.67 ± 0.06 | 25.58 ± 0.32 | 25.62 ± 0.34 |
| 4096 | 64.57 ± 37.24 | 27.82 ± 0.39 | 29.88 ± 3.25 | 23.51 ± 0.24 | **23.50 ± 0.09** | 29.39 ± 0.27 | 25.31 ± 0.27 |
| 8192 | 136.75 ± 79.70 | 35.90 ± 2.55 | 64.13 ± 14.74 | 26.46 ± 1.40 | **25.12 ± 0.23** | 61.05 ± 2.38 | 26.92 ± 0.24 |
| 16384 | 281.95 ± 105.40 | 55.97 ± 5.22 | 157.81 ± 33.97 | 35.46 ± 3.49 | **30.67 ± 0.34** | 169.87 ± 16.98 | 32.61 ± 0.30 |

B shows the usual RoPE collapse past its training length, with a large seed-to-seed spread; NTK scaling recovers it partly (2.4x its training-length perplexity at 16384). Raw Jm is ahead of every B variant to 2048, level with B + NTK at 4096, and then degrades through the cross-segment mass effect. With the offset, Jm is at 1.34x its training-length perplexity at 16x with a seed spread of 0.34: 30% better than B + NTK at 8192 and 45% better at 16384. The derived value c = 1 gets most of the way; c = 1.5 is slightly better and tighter across seeds. Jmr is behind Jm at every length including the training length, so the learned addresses are doing real work after longer training.

#### Passkey retrieval

`passkey.py` inserts "The pass key is NNNNN . Remember it ." at a random sentence boundary in WikiText filler and asks for the number at the end. Every model is fine-tuned from its checkpoint on passkey examples of at most 1024 tokens (loss on the answer tokens only), so every length above 1024 is extrapolation. Scores are exact match on all three answer tokens over 100 examples per length; the same examples are used for every model. Offset columns use c = 1, the derived value; c = 1.5 gives the same numbers to within 0.01.

Main run: 5000 fine-tuning steps at learning rate 1e-4, training seed 0.

| ctx | B | B + NTK | Jm | Jm + offset | Jmr | Jmr + offset |
|-----|---|---|---|---|---|---|
| 512 | 0.96 | 0.96 | 0.96 | 0.96 | 0.95 | 0.95 |
| 1024 | 0.97 | 0.97 | 0.96 | 0.96 | 0.95 | 0.95 |
| 2048 | 0.27 | 0.94 | 0.97 | 0.97 | 0.97 | 0.97 |
| 4096 | 0.00 | 0.11 | 0.95 | 0.95 | 0.96 | 0.97 |
| 8192 | 0.00 | 0.00 | 0.93 | 0.93 | 0.91 | 0.92 |
| 16384 | 0.00 | 0.00 | **0.98** | **0.98** | 0.50 | 0.98 |

Robustness run: 1500 fine-tuning steps at learning rate 5e-5, five fine-tuning runs pooled (three fine-tuning seeds on training seed 0, one each on training seeds 1 and 2), mean ± standard deviation.

| ctx | B | B + NTK | Jm | Jm + offset | Jmr | Jmr + offset |
|-----|---|---|---|---|---|---|
| 512 | 0.73 ± 0.04 | 0.73 ± 0.04 | 0.74 ± 0.02 | 0.74 ± 0.02 | 0.71 ± 0.02 | 0.71 ± 0.02 |
| 1024 | 0.75 ± 0.02 | 0.75 ± 0.02 | 0.74 ± 0.02 | 0.74 ± 0.02 | 0.72 ± 0.03 | 0.72 ± 0.03 |
| 2048 | 0.50 ± 0.25 | 0.73 ± 0.03 | 0.75 ± 0.03 | 0.75 ± 0.03 | 0.73 ± 0.03 | 0.73 ± 0.03 |
| 4096 | 0.00 ± 0.01 | 0.40 ± 0.28 | 0.74 ± 0.02 | 0.74 ± 0.01 | 0.71 ± 0.02 | 0.71 ± 0.02 |
| 8192 | 0.00 ± 0.00 | 0.11 ± 0.12 | 0.67 ± 0.05 | **0.71 ± 0.02** | 0.68 ± 0.02 | 0.70 ± 0.02 |
| 16384 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.22 ± 0.15 | **0.79 ± 0.02** | 0.63 ± 0.15 | 0.76 ± 0.02 |

First-token accuracy in the robustness run (whether the model locates the key at all): Jm with the offset is 0.98 to 1.00 at every length; B is 0.13 ± 0.14 at 4096 and 0.00 at 8192 and beyond; B + NTK is 0.75 ± 0.27 at 4096, 0.35 ± 0.32 at 8192, 0.04 at 16384.

Reading the two runs together: in distribution every model copies the key equally well, at 0.96 with the longer fine-tune and 0.74 with the shorter one, so the ceiling is the copy skill and is shared. Beyond the training length, B fails at 2x without scaling and at 4x with it. Jm retrieves at every length out to 16x. With the shorter fine-tune, raw Jm is partly crowded at 16x by the cross-segment mass effect and the offset restores it; with the longer fine-tune the key-sentence match is sharp enough that raw Jm retrieves at 16x on its own (0.98), and the offset changes nothing. Jmr, whose random addresses attenuate the content match, still needs the offset at 16x.

### 5,000 iterations

Validation loss at the training length: B 3.70, Jm 3.71, Jmr 3.77.

#### Perplexity by context length

Same 60 validation windows for every row (`--seed 0`).

| ctx | B | B + NTK | Jm | Jm + offset | Jmr | Jmr + offset |
|-----|------|---------|------|-------------|------|--------------|
| 1024 | 39.40 | 39.40 | 39.67 | 39.67 | 43.37 | 43.37 |
| 2048 | 44.53 | 41.55 | 41.13 | 41.16 | 45.91 | 45.95 |
| 4096 | 49.15 | 41.61 | 42.34 | **39.79** | 45.48 | 44.41 |
| 8192 | 55.21 | 43.93 | 54.44 | **41.00** | 53.10 | 45.37 |
| 16384 | 71.90 | 54.25 | 88.44 | **48.09** | 81.79 | 52.00 |

Offset columns use c = 1.5.

Raw Jm is best out to about 4x and then loses to B, because its cross-segment softmax runs over every earlier segment with no distance term. With the offset it is ahead at every length beyond training: 4% at 4096, 7% at 8192, 11% at 16384 against B with NTK scaling. Multiplying cross-segment scores by a growing factor instead (a sharper softmax) makes things much worse; the problem is total mass, not flatness.

#### Passkey retrieval

`passkey.py` inserts "The pass key is NNNNN . Remember it ." at a random sentence boundary in WikiText filler and asks for the number at the end. Each model is fine-tuned from its checkpoint for 300 steps on examples up to 1024 tokens (loss on the answer tokens only), then scored on 50 identical examples per length. Exact match over all three answer tokens:

| ctx | B | B + NTK | Jm | Jm + offset | Jmr | Jmr + offset |
|-----|------|---------|------|-------------|------|--------------|
| 512 | 0.56 | 0.56 | 0.52 | 0.52 | 0.50 | 0.50 |
| 1024 | 0.54 | 0.54 | 0.56 | 0.56 | 0.56 | 0.56 |
| 2048 | 0.58 | 0.56 | 0.58 | 0.58 | 0.58 | 0.58 |
| 4096 | 0.08 | 0.46 | 0.48 | 0.48 | 0.46 | 0.46 |
| 8192 | 0.00 | 0.34 | 0.50 | 0.54 | 0.52 | 0.52 |
| 16384 | 0.00 | 0.00 | 0.44 | **0.72** | 0.50 | 0.62 |

First-token accuracy (the model finds the key at all): B falls from 0.96 at 1024 to 0.22 at 4096 and 0.04 at 8192; B + NTK holds 0.90 and 0.84 there but drops to 0.38 at 16384; Jm + offset stays at 0.96 at every length including 16384. The ~0.55 exact-match ceiling at short lengths is the copy skill after 300 fine-tuning steps and is shared by all models (50 trials per cell, so about ±0.07). Retrieval by content address does not depend on how far back the key is; retrieval by relative position stops at the training length and is only partly restored by scaling.

### A note on the Jmr ablation

Jmr draws each segment's address angles from a standard normal, which rotates each dimension pair by about one radian. That attenuates the raw query-key match by roughly 0.6 rather than removing it, so Jmr is a noisy-address control, not a no-content control. Jmr tracking Jm means most of the cross-segment matching is carried by the query-key product itself plus the reset positions, with the learned address as a modulation on top.

### Evaluate

```bash
# perplexity sweep; --no_flex uses the SDPA path, which needs far less memory when FlexAttention is uncompiled
python eval_lengths.py --ckpt out-Jm/ckpt.pt --lengths 1024 2048 4096 8192 16384 --no_flex --cross_shift_c 1.5
python eval_lengths.py --ckpt out-B/ckpt.pt  --lengths 1024 2048 4096 8192 16384 --ntk

# passkey: fine-tune from a checkpoint, then score at each length
python passkey.py --ckpt out-Jm/ckpt.pt --out out-Jm-passkey --eval_before --cross_shift_c 0 1.5
python passkey.py --ckpt out-B/ckpt.pt  --out out-B-passkey  --eval_before --ntk
```

## Quick start

### Requirements

- Python 3.10+
- PyTorch 2.6+ (for FlexAttention)
- NumPy, tiktoken, and `datasets` (for data preparation)

### Prepare data

```bash
pip install numpy tiktoken datasets
python data/wikitext103/prepare.py   # downloads WikiText-103-raw, writes data/wikitext103/{train,val}.bin (~230 MB)
```

### Note on GPUs without native bfloat16 (e.g. T4)

`train.py` picks float16 there; PyTorch reports bfloat16 as supported by emulation, which is about 5x slower. If `torch.compile` is unavailable, pass `--use_flex=False` for Jm/Jmr: the SDPA path with a boolean mask gives identical results and is about 4x faster than uncompiled FlexAttention.

### Train

```bash
# Baseline (standard RoPE)
python train.py config/train_wikitext103_small_B.py

# Two-score attention with content addresses
python train.py config/train_wikitext103_small_Jm.py

# Two-score attention with random addresses (ablation)
python train.py config/train_wikitext103_small_Jmr.py
```

## File structure

```
results/          — Raw evaluation logs behind every table in this README (see results/README.md)
data/wikitext103/prepare.py — Dataset preparation
model.py          — GPT model with RoPEAttention (B) and TwoScoreAttention (Jm/Jmr)
train.py          — Training loop (single GPU or DDP)
eval_lengths.py   — Perplexity at multiple context lengths (--no_flex, --cross_shift_c, --ntk)
passkey.py        — Passkey retrieval: fine-tune from a checkpoint, score by context length
causality_test.py — Verify no output depends on later tokens (delimiter-aware)
configurator.py   — CLI config system
config/           — Training configs for WikiText-103
data/             — Dataset directory
```

## Configuration

Key config parameters beyond standard nanoGPT:

- `model_type`: `'B'`, `'Jm'`, or `'Jmr'`
- `delimiter_id`: Token ID for segment boundaries (default: 628 for `\n\n` in GPT-2 BPE; use 764 for ` .` as sentence boundary)
- `cross_shift_c`, `cross_scale_ref`: cross-segment score offset for Jm/Jmr at inference (see Results)
- `rope_ntk`, `rope_ref_len`: dynamic NTK scaling for B at inference
- `use_flex`: FlexAttention (default) or SDPA with an explicit mask for Jm/Jmr on CUDA

## License

Copyright (c) 2026 Mahesh Godavarti. This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See [LICENSE](LICENSE) for details.

## Acknowledgments

Based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (MIT License).
