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

Validation loss at the training length: B 3.17, Jm 3.17, Jmr 3.23.

#### Perplexity by context length

Same 60 validation windows for every row (`--seed 0`). Offset columns use c = 1.5.

| ctx | B | B + NTK | Jm | Jm + offset | Jmr | Jmr + offset |
|-----|------|---------|------|-------------|------|--------------|
| 1024 | 23.07 | 23.07 | 22.80 | 22.80 | 24.75 | 24.75 |
| 2048 | 35.08 | 25.00 | 23.53 | 23.62 | 25.94 | 26.00 |
| 4096 | 107.57 | 28.24 | 26.65 | **23.41** | 29.44 | 25.61 |
| 8192 | 228.51 | 38.85 | 48.53 | **24.87** | 58.39 | 27.18 |
| 16384 | 399.06 | 61.85 | 119.95 | **30.28** | 150.28 | 32.93 |

With more training, B shows the usual RoPE collapse past its training length (17x its training-length perplexity at 16384), and NTK scaling recovers it only partly (2.7x). Raw Jm is ahead of every B variant to 4096 and then degrades through the cross-segment mass effect (5.3x at 16384). With the offset, Jm is at 1.33x its training-length perplexity at 16x, 36% better than B + NTK at 8192 and 51% better at 16384. Jmr is behind Jm at every length now, including the training length, so the learned addresses are doing real work after longer training.

#### Passkey retrieval

Same protocol as below (300 fine-tuning steps at lengths up to 1024, 50 trials per length, exact match on all answer tokens).

| ctx | B | B + NTK | Jm | Jm + offset | Jmr | Jmr + offset |
|-----|------|---------|------|-------------|------|--------------|
| 512 | 0.58 | 0.58 | 0.62 | 0.62 | 0.54 | 0.54 |
| 1024 | 0.58 | 0.58 | 0.60 | 0.60 | 0.60 | 0.60 |
| 2048 | 0.30 | 0.58 | 0.60 | 0.60 | 0.60 | 0.60 |
| 4096 | 0.00 | 0.26 | 0.52 | 0.52 | 0.54 | 0.54 |
| 8192 | 0.00 | 0.06 | 0.34 | **0.56** | 0.52 | 0.56 |
| 16384 | 0.00 | 0.00 | 0.02 | **0.68** | 0.60 | 0.64 |

Raw B now fails at 2x the training length; NTK scaling holds it to 2x and gives partial recovery at 4x. Raw Jm holds to 4x and fails at 16x (first-token accuracy 0.02), because its sharper cross-segment attention makes the mass crowding worse; raw Jmr, whose noisier addresses give weaker cross-segment scores, is crowded less and still retrieves at 16x. With the offset, both retrieve at 16x with first-token accuracy above 0.95. After longer training the offset is necessary for Jm's long-range retrieval, and it is sufficient.

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
