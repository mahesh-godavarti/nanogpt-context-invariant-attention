# nanogpt-context-invariant-attention

A fork of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy that replaces learned position embeddings with **segment-reset RoPE plus content-addressed cross-segment attention**, implemented with PyTorch's [FlexAttention](https://pytorch.org/blog/flexattention/) API.

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

## Extrapolation results

Results for the two-score models are being regenerated with this version of the code (12 layers, 8 heads, 512 dim, ~63M params, 5K iterations on WikiText-103, delimiter `. `, training context 1024). This section will be updated when the runs finish.

## Quick start

### Requirements

- Python 3.10+
- PyTorch 2.6+ (for FlexAttention)
- NumPy

### Prepare data

```bash
python data/wikitext103/prepare.py
```

### Train

```bash
# Baseline (standard RoPE)
python train.py config/train_wikitext103_small_B.py

# Two-score attention with content addresses
python train.py config/train_wikitext103_small_Jm.py

# Two-score attention with random addresses (ablation)
python train.py config/train_wikitext103_small_Jmr.py
```

### Evaluate extrapolation

```bash
python eval_lengths.py --ckpt out-wikitext103-small-Jm/ckpt.pt --lengths 256 512 1024 2048 4096
```

## File structure

```
model.py          — GPT model with RoPEAttention (B) and TwoScoreAttention (Jm/Jmr)
train.py          — Training loop (single GPU or DDP)
eval_lengths.py   — Evaluate perplexity at multiple context lengths
causality_test.py — Verify no output depends on later tokens (delimiter-aware)
configurator.py   — CLI config system
config/           — Training configs for WikiText-103
data/             — Dataset directory
```

## Configuration

Key config parameters beyond standard nanoGPT:

- `model_type`: `'B'`, `'Jm'`, or `'Jmr'`
- `delimiter_id`: Token ID for segment boundaries (default: 628 for `\n\n` in GPT-2 BPE; use 764 for `. ` as sentence boundary)

## License

Copyright (c) 2026 Mahesh Godavarti. This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See [LICENSE](LICENSE) for details.

## Acknowledgments

Based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (MIT License).
