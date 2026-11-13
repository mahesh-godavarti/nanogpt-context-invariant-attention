# Adapting pretrained RoPE models to content-addressed attention

Mahesh Godavarti, 2026-11-06. Final write-up of the adaptation experiments (2026-10-26 to 11-06). Raw tables and run logs:
`logs_vt/ADAPT_RESULTS.md`, `logs_smollm/`; code in this repository (uncommitted); structured-task design in `STRUCT_DESIGN.md`;
patent notes in `/home/ubuntu/patent/`. Earlier draft kept as `WRITEUP_ADAPT_draft_2026-11-05.md`.

## 1. Questions

1. Can a public pretrained RoPE transformer be given content-addressed attention cheaply, with a small corpus of our choosing
   rather than its pretraining data, so that it works far past its pretrained context without losing anything inside it?
2. Can the same mechanism let a pretrained model read facts supplied as structured units, with the relation and argument roles
   carried in rotation angles rather than words?

## 2. Models, data, evaluation

| | SmolLM-135M | Pythia-160M | Pythia-410M |
|---|---|---|---|
| source | Hugging Face, Apache 2.0 | EleutherAI, Apache 2.0 | EleutherAI, Apache 2.0 |
| architecture | LLaMA: RMSNorm, SwiGLU, GQA 9/3 heads, 30 layers, 576 d | GPT-NeoX: parallel residual, biases, 12 layers, 768 d | GPT-NeoX, 24 layers, 1024 d, 16 heads |
| RoPE | all 32 planes per head | 8 of 32 planes per head (rotary 25%) | 8 of 32 planes per head |
| pretraining | 600B tokens web, context 2048 | 300B tokens Pile, context 2048 | 300B tokens Pile, context 2048 |
| adaptation text | PG-19 books, 8M tokens | FineWeb-Edu (2024 web), 8M tokens | FineWeb-Edu, 8M tokens |
| evaluation text | WikiText-103 validation + test | same | same |

Each model is loaded into this repository's model class by a weight converter (`GPT.from_hf_llama`, `GPT.from_hf_neox`); the
converted models reproduce the Hugging Face losses to 1e-3 nats or better. FineWeb-Edu is used for the Pythias because their
pretraining set, the Pile, already contains PG-19. Evaluation: perplexity on 50 random WikiText windows at 512 to 16384 tokens
(8x the pretrained context), and passkey retrieval (a 5-digit key in a sentence at a random depth in WikiText filler, exact-match
accuracy over 50 trials) after a 300-step passkey fine-tune. Baselines: the untouched model, plain and with NTK frequency scaling
(the standard inference-time fix), and a control fine-tuned the same 500 steps on the same adaptation text with the attention
unchanged. One T4 GPU; adaptation runs take 35 minutes to 8 hours.

## 3. Methods

**Base mechanism (from the arXiv paper).** The context is split into segments at sentence ends. Each segment gets a content
address, a vector of rotation angles computed by a small learned map from its pooled token embeddings. Keys read by queries in
later segments are rotated by their segment's address; one softmax runs over a doubled key set with a mask that admits exactly
one copy per query-key pair.

**Anneal (every position converted).** Query/key angle = pos_in_segment*f + (1-alpha)*segment_start*f; cross-segment keys get
alpha times the content address. alpha = 0 is the pretrained model exactly, alpha = 1 the content-addressed model. alpha rises
0.1 -> 1 over the run; the address map gets full-strength gradient (straight-through) and starts at zero. Variants: all bands or
only the slowest bands.

**Window (only far positions converted).** Keys closer than W tokens are scored exactly as the pretrained model scores them:
continuous RoPE, original weights. Keys W or more tokens back are scored by in-segment position plus the segment's content
address; one softmax over both. Far scores carry a learned per-head bias (initial -4). Trained with W = 1024 on 2048-token
sequences so that far keys exist in training; evaluated with W = 2048, where the model is the pretrained model inside its whole
context. No anneal. The passkey fine-tune uses sequences up to 4096 with W = 2048 (activation checkpointing). Two variants:
(i) all weights trained, with the pretrained weights at 2e-5 and the new parameters at 1e-3 (the two-rate recipe);
(ii) pretrained weights frozen, the far path given its own query/key projections (copies of the pretrained ones at init), and only
those, the address map and the far biases trained: 13.4M parameters on SmolLM, 14.5M on Pythia-160M, 50.9M on Pythia-410M.
The new-parameter rate is 1e-3 on SmolLM and 1e-4 on the Pythias; it is the only setting that differs between models.

**Structured units.** A fact (A, relation, B) is a two-token segment [A, B]. The far angle of a unit token is a learned slot angle
plus a learned relation angle; the unit's address is pooled from its token embeddings after that rotation (rotate-then-pool), so
the address encodes which name is the head and which relation it states. Units are read only through the far path, at any
distance; a unit's tokens attend only inside their own unit; text goes through the pretrained near path unchanged. Trained with
the base frozen on contexts of 16 units plus a statement "The <relation> of A is", loss on the answer name; 8 relations, 226-238
single-token names; three of the 16 units share the target's head name so the relation must be read. Tests: new triples, 46
never-seen names, and 64-unit contexts (trained on 16). Metric: hit@1 over 500 contexts. Baseline: the untouched model with the
same facts written as sentences in its context.

## 4. Length extension: perplexity on WikiText-103

Window models are means over three seeds (seed ranges are within 0.2 ppl unless noted); other columns one seed.

**SmolLM-135M**

| length | untouched | untouched + NTK | control (books) + NTK | anneal, 500 steps | window, all weights | window, frozen |
|---|---|---|---|---|---|---|
| 2048 | 17.74 | 17.74 | 20.13 | 27.49 | 19.77 | 17.74 |
| 4096 | 116.54 | 22.86 | 25.94 | 30.84 | 20.77 | 18.46 |
| 8192 | 474.02 | 42.07 | 47.44 | 30.88 | 19.29 | 17.02 |
| 16384 | 1969.89 | 98.37 | 111.00 | 35.49 | 19.36 | 16.64 |

**Pythia-160M**

| length | untouched | untouched + NTK | control (web) + NTK | window, all weights | window, frozen |
|---|---|---|---|---|---|
| 2048 | 27.63 | 27.63 | 29.52 | 29.7-30.6 | 27.63 |
| 4096 | 105.65 | 25.77 | 27.01 | 26.6-27.4 | 25.9-26.1 |
| 8192 | 764.89 | 33.46 | 33.26 | 26.4-27.3 | 26.4-26.5 |
| 16384 | 1585.71 | 73.75 | 69.43 | 25.5-26.4 | 26.3 |

**Pythia-410M**

| length | untouched | untouched + NTK | window, all weights | window, frozen |
|---|---|---|---|---|
| 2048 | 16.32 | 16.32 | 18.1-18.4 | 16.32 |
| 4096 | 98.37 | 15.09 | 16.4-16.6 | 15.5 |
| 8192 | 1365.91 | 18.11 | 16.3-16.6 | 15.9 |
| 16384 | 7263.28 | 45.47 | 16.0-16.2 | 16.3-16.6 |

## 5. Length extension: passkey retrieval (exact match, after the passkey fine-tune)

Window models: range over three seeds.

| model | context | untouched + NTK | window, all weights | window, frozen |
|---|---|---|---|---|
| SmolLM-135M | 4096 | .02 | .94-1.00 | .96-1.00 |
| | 8192 | .00 | .94-1.00 | 1.00 |
| | 16384 | .00 | .48-.98 | .94-.96 |
| Pythia-160M | 4096 | .96 | .86-.92 | .98-1.00 |
| | 8192 | .68 | .78-.94 | .92-.98 |
| | 16384 | .00 | .84-.86 | .96 |
| Pythia-410M | 4096 | .74 | 1.00 | 1.00 |
| | 8192 | .26 | .92-1.00 | 1.00 |
| | 16384 | .02 | .86-1.00 | .94-1.00 |

In every window run the key sentence is found (first digit right) in 92-100% of trials at every length; what varies is copying
the remaining digits. The all-weights SmolLM range at 16384 (.48-.98) and the all-weights 410M seed-0 value (.86) predate the
lowest-loss restoration in the passkey fine-tune (section 9); later seeds with it are at .96-1.00.

## 6. Structured facts: hit@1 on the answer name (three seeds per model)

| model | facts as sentences, 16 | units, 16 | sentences, 16, new names | units, 16, new names | sentences, 64 | units, 64 |
|---|---|---|---|---|---|---|
| SmolLM-135M | .21 | .990-.994 | .29 | .942-.952 | .08 | .870-.898 |
| Pythia-160M | .13 | .966-.984 | .22 | .700-.748 | .04 | .886-.928 |
| Pythia-410M | .45 | 1.000 | .47 | .976-.980 | .16 | .986-1.000 |

Bare name pairs in context score 0 on every model. The address design was decisive: with the slot and relation rotation applied
after pooling, the address was symmetric in the two names and relation-blind, and selection among many units failed (SmolLM
.68 / .55 / .15 at 2000 steps); rotate-then-pool fixed it. The 410M reached .99 / .95 / .92 within 2000 of its 10000 steps.

## 6b. Repeated-document test: ordinary prose at a distance, no fine-tuning

A 300-token WikiText passage, then a gap of unrelated filler, then the same passage again; the second copy is scored as ordinary
next-token prediction (perplexity and token accuracy), 12 paired trials per gap. Inside the window (gap 1024) every model copies
the passage: token accuracy .95-.98. Beyond it:

| model | setting | 4096 | 8192 | 14000 |
|---|---|---|---|---|
| SmolLM-135M | untouched / +NTK | .07 / .18 | .06 / .08 | .06 / .08 |
| | window, all weights / frozen | .55 / .53 | .55 / .54 | .52 / .50 |
| Pythia-160M | untouched / +NTK | .03 / .45 | .04 / .34 | .03 / .17 |
| | window, all weights / frozen | .46 / .40 | .44 / .40 | .45 / .40 |
| Pythia-410M | untouched / +NTK | .02 / .60 | .00 / .21 | .00 / .12 |
| | window, all weights / frozen | .39 / .71 | .39 / .63 | .41 / .62 |

(token accuracy on the second copy; perplexities in `logs_vt/ADAPT_RESULTS.md`; the unseen-passage level is 20-45 ppl and
2-5% accuracy). The window models find and use the first copy of real prose at any distance, with no task fine-tune: accuracy
is flat from 4K to 14K where NTK decays to the unseen level. The copy is partial, 40-71% of tokens against 95-98% inside the
window, so the far path retrieves approximately where the near path retrieves exactly. The frozen 410M is the best at .62-.71.

## 7. What is established

1. **Keeping the pretrained model inside its context and adding content-addressed attention beyond it works.** With all weights
   trained for 500 steps on 8M tokens of unrelated text, all three models hold perplexity flat or falling to 8x their pretrained
   context and retrieve a passkey there in 84-100% of trials, where RoPE collapses and NTK gives 0-2%. The only cost inside the
   context is that of the fine-tuning text, 1-2 perplexity, which a no-address control pays equally.
2. **With the pretrained weights frozen the result is cleaner.** Far-path query/key projections, the address map and the far
   biases are the only new parameters (9-13% of the model). Inside its context the model is the untouched model to the last
   digit; beyond it, perplexity falls below the model's own 2048 value out to 16384 on all three models; retrieval at 8x is
   94-100%. The one per-model setting is the new-parameter learning rate (1e-3 on SmolLM, 1e-4 on the Pythias; at 1e-3 the
   Pythia far path does not train, at 3e-3 it diverges).
3. **Structured facts.** A frozen model reads facts supplied as two-name units through the far path, with no relation words, at
   97-100% among 16 units and 87-100% among 64 (trained on 16), against 13-45% and 4-16% reading the same facts written as
   sentences in its context. The unit address must be built after the slot and relation rotation.
4. **Real prose at a distance.** Without any task fine-tune, the window models predict a repeated 300-token passage placed 4K
   to 14K tokens back at 40-71% token accuracy, flat with distance, where the untouched model is at 2-7% and NTK decays from
   18-60% at 4K to 8-17% at 14K; inside the window all models copy at 95-98%.
5. **Scope.** Three models (135M, 160M, 405M), two labs, two architectures (full and quarter rotary), two adaptation corpora,
   three seeds for every recipe on every model, one T4 GPU, 500 adaptation steps for length and 10,000 short steps for facts.

## 8. What was tried and set aside

The anneal gives the length behaviour but costs 7 ppl at the pretrained length; 10x longer annealing makes it worse (33.0 vs
27.5 at 2048) and converting only the slowest quarter of the bands removes the cost but loses the length behaviour (652 at
16384). A single learning rate for the all-weights window recipe on Pythia costs either the base (1e-4: 7 ppl) or the far path
(2e-5: 16K passkey .56); the two-rate recipe keeps both. Mixed context sizes (16-64 units) during structured training slowed
learning and did not help 64-unit retrieval; the address fix did. The inference-time cross-segment offset (c = 1.5), which
helps the anneal model, hurts window models (their far bias already does that job): all window results use c = 0.

## 9. Pitfalls met (for whoever reruns this)

Half precision: the window path materialises attention scores; Pythia's overflow fp16, and `.float()` inside autocast is
silently undone, so scores are computed with autocast disabled. The adapt loader must copy every architecture field from the
checkpoint (the trainer asserts that every pretrained tensor loads). Passkey fine-tunes must contain far keys (a window-2048
fine-tune on 2048-token sequences never exercises the far path); on the 405M the all-weights passkey fine-tune needs lr 1e-5
(5e-5 diverges); the fine-tune can spike on its last steps, so the script now restores the lowest smoothed-loss parameters
before evaluating. The 16K evaluation of the 405M must run in its own process. Bound every rejection-sampling loop in data
generation (an unbounded one hung for five hours). A suspected reproducibility problem on 11-03 was a misreading of filtered
logs; the saved checkpoint reproduces the logged numbers.

## 10. Open items

A 1B-class model (the frozen recipe makes it feasible on this GPU). The gap between approximate far copying (40-71% of tokens)
and exact near copying (95-98%): the hypothesis that the far path simply lacked exposure to long-range repeats was tested on
2026-11-10 by adapting the frozen SmolLM on 8192-token sequences with the 2048 window, and rejected (far-copy accuracy .48-.50
against .50-.54; perplexity slightly better at 16384, 15.7 against 16.6; passkey slightly worse). The ceiling is in the mechanism,
the token-by-token read-off through far keys, which does not reproduce the pretrained induction circuit the near path uses.
Lengths beyond 16384. The structured task with all weights trained, for comparison. Publishing weights and code after the
provisional is filed.
