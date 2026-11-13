# Structured-data adaptation of a pretrained model: design (2026-10-31)

## Goal
Show that a pretrained language model (SmolLM-135M, frozen) can be taught, with the far path of the window model, to read facts
supplied as structured units -- names with the relation and roles carried in the rotation angles, no relation words -- and answer
text questions about them. Facts at test time are new and appear only as units, never as sentences.

## Data (synthetic knowledge graph)
- Entities: 226 first names that are single tokens in the SmolLM tokenizer (" Adam", " Brian", ...).
- Relations (8): father, mother, employer, city, spouse, teacher, friend, doctor. Each has a statement template, e.g.
  father: "The father of {A} is {B}." ; employer: "{A} works at the company of {B}." (all answers are names, so exact match is one token).
- A fact is (A, r, B). A context holds K = 16 facts (random A, r, B; distinct A-r pairs), then one statement about one of them with
  the answer token as the target. Distractors: other facts share A or B with the target fact, so the relation must be read.
- Splits by triple: training facts are drawn from all (A, r, B) with A in a 180-name pool; test facts use (i) new triples over the
  same names and (ii) 46 names never seen in training (stricter). Metric: exact match of the answer token (hit@1), 500 contexts each.

## How a fact unit enters the model
Each fact is a segment of two tokens [A, B] with per-token metadata: slot (0 = head, 1 = tail), relation id, offset in slot (0).
Angle of a fact token on the far path = offset*f + slot_angle[slot] + rel_angle[relation]   (learned tables, one angle per plane,
initialised to zero). Content address = the unit's token embeddings pooled, through the address map, on the key side for cross-unit
pairs, as for sentences. Fact units are read ONLY through the far path, at any distance (a per-segment type flag forces the far
score); the pretrained near path never sees a fact token. A fact token as a query attends only inside its own unit (far path,
same-unit mask), so units are encoded independently of each other and of the text. Text tokens use the window rule as before
(near path = pretrained attention; here the text is short, so text-text pairs are all near).

## What is trained
Pretrained weights frozen. Trained: far q/k projections (copies at init), address map, far biases, slot angles (2 x 288),
relation angles (8 x 288). ~13.4M parameters. Loss on the answer token only. ~2000 steps, batch 32, contexts of ~45 tokens.

## Arms
A. Untouched SmolLM, facts as sentences in context ("The father of Adam is Brian. ...") then the question statement -- the
   ordinary in-context reading baseline (upper reference).
B. Adapted model, facts as units (this method). Facts never appear as sentences at test.
C. Untouched SmolLM, facts as bare name pairs in context ("Adam Brian. ...") -- what the pretrained model makes of the unit
   content without the structure code (lower reference).
Report hit@1 on new-triple and new-name test sets for A, B, C; also B's dependence on K (16, 64 facts) at test.

## Implementation
model.py: GPT.forward(idx, targets, meta=None); meta = {seg (B,T) segment ids, off (B,T) in-segment offset, slot (B,T), rel (B,T),
is_fact (B,T)}. When meta is given it replaces the delimiter-based segmentation; the far angle adds slot/rel angles; the window
attention takes a per-key "force far" flag and a per-query "same-segment only" flag. struct_adapt.py: data generator, training
loop (frozen base, new-param optimizer), evaluation of arms A/B/C. CPU checks: with no fact tokens the path equals the window model;
fact tokens never enter near scores; causality.

## Results (2026-10-31, SmolLM-135M frozen, one seed)
| facts given as | 16 facts, new triples | 16 facts, new names | 64 facts, new triples |
|---|---|---|---|
| sentences in context, untouched model (A) | .21 | .29 | .08 |
| bare name pairs, untouched model (C) | .00 | .00 | - |
| units via the far path, adapted (B), address pooled before structure, 2000 steps | .68 | .55 | .15 |
| units via the far path, adapted (B), rotate-then-pool address, 10000 steps, 3 seeds | .990-.994 | .942-.952 | .870-.898 |
The address must encode role and relation (rotate by slot + relation angle, then pool); with that, the frozen model reads a fact
out of 64 units with no relation words at 90%, where it reads the same facts written as sentences at 8%.
