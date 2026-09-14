# Result logs

Raw output of the evaluation runs behind the tables in the top-level README.
Models: 12 layers, 8 heads, 512 dim, trained at context 1024 on WikiText-103
(checkpoints not included; ~760 MB each).

50,000-iteration models (three training seeds: `out-50k-X`, `out-50k-X-s1`, `out-50k-X-s2`). `aggregate.py` rebuilds the README tables from these files.

| File | What it is |
|---|---|
| `ppl_50k.log`, `ppl_50k_c1.log` | Training seed 0: seeded 60-batch length sweeps, 1024..16384, B plain and NTK; Jm and Jmr with c = 0, 1.5 (first file) and c = 1.0 (second) |
| `ppl_50k_s1.log`, `ppl_50k_s2.log` | Same sweeps for training seeds 1 and 2 (c = 0, 1.0, 1.5) |
| `passkey_50k_ceiling.log` | Passkey, 5000 fine-tuning steps at lr 1e-4, training seed 0, 100 trials, 512..16384. Main passkey table |
| `passkey_50k_long.log` | Passkey, 1500 fine-tuning steps at lr 5e-5, training seed 0, three fine-tuning seeds x 100 trials |
| `passkey_50k_long_s1.log`, `passkey_50k_long_s2.log` | Same for training seeds 1 and 2, one fine-tuning seed each |
| `passkey_50k.log` | Earliest passkey run, 300 fine-tuning steps, 1 seed x 50 trials |
| `ppl_50k_R_s0.log`, `ppl_50k_R_s1.log`, `ppl_50k_R_s2.log` | Model R: seeded 60-batch length sweeps, 1024..16384, one per training seed |
| `passkey_50k_long_R.log` | Model R: passkey, 1500 fine-tuning steps, five runs (three fine-tuning seeds on training seed 0, one each on seeds 1 and 2) |
| `passkey_50k_ceiling_R.log` | Model R: passkey, 5000 fine-tuning steps at lr 1e-4, training seed 0 |
| `ppl_50k_R_offset_s0.log`, `ppl_50k_R_offset_s1.log`, `ppl_50k_R_offset_s2.log` | Model R with the cross-segment offset (c = 1.0, 1.5), same sweeps |
| `passkey_50k_long_R_offset.log`, `passkey_50k_ceiling_R_offset.log` | The fine-tuned R checkpoints above re-scored with the offset (c = 0 and 1.0; the c = 0 rows reproduce the plain runs) |
| `passkey_50k_budget2500.log` | R and Jm, passkey, 2500 fine-tuning steps at lr 1e-4, training seeds 0-2, c = 0 and 1.0 |
| `passkey_50k_budget5000_s12.log` | R and Jm, passkey, 5000 fine-tuning steps at lr 1e-4, training seeds 1-2 (seed 0 is the ceiling run), c = 0 and 1.0 |

5,000-iteration models:

| File | What it is |
|---|---|
| `eval_5k_B.log`, `eval_5k_Jm.log`, `eval_5k_Jmr.log` | First length sweep, 200 unseeded batches, FlexAttention path (Jm/Jmr fail at 8192 by memory on the uncompiled path) |
| `eval_5k_Jm_8192.log`, `eval_5k_Jmr_8192.log`, `eval_5k_Jm_mid.log` | 8192 and 6144 via the SDPA path; the 1024 and 4096 rows confirm the two paths agree |
| `eval_5k_*_16384.log` | 16384 via the SDPA path |
| `lambda_sweep_Jm.log` | Cross-segment score *multiplier* sweep (makes things worse; see README) |
| `shift_sweep_Jm.log`, `shift_sweep_Jm_2.log` | Cross-segment score *offset* sweep, c in {0, 0.5, 1, 1.5, 2}, seeded windows; source of the Jm columns in the perplexity table |
| `ntk_sweep_B.log` | B plain and with NTK scaling, same seeded windows; source of the B columns |
| `final_extra.log` | Jmr seeded sweep with and without the offset; passkey at 16384 for all models |
| `passkey_5k.log` | Passkey: accuracy before fine-tuning, 300 fine-tuning steps, accuracy after, for B, Jm, Jmr |
| `passkey_5k_B_ntk.log` | Fine-tuned B scored with and without NTK scaling |

Seeded rows (`--seed 0`, 60 batches) are directly comparable across files; the
first sweep used 200 unseeded batches, so its numbers differ from the seeded
ones by sampling noise of a few percent.
