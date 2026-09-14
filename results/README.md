# Result logs

Raw output of the evaluation runs behind the tables in the top-level README.
Models: 12 layers, 8 heads, 512 dim, trained at context 1024 on WikiText-103
(checkpoints not included; ~760 MB each).

50,000-iteration models:

| File | What it is |
|---|---|
| `ppl_50k.log` | Seeded 60-batch length sweeps, 1024..16384: B plain and NTK; Jm and Jmr with c = 0 and c = 1.5 |
| `passkey_50k.log` | Passkey before/after 300 fine-tuning steps, 512..16384: B plain and NTK; Jm and Jmr with c = 0 and c = 1.5 |

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
