# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.
"""Structured-data adaptation of a pretrained model (design: STRUCT_DESIGN.md).

Facts (A, rel, B) over single-token first names. Arm B: facts enter the frozen SmolLM as two-token units read only through the far
path (slot/relation angles + content address); the model completes "The <rel> of A is" with B. Arms A/C: the untouched model with the
facts written as sentences (A) or bare name pairs (C) in context.

    python struct_adapt.py --steps 2000 --out out-struct-smollm
    python struct_adapt.py --eval_only --out out-struct-smollm
"""
import argparse, hashlib, json, math, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig

NAMES = """James John Robert Michael William David Richard Joseph Thomas Charles Christopher Daniel Matthew Anthony Mark Donald Steven Paul Andrew Joshua Kenneth Kevin Brian George Timothy Ronald Edward Jason Jeffrey Ryan Jacob Gary Nicholas Eric Jonathan Stephen Larry Justin Scott Brandon Benjamin Samuel Gregory Alexander Frank Patrick Raymond Jack Dennis Jerry Tyler Aaron Jose Adam Nathan Henry Douglas Zachary Peter Kyle Noah Ethan Jeremy Walter Christian Keith Roger Terry Austin Sean Gerald Carl Harold Dylan Arthur Lawrence Jordan Jesse Bryan Billy Bruce Gabriel Joe Logan Alan Juan Albert Willie Elijah Wayne Randy Vincent Mason Roy Ralph Bobby Russell Bradley Philip Eugene Mary Patricia Jennifer Linda Elizabeth Barbara Susan Jessica Sarah Karen Lisa Nancy Betty Sandra Margaret Ashley Kimberly Emily Donna Michelle Carol Amanda Melissa Deborah Stephanie Dorothy Rebecca Sharon Laura Cynthia Amy Kathleen Angela Shirley Brenda Emma Anna Pamela Nicole Samantha Katherine Christine Helen Debra Rachel Carolyn Janet Maria Catherine Heather Diane Olivia Julie Joyce Victoria Ruth Virginia Lauren Kelly Christina Joan Evelyn Judith Andrea Hannah Megan Cheryl Jacqueline Martha Madison Teresa Gloria Sara Janice Ann Kathryn Abigail Sophia Frances Jean Alice Judy Isabella Julia Grace Amber Denise Danielle Marilyn Beverly Charlotte Natalie Theresa Diana Brittany Doris Kayla Alexis Lori Marie Oscar Felix Hugo Leo Max Victor Simon Louis Oliver Harry Alfie Archie Freddie Theo Harrison Louie Toby Charlie Sebastian Isaac Finley Jude Rory Reuben Ellis Ezra Caleb Harvey Luca Blake Tommy Jenson Hunter Elliot Albie Elliott Roman Carter Riley Kai Dexter Frankie Ollie Grayson Jasper Rupert Otto Ivan Boris Igor Nikolai Dmitri Sergei Yuri Pavel Anton Viktor Marco Matteo Giovanni Paolo Pietro Enzo Dante Carlos Diego Pablo Mateo Alejandro Javier Sergio Rafael Pedro Luis Miguel Fernando Jorge Ricardo Andres Hiroshi Kenji Takeshi Yuki Akira Satoshi Wei Li Chen Ming Hao Jun Raj Arjun Vikram Ravi Amit Sanjay Anil Omar Ali Hassan Ahmed Youssef Karim Tariq Nadia Layla Fatima Zara Priya Anjali Meera Lakshmi Mei Ling Xiu Yuna Sakura Hana Aiko Elena Sofia Irina Olga Natasha Tatiana Svetlana Anya Katya Francesca Chiara Giulia Lucia Alessia Valentina Isabel Carmen Rosa Marta Ines Camila Valeria Ana Beatriz Clara Ingrid Astrid Freya Sigrid Greta Helga Liesel Anneliese Margot Colette Amelie Celine Juliette Manon Louise Chloe""".split()
RELS = ["father", "mother", "spouse", "teacher", "doctor", "boss", "neighbor", "lawyer"]


class KG:
    def __init__(self, tok, n_heldout_names=46, seed=0):
        self.tok = tok
        seen, names = set(), []
        for n in NAMES:
            if n not in seen and len(tok.encode(" " + n)) == 1:
                seen.add(n); names.append(n)
        rng = random.Random(seed); rng.shuffle(names)
        self.names = names; self.held = names[:n_heldout_names]; self.train_names = names[n_heldout_names:]
        self.name_id = {n: tok.encode(" " + n)[0] for n in names}
        self.rel_words = {r: tok.encode(" " + r) for r in RELS}
        assert all(len(v) == 1 for v in self.rel_words.values()), self.rel_words
        self.the, self.of, self.is_, self.dot = tok.encode(" The")[0], tok.encode(" of")[0], tok.encode(" is")[0], tok.encode(".")[0]

    @staticmethod
    def is_test_triple(a, r, b):
        return int(hashlib.md5(f"{a}|{r}|{b}".encode()).hexdigest(), 16) % 10 == 0   # 10% of triples are never trained on

    def sample_context(self, rng, K, split):
        """K facts with distractors sharing the target's head (2, other relations) and tail (1); target fact anywhere.
        Every inner search is bounded; if a distractor cannot be found the whole context is resampled."""
        pool = self.held if split == "new_names" else self.train_names
        def ok(a, r, b):
            t = self.is_test_triple(a, r, b)
            return (t if split != "train" else not t) and a != b
        def pick(cond, tries=400):
            for _ in range(tries):
                cand = rng.choice(pool), rng.randrange(len(RELS)), rng.choice(pool)
                if cond(*cand): return cand
            return None
        while True:
            t = pick(ok)
            if t is None: continue
            a, r, b = t; facts = [t]; used = {(a, r)}; failed = False
            others = [rr for rr in range(len(RELS)) if rr != r]; rng.shuffle(others)
            for rr in others[:2]:
                d = pick(lambda aa, r2, bb: ok(a, rr, bb) and (a, rr) not in used)
                if d is None: failed = True; break
                facts.append((a, rr, d[2])); used.add((a, rr))
            if failed: continue
            d = pick(lambda aa, rr, bb: ok(aa, rr, b) and (aa, rr) not in used)
            if d is None: continue
            facts.append((d[0], d[1], b)); used.add((d[0], d[1]))
            while len(facts) < K:
                d = pick(lambda aa, rr, bb: ok(aa, rr, bb) and (aa, rr) not in used)
                if d is None: failed = True; break
                facts.append(d); used.add((d[0], d[1]))
            if failed: continue
            rng.shuffle(facts)
            return facts, (a, r, b)

    def encode_units(self, facts, target):
        """Arm B: two-token units + statement prefix. Returns ids and meta arrays (lists)."""
        a, r, b = target
        ids, seg, off, slot, rel, fact = [], [], [], [], [], []
        for k, (fa, fr, fb) in enumerate(facts):
            ids += [self.name_id[fa], self.name_id[fb]]; seg += [k, k]; off += [0, 0]; slot += [0, 1]; rel += [fr, fr]; fact += [1, 1]
        stmt = [self.the, self.rel_words[RELS[r]][0], self.of, self.name_id[a], self.is_]
        K = len(facts)
        ids += stmt; seg += [K] * len(stmt); off += list(range(len(stmt))); slot += [0] * len(stmt); rel += [0] * len(stmt); fact += [0] * len(stmt)
        return ids, dict(seg=seg, off=off, slot=slot, rel=rel, fact=fact), self.name_id[b]

    def encode_text(self, facts, target, bare=False):
        """Arms A (sentences) / C (bare pairs): plain token sequence ending with the statement prefix."""
        a, r, b = target
        ids = []
        for fa, fr, fb in facts:
            if bare: ids += [self.name_id[fa], self.name_id[fb], self.dot]
            else: ids += [self.the, self.rel_words[RELS[fr]][0], self.of, self.name_id[fa], self.is_, self.name_id[fb], self.dot]
        ids += [self.the, self.rel_words[RELS[r]][0], self.of, self.name_id[a], self.is_]
        return ids, self.name_id[b]


def batch_units(kg, rng, n, K, split, device):
    seqs = [kg.encode_units(*kg.sample_context(rng, K, split)) for _ in range(n)]
    ids = torch.tensor([s[0] for s in seqs], device=device)
    meta = {k: torch.tensor([s[1][k] for s in seqs], device=device) for k in seqs[0][1]}
    y = torch.tensor([s[2] for s in seqs], device=device)
    return ids, meta, y


def batch_text(kg, rng, n, K, split, device, bare):
    seqs = [kg.encode_text(*kg.sample_context(rng, K, split), bare=bare) for _ in range(n)]
    return torch.tensor([s[0] for s in seqs], device=device), torch.tensor([s[1] for s in seqs], device=device)


@torch.no_grad()
def hit1_units(model, kg, K, split, n, device, seed, bs=64):
    model.eval(); rng = random.Random(seed); hits = 0
    for s in range(0, n, bs):
        ids, meta, y = batch_units(kg, rng, min(bs, n - s), K, split, device)
        with torch.autocast(device, dtype=torch.float16, enabled=(device == 'cuda')):
            logits, _ = model(ids, meta=meta)
        hits += int((logits[:, -1].argmax(-1) == y).sum())
    return hits / n


@torch.no_grad()
def hit1_text(model, kg, K, split, n, device, seed, bare, bs=32):
    model.eval(); rng = random.Random(seed); hits = 0
    for s in range(0, n, bs):
        ids, y = batch_text(kg, rng, min(bs, n - s), K, split, device, bare)
        with torch.autocast(device, dtype=torch.float16, enabled=(device == 'cuda')):
            logits, _ = model(ids)
        hits += int((logits[:, -1].argmax(-1) == y).sum())
    return hits / n


def build(ckpt_path, device, struct):
    c = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ma = dict(c['model_args']); ma.update(use_flex=False, block_size=4096)
    if struct:
        ma.update(model_type='Jm', window=2048, far_bias_init=-4.0, far_proj=True, n_slots=2, n_rels=len(RELS))
    m = GPT(GPTConfig(**ma))
    missing, unexpected = m.load_state_dict(c['model'], strict=False)
    assert not unexpected, unexpected
    if struct:
        for blk in m.transformer.h: blk.attn.init_far_proj()
        torch.nn.init.zeros_(m.seg_proj.weight)
        for n, p in m.named_parameters():
            p.requires_grad_(n.startswith('seg_') or n.endswith('far_bias') or '_far_proj' in n or n.endswith('_angles'))
    return m.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='out-smollm-B/ckpt.pt'); ap.add_argument('--hf_name', default='HuggingFaceTB/SmolLM-135M')
    ap.add_argument('--out', default='out-struct-smollm'); ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--K', type=int, default=16); ap.add_argument('--K_max', type=int, default=0, help='if > 0, training contexts have K..K_max facts (uniform per batch)'); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--eval_every', type=int, default=250); ap.add_argument('--n_eval', type=int, default=256); ap.add_argument('--n_final', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--eval_only', action='store_true'); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_name); kg = KG(tok)
    print(f"names {len(kg.names)} (held-out {len(kg.held)}), relations {len(RELS)}, K={args.K}", flush=True)
    torch.manual_seed(args.seed); rng = random.Random(args.seed); dev = args.device
    os.makedirs(args.out, exist_ok=True); res = {}

    # Arms A and C: untouched model
    base = build(args.ckpt, dev, struct=False).eval()
    for split in ("new_triples", "new_names"):
        res[f"A_text_{split}"] = hit1_text(base, kg, args.K, split, args.n_final, dev, 1000, bare=False)
        res[f"C_bare_{split}"] = hit1_text(base, kg, args.K, split, args.n_final, dev, 1000, bare=True)
    print("untouched model, facts as sentences (A) / bare pairs (C):", {k: round(v, 3) for k, v in res.items()}, flush=True)
    del base; torch.cuda.empty_cache() if dev == 'cuda' else None

    # Arm B: adapted (frozen base, far path + angle tables)
    model = build(args.ckpt, dev, struct=True)
    new_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in new_params):,}", flush=True)
    if not args.eval_only:
        opt = torch.optim.AdamW(new_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
        scaler = torch.amp.GradScaler('cuda', enabled=(dev == 'cuda'))
        t0 = time.time()
        for it in range(args.steps + 1):
            lr = args.lr * it / args.warmup if it < args.warmup else args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (it - args.warmup) / max(1, args.steps - args.warmup))))
            for g in opt.param_groups: g['lr'] = lr
            if it % args.eval_every == 0:
                h1 = hit1_units(model, kg, args.K, "new_triples", args.n_eval, dev, 2000); h2 = hit1_units(model, kg, args.K, "new_names", args.n_eval, dev, 3000)
                h3 = hit1_units(model, kg, 64, "new_triples", args.n_eval // 2, dev, 5000)
                print(f"step {it:5d} | hit@1 units: new triples {h1:.3f}  new names {h2:.3f}  K=64 {h3:.3f} | {time.time()-t0:.0f}s", flush=True)
            if it == args.steps: break
            model.train()
            Kt = rng.randint(args.K, args.K_max) if args.K_max > args.K else args.K
            ids, meta, y = batch_units(kg, rng, args.batch, Kt, "train", dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=(dev == 'cuda')):
                logits, _ = model(ids, meta=meta)
                loss = F.cross_entropy(logits[:, -1].float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(new_params, 1.0); scaler.step(opt); scaler.update()
            if it % 50 == 0: print(f"  iter {it} loss {loss.item():.3f} lr {lr:.2e}", flush=True)
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, os.path.join(args.out, 'new_params.pt'))
    else:
        sd = torch.load(os.path.join(args.out, 'new_params.pt'), map_location=dev); model.load_state_dict(sd, strict=False)
    for split in ("new_triples", "new_names"):
        res[f"B_units_{split}"] = hit1_units(model, kg, args.K, split, args.n_final, dev, 4000)
    res["B_units_new_triples_K64"] = hit1_units(model, kg, 64, "new_triples", args.n_final, dev, 5000)
    res["A_text_new_triples_K64"] = hit1_text(build(args.ckpt, dev, struct=False), kg, 64, "new_triples", args.n_final, dev, 5000, bare=False)
    print("RESULT", json.dumps({k: round(v, 3) for k, v in res.items()}), flush=True)
    json.dump(res, open(os.path.join(args.out, 'results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
