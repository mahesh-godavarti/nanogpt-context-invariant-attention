# Copyright (c) 2026 Mahesh Godavarti
# Licensed under CC BY-NC-SA 4.0. See LICENSE file for details.

"""
GPT Language Model with two-score segment attention (FlexAttention).

Three arms:
  B   -- Standard RoPE + FlashAttention/SDPA
  Jm  -- RoPE resets at each delimiter + content addresses + two-score attention
  Jmr -- RoPE resets at each delimiter + random addresses + two-score attention

Two-score attention (Jm / Jmr). Every query attends causally to every key,
but the score depends on whether the pair lies in the same segment:

  same segment:   q_rot . k_rot      pure reset-RoPE (position within segment)
  other segment:  q_rot . k_addr     K additionally rotated by its segment's
                                     content address (K-side addressing only)

Q never carries an address, so nothing a query computes depends on tokens
after it. K's address is the mean of its own (completed) segment, and a key is
only scored with its address by queries in later segments.

Implementation: the two scores are realised as one softmax over a doubled key
set [k_rot ; k_addr] with a mask that admits exactly one copy per (query, key)
pair. This is algebraically identical to selecting between two score matrices
and lets FlexAttention compile it into a block-sparse fused kernel. On CPU it
falls back to SDPA with an explicit boolean mask (used by causality_test.py).

Based on nanoGPT by Andrej Karpathy.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _FLEX_AVAILABLE = True
except ImportError:
    _FLEX_AVAILABLE = False

# ============================================================================
# RoPE utilities
# ============================================================================

def make_base_freq(n_embd):
    """Standard RoPE frequency schedule: 1/10000^(2i/d) for each dim pair."""
    half = n_embd // 2
    return 1.0 / (10000 ** (torch.arange(0, half).float() / half))


def apply_rotation(x, angles):
    """Apply RoPE rotation. x: (..., D), angles: (..., D//2)."""
    # Compute cos/sin in float32 for precision, then cast to x's dtype
    # so autocast (bf16/fp16) is preserved through the multiplication.
    cos_a = torch.cos(angles).to(x.dtype)
    sin_a = torch.sin(angles).to(x.dtype)
    x_pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
    r0 = x0 * cos_a - x1 * sin_a
    r1 = x0 * sin_a + x1 * cos_a
    return torch.stack([r0, r1], dim=-1).reshape_as(x)

# ============================================================================
# LayerNorm with optional bias
# ============================================================================

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# ============================================================================
# Attention variants
# ============================================================================

class RoPEAttention(nn.Module):
    """Standard causal attention with RoPE on Q and K. Uses SDPA/FlashAttention."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

    def forward(self, x, angles):
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Apply RoPE: angles is (B, T, C//2), split across heads -> (B, H, T, D//2)
        a = angles.view(B, T, H, D // 2).transpose(1, 2)
        q = apply_rotation(q, a)
        k = apply_rotation(k, a)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class TwoScoreAttention(nn.Module):
    """Two-score segment attention.

    plain score (same segment):   q_rot . k_rot
    cross score (other segment):  q_rot . k_addr   (k_addr = k_rot rotated by
                                                    the key's segment address)

    Realised as a single softmax over the doubled key set [k_rot ; k_addr]
    (values duplicated) with a (T x 2T) mask that admits the plain copy for
    same-segment pairs and the addressed copy for cross-segment pairs, both
    subject to causality. Uses FlexAttention on CUDA and SDPA with an explicit
    boolean mask on CPU.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

    def forward(self, x, rope, addr, block_mask=None, attn_mask=None):
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # (B, T, C//2) -> (B, H, T, D//2)
        r = rope.view(B, T, H, D // 2).transpose(1, 2)
        a = addr.view(B, T, H, D // 2).transpose(1, 2)

        q_rot = apply_rotation(q, r)        # Q: reset-RoPE only, never an address
        k_rot = apply_rotation(k, r)        # K for same-segment pairs
        k_addr = apply_rotation(k_rot, a)   # K for cross-segment pairs

        k2 = torch.cat([k_rot, k_addr], dim=2)  # (B, H, 2T, D)
        v2 = torch.cat([v, v], dim=2)           # (B, H, 2T, D)

        if block_mask is not None:
            # FlexAttention path (CUDA)
            y = flex_attention(q_rot, k2, v2, block_mask=block_mask)
        elif attn_mask is not None:
            # SDPA fallback (CPU or no FlexAttention); attn_mask: (B, 1, T, 2T) bool
            y = F.scaled_dot_product_attention(q_rot, k2, v2, attn_mask=attn_mask)
        else:
            raise ValueError("TwoScoreAttention requires block_mask or attn_mask")

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

# ============================================================================
# Transformer Block + MLP
# ============================================================================

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        if config.model_type == 'B':
            self.attn = RoPEAttention(config)
        else:  # Jm, Jmr
            self.attn = TwoScoreAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.model_type = config.model_type

    def forward(self, x, **kwargs):
        if self.model_type == 'B':
            x = x + self.attn(self.ln_1(x), kwargs['angles'])
        else:
            x = x + self.attn(self.ln_1(x), kwargs['rope'], kwargs['addr'],
                              block_mask=kwargs.get('block_mask'),
                              attn_mask=kwargs.get('attn_mask'))
        x = x + self.mlp(self.ln_2(x))
        return x

# ============================================================================
# GPT Config and Model
# ============================================================================

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    model_type: str = 'B'        # 'B', 'Jm', 'Jmr'
    delimiter_id: int = 628      # token id for \n\n in GPT-2 BPE


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Token embedding (no learned positional embedding -- we use RoPE)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # RoPE base frequencies
        self.register_buffer('base_freq', make_base_freq(config.n_embd))

        # Segment addressing (Jm only -- content-based)
        if config.model_type == 'Jm':
            self.seg_ln = LayerNorm(config.n_embd, bias=config.bias)
            self.seg_proj = nn.Linear(config.n_embd, config.n_embd // 2, bias=config.bias)

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Segment detection and addressing
    # ------------------------------------------------------------------

    def _segment_info(self, idx):
        """Detect delimiter boundaries, compute seg_ids and pos_in_seg.

        The delimiter token is the LAST token of a segment; the token after
        it starts a new segment at position 0. seg_ids[i] and pos_in_seg[i]
        depend only on tokens at positions < i.
        """
        B, T = idx.shape
        device = idx.device
        is_delim = (idx == self.config.delimiter_id)
        starts = torch.zeros(B, T, dtype=torch.bool, device=device)
        starts[:, 0] = True
        starts[:, 1:] = is_delim[:, :-1]
        seg_ids = starts.long().cumsum(dim=1) - 1  # (B, T)
        gpos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        marker = torch.where(starts, gpos, torch.zeros_like(gpos))
        seg_start, _ = marker.cummax(dim=1)
        pos_in_seg = (gpos - seg_start).float()
        return seg_ids, pos_in_seg

    def _content_addresses(self, x, rope, seg_ids):
        """Full-segment mean of rotated embeddings -> LN -> proj -> addr.

        Rotate-then-pool: apply reset-RoPE to the token embeddings, scatter-add
        per segment, mean-pool, project to angle space. Each token receives its
        own segment's address; it is only ever consumed on the K side, by
        queries in LATER segments, so the address of the final (possibly
        unfinished) segment is computed but never used.
        """
        B, T, C = x.shape
        half = C // 2
        device = x.device

        rotated = apply_rotation(x, rope)  # (B, T, C)
        n_segs = seg_ids.max().item() + 1

        lid_exp = seg_ids.unsqueeze(-1).expand(-1, -1, C)  # (B, T, C)
        sums = torch.zeros(B, n_segs, C, device=device, dtype=rotated.dtype)
        sums.scatter_add_(1, lid_exp, rotated)
        counts = torch.zeros(B, n_segs, 1, device=device, dtype=rotated.dtype)
        counts.scatter_add_(1, seg_ids.unsqueeze(-1),
                            torch.ones(B, T, 1, device=device, dtype=rotated.dtype))
        pooled = sums / counts.clamp(min=1)  # (B, n_segs, C)
        seg_angles = self.seg_proj(self.seg_ln(pooled))  # (B, n_segs, half)

        addr = seg_angles.gather(
            1, seg_ids.unsqueeze(-1).expand(-1, -1, half))  # (B, T, half)
        return addr

    # ------------------------------------------------------------------
    # Attention masks over the doubled key set [k_rot ; k_addr]
    # ------------------------------------------------------------------
    #
    # Key index kv < T refers to k_rot[kv] (plain copy), kv >= T refers to
    # k_addr[kv - T] (addressed copy). For a query i and underlying key j:
    #   plain copy allowed  iff  i >= j and same_seg(i, j)
    #   addr  copy allowed  iff  i >= j and not same_seg(i, j)
    # Exactly one copy is admitted per causal (i, j) pair.

    def _build_block_mask(self, seg_ids):
        """FlexAttention BlockMask of shape (B, 1, T, 2T)."""
        B, T = seg_ids.shape

        def mask_mod(b, h, q_idx, kv_idx):
            plain = kv_idx < T
            j = torch.where(plain, kv_idx, kv_idx - T)
            same = seg_ids[b, q_idx] == seg_ids[b, j]
            return (q_idx >= j) & (plain == same)

        return create_block_mask(mask_mod, B, None, T, 2 * T, device=seg_ids.device)

    def _build_attn_mask(self, seg_ids):
        """SDPA-compatible boolean mask (CPU fallback), shape (B, 1, T, 2T)."""
        B, T = seg_ids.shape
        device = seg_ids.device
        causal = torch.ones(T, T, device=device, dtype=torch.bool).tril().unsqueeze(0)
        same = (seg_ids.unsqueeze(2) == seg_ids.unsqueeze(1))  # (B, T, T)
        mask = torch.cat([causal & same, causal & ~same], dim=2)  # (B, T, 2T)
        return mask.unsqueeze(1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        if self.config.model_type == 'B':
            # Standard continuous RoPE
            pos = torch.arange(t, device=device, dtype=torch.float)
            angles = torch.outer(pos, self.base_freq).unsqueeze(0).expand(b, -1, -1)
            for block in self.transformer.h:
                x = block(x, angles=angles)

        elif self.config.model_type in ('Jm', 'Jmr'):
            seg_ids, pos_in_seg = self._segment_info(idx)
            rope = pos_in_seg.unsqueeze(-1) * self.base_freq  # (B, T, C//2)

            half = self.config.n_embd // 2
            if self.config.model_type == 'Jmr':
                # Random i.i.d. address per segment (ablation control)
                n_segs = seg_ids.max().item() + 1
                seg_addr = torch.randn(b, n_segs, half, device=device)
                addr = seg_addr.gather(
                    1, seg_ids.unsqueeze(-1).expand(-1, -1, half))
            else:
                # Content-based address per segment
                addr = self._content_addresses(x, rope, seg_ids)

            # FlexAttention on CUDA, SDPA fallback on CPU
            if device.type == 'cuda' and _FLEX_AVAILABLE:
                block_mask = self._build_block_mask(seg_ids)
                for block in self.transformer.h:
                    x = block(x, rope=rope, addr=addr, block_mask=block_mask)
            else:
                attn_mask = self._build_attn_mask(seg_ids)
                for block in self.transformer.h:
                    x = block(x, rope=rope, addr=addr, attn_mask=attn_mask)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        config_args['model_type'] = 'B'  # pretrained weights are standard GPT-2
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias')
                      and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_keys_hf:
            if 'wpe' in k:
                print(f"  skipping pretrained key: {k} (using RoPE instead)")
                continue
            if k not in sd:
                print(f"  skipping pretrained key: {k} (not in model)")
                continue
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12  # A100 bfloat16 peak
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
