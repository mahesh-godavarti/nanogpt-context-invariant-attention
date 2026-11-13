# Copyright (c) 2026 Mahesh Godavarti
# Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE file for details.

"""
GPT Language Model with two-score segment attention (FlexAttention).

Five arms:
  B   -- Standard RoPE + FlashAttention/SDPA
  A   -- ALiBi: no rotation, per-head linear distance penalty on the scores (SDPA, float mask)
  R   -- RoPE resets at each delimiter, no addresses (plain causal attention)
  Jm  -- RoPE resets at each delimiter + content addresses + two-score attention
  Jmr -- RoPE resets at each delimiter + random addresses + two-score attention

R isolates the position reset from the addressing: it is Jm with a zero
address, under which the two key copies coincide and the two-score softmax
is ordinary causal attention over reset positions.

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
import torch.utils.checkpoint
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

def make_base_freq(n_embd, base=10000.0, n_head=None, rotary_dims=0):
    """RoPE frequency schedule over the C//2 rotation pairs.
    n_head None: one schedule across the whole embedding, 1/base^(i/(C/2)) (this repo's original models).
    n_head given: the standard per-head schedule 1/base^(2j/D), j < D/2, repeated for every head (LLaMA/SmolLM layout)."""
    half = n_embd // 2
    if n_head is None:
        return 1.0 / (base ** (torch.arange(0, half).float() / half))
    hd2 = half // n_head
    if rotary_dims:   # partial rotary (GPT-NeoX/Pythia): only the first rotary_dims dims of each head rotate; the rest get frequency 0
        rd2 = rotary_dims // 2
        f = torch.zeros(hd2); f[:rd2] = 1.0 / (base ** (torch.arange(0, rd2).float() / rd2))
    else:
        f = 1.0 / (base ** (torch.arange(0, hd2).float() / hd2))
    return f.repeat(n_head)


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
        self.rotate_v = config.rotate_v

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
        if self.rotate_v:
            v = apply_rotation(v, a)                  # value transport: V in the key's frame

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        if self.rotate_v:
            y = apply_rotation(y, -a)                 # back into the query's frame: out_m = sum_n p_mn R(a_n - a_m) v_n
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class ALiBiAttention(nn.Module):
    """Standard causal attention with ALiBi: no rotation, a per-head linear distance penalty
    -m_h * (i - j) added to the scores. Slopes m_h = 2^(-8h/H), h = 1..H (Press et al., 2022).
    Uses SDPA with an explicit float mask (the bias and the causal mask combined)."""

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
        H = config.n_head
        self.register_buffer('slopes', torch.tensor([2 ** (-8.0 * h / H) for h in range(1, H + 1)]))

    def forward(self, x):
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        pos = torch.arange(T, device=x.device)
        if T <= 4096:
            rel = pos.unsqueeze(1) - pos.unsqueeze(0)                       # (T, T): i - j
            bias = -self.slopes.view(H, 1, 1) * rel.to(q.dtype).unsqueeze(0)   # (H, T, T)
            bias = bias.masked_fill(rel.unsqueeze(0) < 0, float('-inf'))      # causal
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias.unsqueeze(0),
                dropout_p=self.dropout if self.training else 0.0)
        else:
            # Long contexts (evaluation): same computation in query chunks, so the (H, T, T)
            # bias is never materialised. Identical result; no dropout (eval only).
            y = self._chunked(q, k, v, pos)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

    def _chunked(self, q, k, v, pos, chunk=512):
        B, H, T, D = q.shape
        scale = 1.0 / math.sqrt(D)
        out = torch.empty_like(q)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            rel = pos[s:e].unsqueeze(1) - pos.unsqueeze(0)                   # (c, T)
            bias = -self.slopes.view(1, H, 1, 1) * rel.to(q.dtype).view(1, 1, e - s, T)
            bias = bias.masked_fill(rel.view(1, 1, e - s, T) < 0, float('-inf'))
            scores = (q[:, :, s:e] @ k.transpose(-2, -1)) * scale + bias
            out[:, :, s:e] = F.softmax(scores.float(), dim=-1).to(v.dtype) @ v
        return out


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
        self.rotate_v = config.rotate_v

    def forward(self, x, rope, addr, block_mask=None, attn_mask=None, lam=None, shift=None):
        """
        lam:   optional (B, T) per-query multiplier applied to cross-segment scores only.
        shift: optional (B, T) per-query offset subtracted from cross-segment scores only.
        """
        if shift is not None and lam is None:
            lam = torch.ones_like(shift)
        if lam is not None and shift is None:
            shift = torch.zeros_like(lam)
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
        if self.rotate_v:
            v_rot = apply_rotation(v, r)         # plain copy: V in the key's reset-RoPE frame
            v_addr = apply_rotation(v_rot, a)    # addressed copy: plus the key's segment address
            v2 = torch.cat([v_rot, v_addr], dim=2)
        else:
            v2 = torch.cat([v, v], dim=2)           # (B, H, 2T, D)

        y = two_score_attend(q_rot, k2, v2, block_mask, attn_mask, lam, shift)
        if self.rotate_v:
            y = apply_rotation(y, -r)            # back into the query's reset-RoPE frame
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


def two_score_attend(q_rot, k2, v2, block_mask=None, attn_mask=None, lam=None, shift=None, chunk_above=2048):
    """Softmax over the doubled key set. FlexAttention with a block mask on CUDA; SDPA with a boolean mask otherwise;
    the explicit chunked form when a per-query cross-segment multiplier/offset is given or the context is long
    (SDPA with an arbitrary mask falls back to materialising the full (T x 2T) score matrix)."""
    T = q_rot.shape[2]
    if block_mask is not None:
        if lam is None:
            return flex_attention(q_rot, k2, v2, block_mask=block_mask)
        def score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(kv_idx >= T, score * lam[b, q_idx] - shift[b, q_idx], score)
        return flex_attention(q_rot, k2, v2, block_mask=block_mask, score_mod=score_mod)
    if attn_mask is None:
        raise ValueError("two-score attention requires block_mask or attn_mask")
    if lam is None and T <= chunk_above:
        return F.scaled_dot_product_attention(q_rot, k2, v2, attn_mask=attn_mask)
    if lam is None:
        lam = torch.ones(q_rot.shape[0], T, device=q_rot.device, dtype=q_rot.dtype)
        shift = torch.zeros_like(lam)
    return _chunked_two_score(q_rot, k2, v2, attn_mask, lam, shift)


def _chunked_two_score(q_rot, k2, v2, attn_mask, lam, shift, chunk=512):
    """Explicit two-score attention with per-query cross-segment multiplier and offset,
    computed in query chunks so long contexts do not materialise a full score matrix.
    v2 = [v_plain ; v_cross] (identical copies without value transport)."""
    B, H, T, D = q_rot.shape
    scale = 1.0 / math.sqrt(D)
    k_rot, k_addr = k2[:, :, :T], k2[:, :, T:]
    v_plain, v_cross = v2[:, :, :T], v2[:, :, T:]
    out = torch.empty_like(q_rot)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        qc = q_rot[:, :, s:e]
        with torch.autocast(device_type=q_rot.device.type, enabled=False):   # scores in float32 (fp16 overflow safety)
            qs = qc.float() * scale
            plain = qs @ k_rot.float().transpose(-2, -1)                     # (B, H, c, T)
            cross = ((qs @ k_addr.float().transpose(-2, -1)) * lam[:, None, s:e, None].float()
                     - shift[:, None, s:e, None].float())
            m_plain = attn_mask[:, :, s:e, :T]
            m_cross = attn_mask[:, :, s:e, T:]
            scores = torch.where(m_plain, plain,
                                 torch.where(m_cross, cross, torch.full_like(plain, float('-inf'))))
            p = F.softmax(scores, dim=-1).to(v2.dtype)
        out[:, :, s:e] = (p * m_plain) @ v_plain + (p * m_cross) @ v_cross
    return out


def window_two_score_attend(q_near, k_near, q_far, k_far, v, window, far_bias, shift=None, chunk=512, fact=None, seg=None):
    """Windowed two-score attention, computed in query chunks.
    near score (0 <= i-j < window):  q_near . k_near   (continuous RoPE on both, as pretrained)
    far score  (i-j >= window):      q_far . k_far + far_bias[h] - shift[b,i]   (reset-RoPE + address on k_far)
    One softmax over both; exactly one score per causal pair. v is shared by both copies."""
    B, H, T, D = q_near.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q_near)
    j = torch.arange(T, device=q_near.device)
    fb = far_bias.view(1, H, 1, 1).to(q_near.dtype)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        dist = torch.arange(s, e, device=q_near.device).unsqueeze(1) - j.unsqueeze(0)       # (c, T)
        m_near = (dist >= 0) & (dist < window)
        m_far = dist >= window
        if fact is not None:
            # structured units (fact tokens): a fact key is ALWAYS scored by the far path; a fact query attends only inside its own
            # unit (far path); text-text pairs follow the window rule.  fact, seg: (B, T) -> masks (B, 1, c, T)
            qf = fact[:, None, s:e, None]; kf = fact[:, None, None, :]
            same = (seg[:, None, s:e, None] == seg[:, None, None, :])
            causal = (dist >= 0)[None, None]
            m_far = causal & ((qf & same) | (~qf & (kf | (dist >= window)[None, None])))
            m_near = causal & ~qf & ~kf & (dist < window)[None, None]
        # scores in float32 with autocast OFF (autocast would re-cast the matmul to fp16, where the attention scores of some
        # pretrained models -- Pythia, deep layers -- overflow)
        with torch.autocast(device_type=q_near.device.type, enabled=False):
            near = (q_near[:, :, s:e].float() * scale) @ k_near.float().transpose(-2, -1)      # (B, H, c, T)
            far = (q_far[:, :, s:e].float() * scale) @ k_far.float().transpose(-2, -1) + fb.float()
            if shift is not None:
                far = far - shift[:, None, s:e, None].float()
            scores = torch.where(m_near, near, torch.where(m_far, far, torch.full_like(near, float('-inf'))))
            p = F.softmax(scores, dim=-1).to(v.dtype)
        out[:, :, s:e] = p @ v
    return out


# ============================================================================
# LLaMA-style pieces (for adapting public pretrained models such as SmolLM): RMSNorm, SwiGLU MLP,
# grouped-query attention with separate q/k/v/o projections. The q/k projections are stored in
# interleaved-pair layout (see GPT.from_hf_llama), so apply_rotation applies unchanged.
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * xf).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class LlamaAttention(nn.Module):
    """Grouped-query attention. Plain causal RoPE attention when called with angles (model B),
    two-score segment attention when called with rope and addr (R / Jm / Jmr)."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        ab = config.attn_bias
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=ab)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=ab)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=ab)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=ab)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.rotate_v = config.rotate_v
        self.window = config.window
        if config.window > 0:
            self.far_bias = nn.Parameter(torch.full((self.n_head,), float(config.far_bias_init)))
        self.far_proj = config.far_proj and config.window > 0
        if self.far_proj:   # far-path query/key projections; copied from q_proj/k_proj by init_far_proj()
            self.q_far_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.attn_bias)
            self.k_far_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.attn_bias)

    def init_far_proj(self):
        with torch.no_grad():
            self.q_far_proj.weight.copy_(self.q_proj.weight); self.k_far_proj.weight.copy_(self.k_proj.weight)
            if self.q_far_proj.bias is not None:
                self.q_far_proj.bias.copy_(self.q_proj.bias); self.k_far_proj.bias.copy_(self.k_proj.bias)

    def _qkv(self, x):
        B, T, C = x.size()
        H, Hk, D = self.n_head, self.n_kv_head, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, Hk, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hk, D).transpose(1, 2)
        if Hk != H:
            k = k.repeat_interleave(H // Hk, dim=1)   # head h uses kv head h // (H/Hk), as HF repeat_kv
            v = v.repeat_interleave(H // Hk, dim=1)
        return q, k, v

    def forward(self, x, angles=None, rope=None, addr=None, block_mask=None, attn_mask=None, lam=None, shift=None, rope_near=None, window=None, fact=None, seg=None):
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim
        q, k, v = self._qkv(x)
        if rope_near is not None:
            # window mode: near pairs continuous RoPE (pretrained), far pairs reset-RoPE + address
            rn = rope_near.view(B, T, H, D // 2).transpose(1, 2)
            rf = rope.view(B, T, H, D // 2).transpose(1, 2)
            a = addr.view(B, T, H, D // 2).transpose(1, 2)
            if self.far_proj:   # far path with its own projections
                Hk = self.n_kv_head
                qf = self.q_far_proj(x).view(B, T, H, D).transpose(1, 2)
                kf = self.k_far_proj(x).view(B, T, Hk, D).transpose(1, 2)
                if Hk != H:
                    kf = kf.repeat_interleave(H // Hk, dim=1)
            else:
                qf, kf = q, k
            y = window_two_score_attend(apply_rotation(q, rn), apply_rotation(k, rn),
                                        apply_rotation(qf, rf), apply_rotation(kf, rf + a), v, window or self.window, self.far_bias, shift,
                                        fact=fact, seg=seg)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_dropout(self.o_proj(y))
        if rope is None:
            a = angles.view(B, T, H, D // 2).transpose(1, 2)
            q = apply_rotation(q, a); k = apply_rotation(k, a)
            if self.rotate_v:
                v = apply_rotation(v, a)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
            if self.rotate_v:
                y = apply_rotation(y, -a)
        else:
            if shift is not None and lam is None:
                lam = torch.ones_like(shift)
            if lam is not None and shift is None:
                shift = torch.zeros_like(lam)
            r = rope.view(B, T, H, D // 2).transpose(1, 2)
            a = addr.view(B, T, H, D // 2).transpose(1, 2)
            q_rot = apply_rotation(q, r)
            k_rot = apply_rotation(k, r)
            k_addr = apply_rotation(k_rot, a)
            k2 = torch.cat([k_rot, k_addr], dim=2)
            if self.rotate_v:
                v_rot = apply_rotation(v, r)
                v2 = torch.cat([v_rot, apply_rotation(v_rot, a)], dim=2)
            else:
                v2 = torch.cat([v, v], dim=2)
            y = two_score_attend(q_rot, k2, v2, block_mask, attn_mask, lam, shift)
            if self.rotate_v:
                y = apply_rotation(y, -r)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(y))

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
        self.arch = config.arch
        if config.arch == 'neox':
            assert config.model_type in ('B', 'R', 'Jm', 'Jmr'), "neox arch: B or two-score models only"
            self.ln_1 = LayerNorm(config.n_embd, bias=True)
            self.attn = LlamaAttention(config)          # separate q/k/v/o with biases (converted from the fused NeoX projection)
            self.ln_2 = LayerNorm(config.n_embd, bias=True)
            self.mlp = MLP(config)                      # GELU MLP with biases (config.bias = True)
        elif config.arch == 'llama':
            assert config.model_type in ('B', 'R', 'Jm', 'Jmr'), "llama arch: B or two-score models only"
            self.ln_1 = RMSNorm(config.n_embd, config.rms_eps)
            self.attn = LlamaAttention(config)
            self.ln_2 = RMSNorm(config.n_embd, config.rms_eps)
            self.mlp = SwiGLU(config)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            if config.model_type == 'B':
                self.attn = RoPEAttention(config)
            elif config.model_type == 'A':
                self.attn = ALiBiAttention(config)
            else:  # R, Jm, Jmr
                self.attn = TwoScoreAttention(config)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            self.mlp = MLP(config)
        self.model_type = config.model_type

    def forward(self, x, **kwargs):
        if self.arch == 'neox':   # parallel residual: x + attn(ln1(x)) + mlp(ln2(x))
            if self.model_type == 'B':
                a = self.attn(self.ln_1(x), angles=kwargs['angles'])
            else:
                a = self.attn(self.ln_1(x), rope=kwargs['rope'], addr=kwargs['addr'],
                              block_mask=kwargs.get('block_mask'), attn_mask=kwargs.get('attn_mask'),
                              lam=kwargs.get('lam'), shift=kwargs.get('shift'), rope_near=kwargs.get('rope_near'), window=kwargs.get('window'), fact=kwargs.get('fact'), seg=kwargs.get('seg'))
            return x + a + self.mlp(self.ln_2(x))
        if self.arch == 'llama':
            if self.model_type == 'B':
                x = x + self.attn(self.ln_1(x), angles=kwargs['angles'])
            else:
                x = x + self.attn(self.ln_1(x), rope=kwargs['rope'], addr=kwargs['addr'],
                                  block_mask=kwargs.get('block_mask'), attn_mask=kwargs.get('attn_mask'),
                                  lam=kwargs.get('lam'), shift=kwargs.get('shift'), rope_near=kwargs.get('rope_near'), window=kwargs.get('window'), fact=kwargs.get('fact'), seg=kwargs.get('seg'))
        elif self.model_type == 'B':
            x = x + self.attn(self.ln_1(x), kwargs['angles'])
        elif self.model_type == 'A':
            x = x + self.attn(self.ln_1(x))
        else:
            x = x + self.attn(self.ln_1(x), kwargs['rope'], kwargs['addr'],
                              block_mask=kwargs.get('block_mask'),
                              attn_mask=kwargs.get('attn_mask'),
                              lam=kwargs.get('lam'), shift=kwargs.get('shift'))
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
    model_type: str = 'B'        # 'B', 'A', 'R', 'Jm', 'Jmr'
    # Architecture: 'gpt' (this repo's models: LayerNorm, GELU MLP, fused qkv, one RoPE schedule across the embedding)
    # or 'llama' (RMSNorm, SwiGLU, grouped-query attention, per-head RoPE schedule) for public pretrained models (SmolLM).
    arch: str = 'gpt'            # 'gpt' | 'llama' | 'neox' (GPT-NeoX/Pythia: LayerNorm+bias, parallel residual, GELU MLP, partial rotary, untied head)
    rotary_dims: int = 0         # neox: rotary dims per head (0 = all)
    attn_bias: bool = False      # llama/neox attention projections with bias
    untied_head: bool = False    # separate lm_head weight (not tied to wte)
    n_kv_head: int = 0           # llama: number of key/value heads (0 = n_head)
    intermediate_size: int = 0   # llama: SwiGLU hidden size
    rms_eps: float = 1e-5
    rope_base: float = 10000.0
    rope_per_head: bool = False  # per-head RoPE schedule repeated over heads (llama) instead of one schedule over the embedding
    delimiter_ids: tuple = ()    # if non-empty, the set of segment-delimiter token ids (overrides delimiter_id)
    # Adaptation of a pretrained continuous-RoPE model (B) to content addresses (user 2026-10-26):
    #   key/query angle = pos_in_seg*f + (1-alpha)*seg_start*f ; cross-segment key angle adds alpha*content.
    #   alpha = 0: continuous RoPE exactly (no distribution shift); alpha = 1: reset-RoPE + content address (Jm).
    #   The content map's gradient is NOT scaled by alpha (straight-through on the alpha factor).
    #   adapt_slow_frac < 1: only that fraction of the frequency bands (the slowest) is annealed; the fast bands keep
    #   continuous RoPE and carry no address.
    adapt: bool = False
    adapt_alpha: float = 1.0
    adapt_slow_frac: float = 1.0
    # Window mode (user 2026-10-28, Jm only): keys closer than `window` tokens are scored exactly as the pretrained model scores them
    # (continuous RoPE, original weights); keys `window` or more tokens back are scored with reset-RoPE + the content address.
    # On inputs no longer than `window` the model IS the pretrained model. Far scores carry a learned per-head bias (init far_bias_init)
    # so the start point is a sliding-window model; no anneal. 0 = off (segment rule as before).
    window: int = 0
    far_bias_init: float = -4.0
    n_slots: int = 0                # structured units: number of slot roles (0 = none); learned per-plane slot angles
    n_rels: int = 0                 # structured units: number of relation types; learned per-plane relation angles
    far_proj: bool = False          # window mode: separate q/k projections for the far path (initialised from the pretrained ones)
    freeze_pretrained: bool = False # train only the adaptation parameters (address map, far biases, far projections)
    grad_checkpoint: bool = False   # window mode: recompute each block in backward (fits 4096-token training on a 15 GB GPU)
    rotate_v: bool = False       # value transport: rotate V by the key's angles (RoPE, plus the address on the addressed copy) and un-rotate the output by the query's RoPE angle
    delimiter_id: int = 764      # token id for " ." in GPT-2 BPE (sentence end; "\n\n" = 628 does not occur in the prepared WikiText-103)
    use_flex: bool = True        # Jm/Jmr on CUDA: FlexAttention (True) or SDPA with a boolean mask (False)
    # Cross-segment score scaling (Jm/Jmr): lambda_i = max(1, 1 + s * log(n_i / n_ref)),
    # n_i = number of segments before query i's segment. Keeps the cross-segment softmax
    # from flattening as the segment count grows. s = 0 disables it.
    cross_scale_s: float = 0.0
    cross_scale_ref: float = 32.0
    # Cross-segment score offset: subtract c * log(n_i / n_ref) (clamped at 0) from every
    # cross-segment score, which divides the total cross-segment attention mass by
    # (n_i / n_ref)^c and keeps it near its training-length level. Uniform over all
    # cross-segment keys, so no distance preference. c = 0 disables it.
    cross_shift_c: float = 0.0
    # Dynamic NTK scaling for model B at inference: when the sequence is longer than
    # rope_ref_len, the RoPE base is raised by (T / rope_ref_len)^(d / (d - 2)) so the
    # lowest frequency's wavelength grows with the context. Standard inference-time
    # length correction for RoPE; off by default.
    rope_ntk: bool = False
    rope_ref_len: int = 1024


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
            ln_f = RMSNorm(config.n_embd, config.rms_eps) if config.arch == 'llama' else LayerNorm(config.n_embd, bias=(config.bias or config.arch == 'neox')),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if not config.untied_head:
            self.transformer.wte.weight = self.lm_head.weight

        # RoPE base frequencies
        self.register_buffer('base_freq', make_base_freq(config.n_embd, config.rope_base, config.n_head if config.rope_per_head else None, config.rotary_dims))

        # Segment addressing (Jm only -- content-based)
        if config.model_type == 'Jm':
            self.seg_ln = LayerNorm(config.n_embd, bias=config.bias)
            self.seg_proj = nn.Linear(config.n_embd, config.n_embd // 2, bias=config.bias)
            if config.n_slots > 0:   # structured units: angle of a fact token = offset*f + slot_angle[slot] + rel_angle[rel]
                self.slot_angles = nn.Parameter(torch.zeros(config.n_slots, config.n_embd // 2))
                self.rel_angles = nn.Parameter(torch.zeros(max(config.n_rels, 1), config.n_embd // 2))

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
        if self.config.delimiter_ids:
            is_delim = torch.isin(idx, torch.tensor(self.config.delimiter_ids, device=device))
        else:
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

    def forward(self, idx, targets=None, meta=None):
        """meta (structured units, Jm window mode only): dict with (B, T) tensors seg (segment id), off (offset within the
        segment/slot), fact (bool: token belongs to a structured unit), slot, rel (indices into the learned angle tables)."""
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        if self.config.model_type == 'B':
            # Standard continuous RoPE (optionally NTK-scaled beyond the reference length)
            pos = torch.arange(t, device=device, dtype=torch.float)
            base_freq = self.base_freq
            if self.config.rope_ntk and t > self.config.rope_ref_len:
                d = self.config.n_embd
                s = t / self.config.rope_ref_len
                dd = d // self.config.n_head if self.config.rope_per_head else d
                dd = self.config.rotary_dims or dd
                base_freq = make_base_freq(d, base=self.config.rope_base * s ** (dd / (dd - 2)),
                                           n_head=self.config.n_head if self.config.rope_per_head else None, rotary_dims=self.config.rotary_dims).to(device)
            angles = torch.outer(pos, base_freq).unsqueeze(0).expand(b, -1, -1)
            for block in self.transformer.h:
                x = block(x, angles=angles)

        elif self.config.model_type == 'A':
            # ALiBi: no position encoding at all; the distance penalty lives in the attention
            for block in self.transformer.h:
                x = block(x)

        elif self.config.model_type in ('R', 'Jm', 'Jmr'):
            if meta is not None:
                seg_ids, pos_in_seg = meta['seg'], meta['off'].float()
            else:
                seg_ids, pos_in_seg = self._segment_info(idx)
            rope = pos_in_seg.unsqueeze(-1) * self.base_freq  # (B, T, C//2)
            if meta is not None and 'fact' in meta and self.config.n_slots > 0:
                # structured units: slot + relation angles enter the far angle AND the address pooling (rotate-then-pool), so the
                # unit's address encodes which token is the head and which relation it is, not just the two names
                fact_f = meta['fact'].bool().unsqueeze(-1).to(rope.dtype)
                rope = rope + (self.slot_angles[meta['slot']] + self.rel_angles[meta['rel']]) * fact_f

            half = self.config.n_embd // 2
            if self.config.model_type == 'R':
                # No addresses: reset-RoPE with plain causal attention
                addr = torch.zeros(b, t, half, device=device)
            elif self.config.model_type == 'Jmr':
                # Random i.i.d. address per segment (ablation control)
                n_segs = seg_ids.max().item() + 1
                seg_addr = torch.randn(b, n_segs, half, device=device)
                addr = seg_addr.gather(
                    1, seg_ids.unsqueeze(-1).expand(-1, -1, half))
            else:
                # Content-based address per segment
                addr = self._content_addresses(x, rope, seg_ids)
                if self.config.rotary_dims:
                    addr = addr * (self.base_freq > 0).to(addr.dtype)   # partial rotary: no address on planes that never rotate

            if self.config.adapt and self.config.model_type == 'Jm':
                alpha = float(self.config.adapt_alpha)
                gpos = torch.arange(t, device=device, dtype=torch.float).unsqueeze(0).expand(b, -1)
                n_slow = int(round(self.config.adapt_slow_frac * half))               # the n_slow slowest frequency bands are annealed
                band = torch.zeros(half, device=device); band[torch.argsort(self.base_freq)[:n_slow]] = 1.0  # 1 = annealed, 0 = continuous RoPE
                base_freq = self.base_freq
                if self.config.rope_ntk and t > self.config.rope_ref_len and n_slow < half:
                    # Inference-time NTK scaling of the bands that stayed on continuous RoPE (same rule as model B); annealed bands unchanged.
                    d = self.config.n_embd; dd = d // self.config.n_head if self.config.rope_per_head else d
                    s_ = t / self.config.rope_ref_len
                    ntk_freq = make_base_freq(d, base=self.config.rope_base * s_ ** (dd / (dd - 2)),
                                              n_head=self.config.n_head if self.config.rope_per_head else None).to(device)
                    base_freq = torch.where(band > 0, self.base_freq, ntk_freq)
                    rope = pos_in_seg.unsqueeze(-1) * base_freq
                seg_start = (gpos - pos_in_seg).unsqueeze(-1) * base_freq            # (B, T, half): angle of the segment's first token
                # angle shared by q and both k copies: continuous position in un-annealed bands, (1-alpha)*start + pos_in_seg in annealed ones
                rope = rope + seg_start * (1.0 - alpha * band)
                # cross-segment key extra: alpha * content in annealed bands, with the alpha factor straight-through for the gradient
                addr = (alpha * addr.detach() + addr - addr.detach()) * band

            # Optional cross-segment score scaling / offset by the number of earlier segments.
            lam, shift = None, None
            if self.config.cross_scale_s > 0 or self.config.cross_shift_c > 0:
                n_before = seg_ids.float()  # segments strictly before the query's segment
                log_ratio = torch.log((n_before + 1.0) / self.config.cross_scale_ref).clamp(min=0.0)
                lam = 1.0 + self.config.cross_scale_s * log_ratio     # (B, T), >= 1
                shift = self.config.cross_shift_c * log_ratio          # (B, T), >= 0

            if self.config.window > 0:
                assert self.config.model_type == 'Jm' and self.config.arch in ('llama', 'neox'), "window mode: llama/neox-arch Jm only"
                gpos = torch.arange(t, device=device, dtype=torch.float).unsqueeze(0).expand(b, -1)
                rope_near = gpos.unsqueeze(-1) * self.base_freq          # continuous RoPE for near pairs (as pretrained)
                fact = seg = None
                if meta is not None and 'fact' in meta:
                    fact, seg = meta['fact'].bool(), meta['seg']   # (the slot/relation angles are already in rope, see above)
                for block in self.transformer.h:                          # rope = reset-RoPE for far pairs, addr = content address
                    if self.config.grad_checkpoint and self.training and torch.is_grad_enabled():
                        x = torch.utils.checkpoint.checkpoint(block, x, rope=rope, addr=addr, rope_near=rope_near, shift=shift,
                                                              window=self.config.window, fact=fact, seg=seg, use_reentrant=False)
                    else:
                        x = block(x, rope=rope, addr=addr, rope_near=rope_near, shift=shift, window=self.config.window, fact=fact, seg=seg)
            # FlexAttention on CUDA (unless disabled), SDPA with an explicit mask otherwise.
            # Both paths compute the same attention; SDPA needs far less memory
            # when FlexAttention runs uncompiled.
            elif device.type == 'cuda' and _FLEX_AVAILABLE and self.config.use_flex:
                block_mask = self._build_block_mask(seg_ids)
                for block in self.transformer.h:
                    x = block(x, rope=rope, addr=addr, block_mask=block_mask, lam=lam, shift=shift)
            else:
                attn_mask = self._build_attn_mask(seg_ids)
                for block in self.transformer.h:
                    x = block(x, rope=rope, addr=addr, attn_mask=attn_mask, lam=lam, shift=shift)

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
    def from_hf_llama(cls, name, model_type='B', **overrides):
        """Build this model from a Hugging Face LLaMA-architecture checkpoint (e.g. HuggingFaceTB/SmolLM-135M).
        q/k projection rows are permuted from HF's rotate-half pair layout (j, j+D/2) to the interleaved layout
        (2j, 2j+1) used by apply_rotation, so the pretrained RoPE is reproduced exactly."""
        from transformers import AutoConfig, AutoModelForCausalLM
        hc = AutoConfig.from_pretrained(name)
        rp = getattr(hc, 'rope_parameters', None) or {}
        base = rp.get('rope_theta', getattr(hc, 'rope_theta', 10000.0))
        args = dict(arch='llama', n_layer=hc.num_hidden_layers, n_head=hc.num_attention_heads, n_kv_head=hc.num_key_value_heads,
                    n_embd=hc.hidden_size, intermediate_size=hc.intermediate_size, vocab_size=hc.vocab_size,
                    block_size=hc.max_position_embeddings, bias=False, rms_eps=hc.rms_norm_eps, rope_base=float(base),
                    rope_per_head=True, dropout=0.0, model_type=model_type)
        args.update(overrides)
        assert hc.tie_word_embeddings, "loader assumes tied embeddings"
        model = cls(GPTConfig(**args))
        hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).state_dict()
        D = args['n_embd'] // args['n_head']
        perm = torch.arange(D).view(2, D // 2).t().reshape(-1)   # [0, D/2, 1, D/2+1, ...]
        def interleave(w):   # rows = heads * D
            return w.view(-1, D, w.shape[-1])[:, perm].reshape(w.shape)
        sd = {'transformer.wte.weight': hf['model.embed_tokens.weight'], 'transformer.ln_f.weight': hf['model.norm.weight']}
        for i in range(args['n_layer']):
            p, q = f'model.layers.{i}.', f'transformer.h.{i}.'
            sd[q + 'ln_1.weight'] = hf[p + 'input_layernorm.weight']
            sd[q + 'ln_2.weight'] = hf[p + 'post_attention_layernorm.weight']
            sd[q + 'attn.q_proj.weight'] = interleave(hf[p + 'self_attn.q_proj.weight'])
            sd[q + 'attn.k_proj.weight'] = interleave(hf[p + 'self_attn.k_proj.weight'])
            sd[q + 'attn.v_proj.weight'] = hf[p + 'self_attn.v_proj.weight']
            sd[q + 'attn.o_proj.weight'] = hf[p + 'self_attn.o_proj.weight']
            for n in ('gate_proj', 'up_proj', 'down_proj'):
                sd[q + f'mlp.{n}.weight'] = hf[p + f'mlp.{n}.weight']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        missing = [m for m in missing if m not in ('base_freq', 'lm_head.weight')]   # buffer, and lm_head is tied to wte
        assert all(m.startswith('seg_') for m in missing), missing   # only the content-address map may be new
        print(f"from_hf_llama({name}): loaded {len(sd)} tensors; fresh: {missing}")
        return model

    @classmethod
    def from_hf_neox(cls, name, model_type='B', **overrides):
        """Build this model from a Hugging Face GPT-NeoX checkpoint (e.g. EleutherAI/pythia-160m): fused qkv split into q/k/v,
        the rotary dims of q/k permuted from rotate-half pairs (j, j+r/2) to interleaved pairs, non-rotary dims untouched."""
        from transformers import AutoConfig, AutoModelForCausalLM
        hc = AutoConfig.from_pretrained(name)
        rp = getattr(hc, 'rope_parameters', None) or {}
        base = float(rp.get('rope_theta', getattr(hc, 'rotary_emb_base', 10000.0)))
        pct = rp.get('partial_rotary_factor', getattr(hc, 'rotary_pct', 1.0))
        D = hc.hidden_size // hc.num_attention_heads
        rd = int(D * pct)
        args = dict(arch='neox', n_layer=hc.num_hidden_layers, n_head=hc.num_attention_heads, n_kv_head=hc.num_attention_heads,
                    n_embd=hc.hidden_size, vocab_size=hc.vocab_size, block_size=hc.max_position_embeddings, bias=True, attn_bias=True,
                    untied_head=not hc.tie_word_embeddings, rope_base=base, rope_per_head=True, rotary_dims=rd, dropout=0.0, model_type=model_type)
        args.update(overrides)
        assert hc.use_parallel_residual, "loader assumes the parallel-residual NeoX block"
        model = cls(GPTConfig(**args))
        hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).state_dict()
        H = args['n_head']
        perm = torch.arange(D)
        perm[:rd] = torch.arange(rd).view(2, rd // 2).t().reshape(-1)     # rotary dims: [0, r/2, 1, r/2+1, ...]; others unchanged
        def split_qkv(w):   # fused rows: per head [q(D), k(D), v(D)]
            w = w.view(H, 3, D, *w.shape[1:])
            q, k, v = w[:, 0], w[:, 1], w[:, 2]
            return q[:, perm].reshape(H * D, *w.shape[3:]), k[:, perm].reshape(H * D, *w.shape[3:]), v.reshape(H * D, *w.shape[3:])
        sd = {'transformer.wte.weight': hf['gpt_neox.embed_in.weight'],
              'transformer.ln_f.weight': hf['gpt_neox.final_layer_norm.weight'], 'transformer.ln_f.bias': hf['gpt_neox.final_layer_norm.bias'],
              'lm_head.weight': hf.get('embed_out.weight', hf.get('lm_head.weight'))}
        for i in range(args['n_layer']):
            p, q = f'gpt_neox.layers.{i}.', f'transformer.h.{i}.'
            for n in ('weight', 'bias'):
                sd[q + f'ln_1.{n}'] = hf[p + f'input_layernorm.{n}']; sd[q + f'ln_2.{n}'] = hf[p + f'post_attention_layernorm.{n}']
                qw, kw, vw = split_qkv(hf[p + f'attention.query_key_value.{n}'])
                sd[q + f'attn.q_proj.{n}'], sd[q + f'attn.k_proj.{n}'], sd[q + f'attn.v_proj.{n}'] = qw, kw, vw
                sd[q + f'attn.o_proj.{n}'] = hf[p + f'attention.dense.{n}']
                sd[q + f'mlp.c_fc.{n}'] = hf[p + f'mlp.dense_h_to_4h.{n}']; sd[q + f'mlp.c_proj.{n}'] = hf[p + f'mlp.dense_4h_to_h.{n}']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        missing = [m for m in missing if m != 'base_freq']
        assert all(m.startswith('seg_') or m.endswith('far_bias') for m in missing), missing
        print(f"from_hf_neox({name}): loaded {len(sd)} tensors; fresh: {missing}; rotary dims {rd} of {D}")
        return model

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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, new_param_lr_mult=1.0):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        is_new = lambda n: n.startswith('seg_') or n.endswith('far_bias') or '_far_proj' in n or n.endswith('_angles')   # adaptation parameters
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and not (new_param_lr_mult != 1.0 and is_new(n))]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and not (new_param_lr_mult != 1.0 and is_new(n))]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr_mult': 1.0},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr_mult': 1.0}
        ]
        if new_param_lr_mult != 1.0:   # the new parameters train at new_param_lr_mult x the pretrained weights' learning rate
            new_params = [p for n, p in param_dict.items() if is_new(n)]
            optim_groups.append({'params': new_params, 'weight_decay': 0.0, 'lr_mult': new_param_lr_mult})
            print(f"new-parameter group: {len(new_params)} tensors, {sum(p.numel() for p in new_params):,} params at {new_param_lr_mult}x lr")
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
