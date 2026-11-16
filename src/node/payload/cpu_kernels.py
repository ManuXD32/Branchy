#!/usr/bin/env python3
"""Opt-in experimental CPU alternatives for the Branchy worker.

Overrides *only* F.scaled_dot_product_attention and/or F.gelu, which Branchy
CausalBlock calls. Do not use in general-purpose PyTorch applications.

The full-precision math paths are used for correctness A/B comparisons, not
claimed to be faster. No changes to model weights or checkpoint format.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

_NATIVE_SDPA = F.scaled_dot_product_attention
_NATIVE_GELU = F.gelu
_ACTIVE_MODE = 'native'


def math_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                   scale=None, enable_gqa=False):
    """Equivalent math implementation for Branchy's FP32, dropout=0 workloads.

    Operates on [batch, heads, query, head_dim]. Causal masking exactly follows
    F.sdpa for square initial prefill; cached chunk masking is supplied by
    the caller and is supported through attn_mask.
    """
    if q.device.type != 'cpu' or q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        return _NATIVE_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    if dropout_p != 0 or enable_gqa or (is_causal and attn_mask is not None):
        raise ValueError('Branchy math-attn supports only dropout=0, GQA off, and no mask with is_causal')
    score = torch.matmul(q.contiguous(), k.transpose(-2, -1).contiguous())
    score = score * (float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1]))
    if is_causal:
        # SDPA's standard upper-left causal alignment; for Branchy's prefill L=S.
        keep = torch.ones(score.shape[-2:], device=score.device, dtype=torch.bool).tril()
        score = score.masked_fill(~keep, float('-inf'))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            score = score.masked_fill(~attn_mask, float('-inf'))
        else:
            score = score + attn_mask
    weights = torch.softmax(score, dim=-1)
    return torch.matmul(weights, v.contiguous())


def math_gelu(x, approximate='none'):
    if x.device.type != 'cpu' or x.dtype != torch.float32:
        return _NATIVE_GELU(x, approximate=approximate)
    if approximate == 'none':
        return (0.5 * x) * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))
    # Branchy's actual layers always use approximate='none'.
    return _NATIVE_GELU(x, approximate=approximate)


def install_kernels(mode: str):
    global _ACTIVE_MODE
    modes = {'native', 'math-attn', 'math-gelu', 'math-both'}
    if mode not in modes:
        raise ValueError(f'Invalid CPU kernel mode {mode!r}; choose {sorted(modes)}')
    # A fresh process is required when changing the mode, matching how the
    # worker's fixed PyTorch/OpenMP thread count is managed.
    if _ACTIVE_MODE != 'native' and mode != _ACTIVE_MODE:
        raise RuntimeError('Do not change kernel modes in a running process')
    F.scaled_dot_product_attention = math_attention if mode in ('math-attn', 'math-both') else _NATIVE_SDPA
    F.gelu = math_gelu if mode in ('math-gelu', 'math-both') else _NATIVE_GELU
    _ACTIVE_MODE = mode
    return mode
