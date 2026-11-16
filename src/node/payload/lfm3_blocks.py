"""LFM3 decoder layers in plain PyTorch, so Branchy workers (which have no `transformers`) can run them.

A port for inference of the layer mathematics of LiquidAI's `modeling_lfm3.py` (LFM3-1.7B, LFM Open License v1.0; that
file is the reference, this one was written for this project and changed in form: batch 1, causal, one cache per
layer, no padding or packed documents). Checked numerically against the original with src/pretrained/lfm3_port_check.py.

Three layer kinds, all with sandwich normalisation and a gated MLP:
  conv               gated depthwise causal convolution over the last 7 positions
  full_attention     grouped-query attention with normalised queries and keys, a sigmoid gate per head, partial RoPE
  sliding_attention  the same over the last 256 positions, full RoPE

A layer is called as `x, cache = layer(x, cache)` like the Branchy blocks, so the node runs stacks of them unchanged.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

import stacks

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER, KERNEL, WINDOW, EPS = 2048, 16, 4, 128, 7680, 7, 256, 1e-5
ROPE = {'full_attention': (1_000_000.0, 0.5), 'sliding_attention': (10_000.0, 1.0)}     # theta, rotated share of the head


def rms(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight


def lin(weight, x):
    """Bias-free linear layer; single-token calls use a matrix-vector product in 'mv' mode (see stacks.set_linear_mode)."""
    if stacks.LINEAR_MODE == 'mv' and x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == 1:
        return torch.mv(weight, x.reshape(-1)).reshape(1, 1, -1)
    return F.linear(x, weight)


class Layer(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in ('conv', 'full_attention', 'sliding_attention'):
            raise ValueError(f'unknown LFM3 layer kind {kind!r}')
        self.kind = kind
        p = lambda *shape: nn.Parameter(torch.zeros(*shape), requires_grad=False)      # noqa: E731
        self.operator_norm, self.post_operator_norm, self.ffn_norm, self.post_ffn_norm = p(HIDDEN), p(HIDDEN), p(HIDDEN), p(HIDDEN)
        self.w1, self.w3, self.w2 = p(INTER, HIDDEN), p(INTER, HIDDEN), p(HIDDEN, INTER)
        if kind == 'conv':
            self.in_proj, self.out_proj, self.conv = p(3 * HIDDEN, HIDDEN), p(HIDDEN, HIDDEN), p(HIDDEN, KERNEL)
        else:
            self.q, self.k, self.v = p(HEADS * HEAD_DIM, HIDDEN), p(KV_HEADS * HEAD_DIM, HIDDEN), p(KV_HEADS * HEAD_DIM, HIDDEN)
            self.o, self.gate = p(HIDDEN, HEADS * HEAD_DIM), p(HEADS, HIDDEN)
            self.q_norm, self.k_norm = p(HEAD_DIM), p(HEAD_DIM)
            theta, share = ROPE[kind]
            angles = int(share * HEAD_DIM) // 2
            inv = 1.0 / (theta ** (torch.arange(0, 2 * angles, 2, dtype=torch.float) / HEAD_DIM))
            self.inv_freq = torch.cat((inv, torch.zeros(HEAD_DIM // 2 - angles)))        # plain attribute: not part of the weights

    # ---------------------------------------------------------------- operators
    def _conv(self, n, cache):
        b, c, x = lin(self.in_proj, n).transpose(1, 2).chunk(3, dim=1)                   # each [1, HIDDEN, T]
        h = b * x
        state = h.new_zeros(1, HIDDEN, KERNEL - 1) if cache is None else cache[0]
        full = torch.cat((state, h), dim=-1)
        y = (full.unfold(-1, KERNEL, 1) * self.conv[None, :, None, :]).sum(-1)           # causal depthwise convolution
        return lin(self.out_proj, (c * y).transpose(1, 2)), (full[:, :, -(KERNEL - 1):].contiguous(),)

    def _attention(self, n, cache):
        t = n.shape[1]
        start = 0 if cache is None else int(cache[2])
        q = rms(lin(self.q, n).view(1, t, HEADS, HEAD_DIM), self.q_norm).transpose(1, 2)
        k = rms(lin(self.k, n).view(1, t, KV_HEADS, HEAD_DIM), self.k_norm).transpose(1, 2)
        v = lin(self.v, n).view(1, t, KV_HEADS, HEAD_DIM).transpose(1, 2)
        freqs = torch.arange(start, start + t, dtype=torch.float)[:, None] * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        def rotate(z):
            half = z.shape[-1] // 2
            return z * cos + torch.cat((-z[..., half:], z[..., :half]), dim=-1) * sin
        q, k = rotate(q), rotate(k)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        sliding = self.kind == 'sliding_attention'
        if sliding and t == 1 and k.shape[2] > WINDOW:
            k, v = k[:, :, -WINDOW:], v[:, :, -WINDOW:]
        s = k.shape[2]
        groups = HEADS // KV_HEADS
        scores = torch.matmul(q.view(1, KV_HEADS, groups, t, HEAD_DIM), k.transpose(2, 3)[:, :, None]) * HEAD_DIM ** -0.5   # [1, 4, 4, T, S]
        if t > 1:
            key_pos = torch.arange(start + t - s, start + t)[None, :]
            query_pos = torch.arange(start, start + t)[:, None]
            allowed = key_pos <= query_pos
            if sliding:
                allowed = allowed & (key_pos > query_pos - WINDOW)
            scores = scores.masked_fill(~allowed, float('-inf'))
        out = torch.matmul(torch.softmax(scores, dim=-1), v[:, :, None])                  # [1, 4, 4, T, HEAD_DIM]
        out = out.reshape(1, HEADS, t, HEAD_DIM).transpose(1, 2)                           # [1, T, HEADS, HEAD_DIM]
        out = (out * torch.sigmoid(lin(self.gate, n)).unsqueeze(-1)).reshape(1, t, HEADS * HEAD_DIM)
        if sliding and k.shape[2] > WINDOW:
            k, v = k[:, :, -WINDOW:], v[:, :, -WINDOW:]
        return lin(self.o, out), (k.contiguous(), v.contiguous(), torch.tensor(start + t))

    def forward(self, x, cache=None):
        n = rms(x, self.operator_norm)
        y, new = self._conv(n, cache) if self.kind == 'conv' else self._attention(n, cache)
        x = x + rms(y, self.post_operator_norm)
        m = rms(x, self.ffn_norm)
        f = lin(self.w2, F.silu(lin(self.w1, m)) * lin(self.w3, m))
        return x + rms(f, self.post_ffn_norm), new


def build_stack(kinds) -> nn.ModuleList:
    """A stack of LFM3 layers of the given kinds, e.g. ['conv', 'full_attention']."""
    kinds = list(kinds)
    if not 1 <= len(kinds) <= 32:
        raise ValueError('invalid number of LFM3 layers')
    stack = nn.ModuleList([Layer(kind) for kind in kinds])
    stack.eval()
    return stack


def layer_state(prefix, state, kind):
    """Weights of one layer of the original model (state dict keys under `prefix`, e.g. 'model.layers.3.') in this
    module's naming."""
    g = lambda name: state[prefix + name].float()                                        # noqa: E731
    out = {'operator_norm': g('operator_norm.weight'), 'post_operator_norm': g('post_operator_norm.weight'),
           'ffn_norm': g('ffn_norm.weight'), 'post_ffn_norm': g('post_ffn_norm.weight'),
           'w1': g('feed_forward.w1.weight'), 'w3': g('feed_forward.w3.weight'), 'w2': g('feed_forward.w2.weight')}
    if kind == 'conv':
        out.update(in_proj=g('conv.in_proj.weight'), out_proj=g('conv.out_proj.weight'), conv=g('conv.conv.weight').squeeze(1))
    else:
        out.update(q=g('self_attn.q_proj.weight'), k=g('self_attn.k_proj.weight'), v=g('self_attn.v_proj.weight'),
                   o=g('self_attn.o_proj.weight'), gate=g('self_attn.gate_proj.weight'),
                   q_norm=g('self_attn.q_norm.weight'), k_norm=g('self_attn.k_norm.weight'))
    return out
