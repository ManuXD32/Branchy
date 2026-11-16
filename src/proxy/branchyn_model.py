"""BranchyN: research model family for the small-proxy architecture experiments (E2).

One trunk, N equal branches, a fusion stage, final blocks. Options under test:
  fusion='flat'     one linear merge over all branch outputs (the Branchy4 design, generalized to N)
  fusion='grouped'  fixed groups of branches, each with its own intermediate fusion block(s), then a final merge
  fusion='shared'   one shared local-fusion module applied to any set of branches; trained on random partitions
  delay=d           branches other than the first `local_branches` contribute with a lag of d positions
  drop_p=p          during training each non-local branch is dropped with probability p (per sequence)
kind='sequential' is the matched plain transformer.

Training-only code (no KV cache). Inference parity code lives in src/node.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    kind: str = 'branched'            # 'branched' | 'sequential'
    vocab: int = 8192
    context: int = 512
    width: int = 384
    heads: int = 6
    seq_blocks: int = 28
    shared_blocks: int = 2
    fuse_blocks: int = 2
    n_branches: int = 4
    branch_blocks: int = 6
    branch_width: int = 384
    branch_heads: int = 6
    fusion: str = 'flat'              # 'flat' | 'grouped' | 'shared'
    groups: int = 4                   # fixed groups for fusion='grouped'
    local_blocks: int = 1             # intermediate fusion blocks (per group, or shared)
    local_width: int = 384
    local_branches: int = 1           # branches that are never delayed or dropped (they live on the coordinator)
    delay: int = 0
    drop_p: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, eps=1e-5)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm_attn = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm_ffn = RMSNorm(width)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm_attn(x)).chunk(3, dim=-1)

        def rs(z):
            return z.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(rs(q), rs(k), rs(v), is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(F.gelu(self.fc1(self.norm_ffn(x))))


def stack(width, heads, n):
    return nn.ModuleList([Block(width, heads) for _ in range(n)])


def run(blocks, x):
    for block in blocks:
        x = block(x)
    return x


def fixed_groups(n, g):
    size = n // g
    return [list(range(i * size, (i + 1) * size if i < g - 1 else n)) for i in range(g)]


def random_partition(n, rng: random.Random):
    """Random split of range(n) into 1..n non-empty groups (a random cluster)."""
    order = list(range(n))
    rng.shuffle(order)
    g = rng.choice([1, 2, 3, 4, 6, 8, n]) if n >= 8 else rng.randint(1, n)
    g = max(1, min(g, n))
    cuts = sorted(rng.sample(range(1, n), g - 1)) if g > 1 else []
    return [order[a:b] for a, b in zip([0] + cuts, cuts + [n])]


class BranchyN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = c = cfg
        self.tok = nn.Embedding(c.vocab, c.width)
        self.pos = nn.Embedding(c.context, c.width)
        self.final = RMSNorm(c.width)
        self.checkpoint_branches = False   # training at 300M: recompute each branch in the backward pass (saves VRAM)
        if c.kind == 'sequential':
            self.blocks = stack(c.width, c.heads, c.seq_blocks)
            return
        bw = c.branch_width
        self.shared = stack(c.width, c.heads, c.shared_blocks)
        self.inproj = nn.ModuleList([nn.Linear(c.width, bw) if bw != c.width else nn.Identity() for _ in range(c.n_branches)])
        self.branches = nn.ModuleList([stack(bw, c.branch_heads, c.branch_blocks) for _ in range(c.n_branches)])
        if c.fusion == 'flat':
            self.merge = nn.Linear(c.n_branches * bw, c.width)
        elif c.fusion == 'grouped':
            self.group_ids = fixed_groups(c.n_branches, c.groups)
            self.group_merge = nn.ModuleList([nn.Linear(len(g) * bw, c.local_width) for g in self.group_ids])
            self.group_blocks = nn.ModuleList([stack(c.local_width, c.heads, c.local_blocks) for _ in self.group_ids])
            self.merge = nn.Linear(c.groups * c.local_width, c.width)
        elif c.fusion == 'shared':
            self.outproj = nn.ModuleList([nn.Linear(bw, c.local_width) for _ in range(c.n_branches)])
            self.local_in = RMSNorm(c.local_width)
            self.local_blocks = stack(c.local_width, c.heads, c.local_blocks)
            self.local_out = RMSNorm(c.local_width)
            self.merge = nn.Linear(c.local_width, c.width)
        else:
            raise ValueError(c.fusion)
        self.fused = stack(c.width, c.heads, c.fuse_blocks)

    # ------------------------------------------------------------------
    def branch_outputs(self, h, keep=None):
        """keep: optional bool tensor [B, N]; dropped branches contribute zeros."""
        c = self.cfg
        outs = []
        for i, blocks in enumerate(self.branches):
            if keep is not None and not bool(keep[:, i].any()):          # dropped for the whole batch: do not compute it
                outs.append(h.new_zeros(h.shape[0], h.shape[1], c.branch_width))
                continue
            if self.checkpoint_branches and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                y = checkpoint(lambda t, i=i, blocks=blocks: run(blocks, self.inproj[i](t)), h, use_reentrant=False)
            else:
                y = run(blocks, self.inproj[i](h))
            if c.delay and i >= c.local_branches:
                y = F.pad(y, (0, 0, c.delay, 0))[:, :y.shape[1]]      # position t sees the branch state of t - delay
            if keep is not None:
                y = y * keep[:, i, None, None].to(y.dtype)
            outs.append(y)
        return outs

    def fuse(self, outs, partition=None):
        c = self.cfg
        if c.fusion == 'flat':
            return self.merge(torch.cat(outs, dim=-1))
        if c.fusion == 'grouped':
            mids = [run(blocks, merge(torch.cat([outs[i] for i in ids], dim=-1)))
                    for ids, merge, blocks in zip(self.group_ids, self.group_merge, self.group_blocks)]
            return self.merge(torch.cat(mids, dim=-1))
        partition = partition or [list(range(c.n_branches))]
        b = outs[0].shape[0]
        mixed = torch.cat([self.local_in(sum(self.outproj[i](outs[i]) for i in ids)) for ids in partition], dim=0)
        local = run(self.local_blocks, mixed)                          # every group goes through the same module
        total = sum(local[g * b:(g + 1) * b] for g in range(len(partition)))
        return self.merge(self.local_out(total))

    def forward(self, idx, partition=None, keep=None):
        c = self.cfg
        h = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        if c.kind == 'sequential':
            h = run(self.blocks, h)
        else:
            h = run(self.shared, h)
            if keep is None and self.training and c.drop_p > 0:
                keep = torch.rand(idx.shape[0], c.n_branches, device=idx.device) >= c.drop_p
                keep[:, :c.local_branches] = True
            h = run(self.fused, self.fuse(self.branch_outputs(h, keep), partition))
        return F.linear(self.final(h), self.tok.weight)


def initialize(model: BranchyN, seed: int):
    torch.manual_seed(seed)
    c = model.cfg
    depth = c.seq_blocks if c.kind == 'sequential' else c.shared_blocks + c.branch_blocks + c.fuse_blocks + \
        (c.local_blocks if c.fusion != 'flat' else 0)
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.endswith(('.proj', '.fc2')):
            nn.init.normal_(m.weight, 0.0, 0.02 / math.sqrt(2 * depth))
    return model


def count(model):
    return sum(p.numel() for p in model.parameters())


def pick_heads(width):
    heads = max(2, round(width / 64))
    while width % heads:
        heads -= 1
    return heads


def match_branch_width(cfg: Config, target_params: int):
    """Largest branch width (multiple of 8) whose total parameter count does not exceed the target."""
    best = None
    for w in range(64, cfg.width + 1, 8):
        trial = Config(**{**asdict(cfg), 'branch_width': w, 'branch_heads': pick_heads(w)})
        n = count(BranchyN(trial))
        if n <= target_params:
            best = (trial, n)
    if best is None:
        raise ValueError('target too small')
    return best


def unit_bytes(cfg: Config):
    """fp32 bytes one branch unit reads per token (what a device must hold and read for one branch)."""
    w = cfg.branch_width
    per_block = 12 * w * w + 13 * w
    extra = (cfg.width * w + w) if w != cfg.width else 0
    return 4 * (cfg.branch_blocks * per_block + extra)
