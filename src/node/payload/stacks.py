"""Generic transformer block stacks for Branchy nodes.

`Block` is numerically the same module as `branchy4_model.Block` and the legacy
`y_poc.CausalBlock` (same parameter names), so a stack can host any branch of
the 2-way or 4-way models, at any width/depth.

Also: pickle-free weight blobs (raw little-endian fp32 in sorted-key order) and
bit-reproducible synthetic weights/inputs built from integer arithmetic only,
so every device can construct identical untrained weights without an upload.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_HAS_RMS = hasattr(F, 'rms_norm')


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        if _HAS_RMS:
            return F.rms_norm(x, (x.shape[-1],), self.weight, eps=1e-5)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.weight


LINEAR_MODE = 'gemm'


def set_linear_mode(mode: str) -> str:
    """'gemm' = plain nn.Linear. 'mv' = single-token calls use a matrix-vector product (addmv).

    Same math, different BLAS routine: on OpenBLAS-backed torch builds the one-row
    gemm path is about 2x slower than gemv for cached decode (see docs/LESSONS.md).
    """
    global LINEAR_MODE
    if mode not in ('gemm', 'mv'):
        raise ValueError('linear mode must be gemm or mv')
    LINEAR_MODE = mode
    return mode


def lin(layer: nn.Linear, x):
    if LINEAR_MODE == 'mv' and x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == 1:
        return torch.addmv(layer.bias, layer.weight, x.reshape(-1)).reshape(1, 1, -1)
    return layer(x)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm_attn = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm_ffn = RMSNorm(width)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x, past=None):
        b, t, d = x.shape
        q, k, v = lin(self.qkv, self.norm_attn(x)).chunk(3, dim=-1)

        def rs(z):
            return z.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = rs(q), rs(k), rs(v)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        # F.<fn> is looked up at call time on purpose: cpu_kernels may patch it.
        if past is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.)
        elif t == 1:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.)
        else:
            prefix = past[0].shape[2]
            mask = torch.arange(prefix + t)[None, :] <= (prefix + torch.arange(t))[:, None]
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, dropout_p=0.)
        a = a.transpose(1, 2).contiguous().reshape(b, t, d)
        x = x + lin(self.proj, a)
        x = x + lin(self.fc2, F.gelu(lin(self.fc1, self.norm_ffn(x))))
        return x, (k, v)


def build_stack(width: int, heads: int, blocks: int, empty=False) -> nn.ModuleList:
    """empty=True: the weights get assigned afterwards (memory-mapped file), so the freshly allocated ones are not
    initialised. Untouched memory is never backed by RAM, which matters on a device with less RAM than the model."""
    if width % heads or not (1 <= blocks <= 64) or not (8 <= width <= 8192):
        raise ValueError('invalid stack geometry')
    if empty:
        saved = nn.Linear.reset_parameters
        nn.Linear.reset_parameters = lambda self: None
        try:
            stack = nn.ModuleList([Block(width, heads) for _ in range(blocks)])
        finally:
            nn.Linear.reset_parameters = saved
    else:
        stack = nn.ModuleList([Block(width, heads) for _ in range(blocks)])
    stack.eval()
    for p in stack.parameters():
        p.requires_grad_(False)
    return stack


class Budget:
    """A RAM budget for memory-mapped weights, enforced by the process itself (no root, works on Android).

    Emulates a device with too little memory on a real device: the weights of at most `megabytes` stay in RAM. Before a
    block runs it is marked as used; if the blocks in RAM then exceed the budget, the least recently used ones are
    handed back to the kernel (madvise MADV_PAGEOUT drops their clean file pages), so the next use reads them from the
    device's own storage again. With cyclic use (every token walks through all blocks) this is what the kernel does
    under memory pressure. Only blocks loaded with mmap can be given back."""
    PAGEOUT, PAGE = 21, 4096

    def __init__(self, megabytes):
        import collections
        import ctypes
        self.limit = int(megabytes * 2**20)
        self.units = collections.OrderedDict()     # key -> [ranges, bytes, in RAM?]; order = least recently used first
        self.in_ram = 0
        self.evictions = self.evicted_bytes = 0
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

    def touch(self, key, tensors):
        unit = self.units.get(key)
        if unit is None:
            ranges = []
            for t in tensors:
                start = -(-t.data_ptr() // self.PAGE) * self.PAGE          # whole pages inside the tensor
                end = (t.data_ptr() + t.numel() * t.element_size()) // self.PAGE * self.PAGE
                if end > start:
                    ranges.append((start, end - start))
            unit = self.units[key] = [ranges, sum(n for _, n in ranges), False]
        self.units.move_to_end(key)
        if not unit[2]:
            unit[2] = True
            self.in_ram += unit[1]
        for other_key, other in list(self.units.items()):
            if self.in_ram <= self.limit:
                break
            if other_key == key or not other[2]:
                continue
            self.release(other)

    def release(self, unit):
        import ctypes
        import os
        for start, size in unit[0]:
            if self.libc.madvise(start, size, self.PAGEOUT) != 0:
                raise OSError(ctypes.get_errno(), 'madvise(MADV_PAGEOUT) failed: ' + os.strerror(ctypes.get_errno()))
        unit[2] = False
        self.in_ram -= unit[1]
        self.evictions += 1
        self.evicted_bytes += unit[1]

    def status(self):
        return dict(limit_mb=self.limit / 2**20, in_ram_mb=self.in_ram / 2**20, units=len(self.units),
                    evictions=self.evictions, evicted_mb=self.evicted_bytes / 2**20)


BUDGET = None


def set_budget(megabytes):
    """megabytes None or 0: no budget. Returns the status of the previous budget (or None)."""
    global BUDGET
    old = BUDGET.status() if BUDGET else None
    BUDGET = Budget(float(megabytes)) if megabytes else None
    return old


def run_stack(stack, x, cache=None):
    new = []
    for i, block in enumerate(stack):
        if BUDGET is not None:
            BUDGET.touch(id(block), block.parameters())
        x, kv = block(x, None if cache is None else cache[i])
        new.append(kv)
    return x, new


def truncate_cache(cache, length: int):
    return [(k[:, :, :length].contiguous(), v[:, :, :length].contiguous()) for k, v in cache]


def stack_bytes(width: int, blocks: int) -> int:
    per_block = (3 * width * width + 3 * width) + (width * width + width) + (4 * width * width + 4 * width) \
        + (4 * width * width + width) + 2 * width
    return 4 * per_block * blocks


# ---------- raw weight blobs (no pickle) ----------

def state_to_blob(state: dict) -> bytes:
    parts = []
    for name in sorted(state):
        parts.append(state[name].detach().to('cpu', torch.float32).contiguous().numpy().astype('<f4', copy=False).tobytes())
    return b''.join(parts)


def load_blob_into(stack: nn.Module, raw) -> None:
    state = stack.state_dict()
    view = memoryview(raw)
    offset = 0
    new = {}
    for name in sorted(state):
        count = state[name].numel()
        size = 4 * count
        if offset + size > len(view):
            raise ValueError('weight blob too short for this stack geometry')
        arr = np.frombuffer(view[offset:offset + size], dtype='<f4').astype(np.float32).reshape(tuple(state[name].shape))
        new[name] = torch.from_numpy(arr.copy())
        offset += size
    if offset != len(view):
        raise ValueError('weight blob size does not match this stack geometry')
    stack.load_state_dict(new, strict=True)


def load_blob_file(stack: nn.Module, path, mmap=False) -> None:
    """Weights of a stack from a blob file. With mmap the tensors are views of the mapped file: nothing is copied, the
    kernel reads pages when they are used and may drop them again when memory is short (as llama.cpp does)."""
    if not mmap:
        return load_blob_into(stack, open(path, 'rb').read())
    state = stack.state_dict()
    total = sum(v.numel() for v in state.values())
    import os
    if os.path.getsize(path) != 4 * total:
        raise ValueError('weight blob size does not match this stack geometry')
    flat = torch.from_file(str(path), shared=False, size=total, dtype=torch.float32)
    offset, new = 0, {}
    for name in sorted(state):
        count = state[name].numel()
        new[name] = flat[offset:offset + count].view(state[name].shape)
        offset += count
    stack.load_state_dict(new, strict=True, assign=True)


# ---------- bit-reproducible synthetic numbers ----------

_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix(index: np.ndarray, offset: int) -> np.ndarray:
    """splitmix64 finalizer over uint64 counters; pure modular integer math."""
    with np.errstate(over='ignore'):
        z = index + np.uint64(offset)
        z = z * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def synthetic_array(count: int, key: str, std: float) -> np.ndarray:
    """`count` fp32 values, uniform with the requested std, identical on every platform."""
    offset = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
    out = np.empty(count, dtype=np.float32)
    step = 1 << 20
    half = std * math.sqrt(3.0)
    for start in range(0, count, step):
        idx = np.arange(start, min(count, start + step), dtype=np.uint64)
        top = (_splitmix(idx, offset) >> np.uint64(40)).astype(np.float64)   # 24 exact bits
        out[start:start + len(idx)] = ((top + 0.5) / 8388608.0 - 1.0) * half
    return out


def synthetic_tensor(shape, key: str, std: float = 1.0) -> torch.Tensor:
    count = int(np.prod(shape))
    return torch.from_numpy(synthetic_array(count, key, std).reshape(tuple(shape)))


def synthetic_state(stack: nn.Module, seed: str, depth: int = 11) -> dict:
    """Untrained weights with the same scale as branchy4_model.initialize()."""
    state = {}
    for name, tensor in stack.state_dict().items():
        if name.endswith('.bias'):
            state[name] = torch.zeros_like(tensor)
        elif '.norm_' in name or name.startswith('norm_'):
            state[name] = torch.ones_like(tensor)
        else:
            std = 0.02 / math.sqrt(2 * depth) if name.endswith(('proj.weight', 'fc2.weight')) else 0.02
            state[name] = synthetic_tensor(tensor.shape, f'{seed}/{name}', std)
    return state


def fill_synthetic(stack: nn.Module, seed: str, depth: int = 11) -> None:
    stack.load_state_dict(synthetic_state(stack, seed, depth), strict=True)


def selftest_run(width=256, heads=4, blocks=3, prefill=64, decode=8, seed='selftest'):
    """Seeded prefill + single-token cached decode on synthetic weights.

    Returns (meta, outputs[1, prefill+decode, width]). Every device must produce
    the same outputs up to float rounding (about 1e-5); larger differences mean a
    broken kernel/thread combination on that device.
    """
    import time
    stack = build_stack(width, heads, blocks)
    fill_synthetic(stack, seed, depth=1)   # depth=1: big residual updates, a sharper numeric test
    x = synthetic_tensor([1, prefill + decode, width], seed + '/x', 1.0)

    def cached_run():
        y, cache = run_stack(stack, x[:, :prefill], None)
        pieces = [y]
        for i in range(prefill, prefill + decode):
            y, cache = run_stack(stack, x[:, i:i + 1], cache)
            pieces.append(y)
        return torch.cat(pieces, dim=1)
    with torch.inference_mode():
        t = time.perf_counter()
        first = cached_run()
        elapsed = time.perf_counter() - t
        second = cached_run()
        full, _ = run_stack(stack, x, None)
    raw = first.numpy().astype('<f4', copy=False).tobytes()
    meta = dict(shape=list(first.shape), sha=hashlib.sha256(raw).hexdigest(), finite=bool(torch.isfinite(first).all()),
                repeat_max_abs=float((first - second).abs().max()),
                cache_vs_full_max_abs=float((first - full).abs().max()),
                out_abs_mean=float(first.abs().mean()), run_s=elapsed)
    return meta, first
