"""Coordinator-side model for Branchy experiments: trunk + N branches + fusion, or plain sequential.

Built from `stacks` so the coordinator and the nodes run the same block code.
Supports synthetic (untrained, bit-reproducible) weights and the trained
checkpoints of the legacy 2-way Y model, the sequential baseline and Branchy4.
"""
from __future__ import annotations

import statistics
import time

import torch
from torch.nn import functional as F

import stacks

perf = time.perf_counter


class Model:
    def __init__(self, width, heads, vocab, context, tok, pos, final_w, shared, branches=(), merge_w=None, merge_b=None,
                 fused=None, label='model', branch_seeds=None, depth=11, inproj=None, branch_width=None, branch_heads=None):
        self.width, self.heads, self.vocab, self.context = width, heads, vocab, context
        self.tok, self.pos, self.final_w = tok, pos, final_w
        self.shared, self.branches, self.fused = shared, list(branches), fused
        self.merge_w, self.merge_b = merge_w, merge_b
        self.label = label
        self.branch_seeds = branch_seeds      # set for synthetic models: nodes can rebuild the weights themselves
        self.depth = depth
        self.inproj = inproj                  # per-branch (weight, bias) when the branches are narrower than the trunk
        self.delay = 0                        # 1: branches other than the first `local_branches` are fused one token late
        self.local_branches = 1
        self.branch_width, self.branch_heads = branch_width or width, branch_heads or heads

    # ------------------------------------------------------------ constructors
    @classmethod
    def synthetic(cls, n_branches=4, width=880, heads=10, shared=2, branch=7, fuse=2, vocab=8192, context=1024, tag='syn'):
        depth = shared + branch + fuse

        def stack(n, seed):
            s = stacks.build_stack(width, heads, n)
            stacks.fill_synthetic(s, seed, depth)
            return s
        seeds = [f'{tag}/branch{i}' for i in range(n_branches)]
        return cls(width, heads, vocab, context,
                   stacks.synthetic_tensor([vocab, width], f'{tag}/tok', 0.02),
                   stacks.synthetic_tensor([context, width], f'{tag}/pos', 0.02), torch.ones(width),
                   stack(shared, f'{tag}/shared'), [stack(branch, s) for s in seeds],
                   stacks.synthetic_tensor([width, n_branches * width], f'{tag}/merge', 0.02), torch.zeros(width),
                   stack(fuse, f'{tag}/fused'), label=f'synthetic {n_branches}-way w{width} {shared}+{branch}x{n_branches}+{fuse}',
                   branch_seeds=seeds, depth=depth)

    @classmethod
    def synthetic_sequential(cls, width=1152, heads=12, blocks=19, vocab=8192, context=1024, tag='seq'):
        s = stacks.build_stack(width, heads, blocks)
        stacks.fill_synthetic(s, f'{tag}/blocks', blocks)
        return cls(width, heads, vocab, context, stacks.synthetic_tensor([vocab, width], f'{tag}/tok', 0.02),
                   stacks.synthetic_tensor([context, width], f'{tag}/pos', 0.02), torch.ones(width), s,
                   label=f'synthetic sequential w{width} x{blocks}')

    @classmethod
    def from_checkpoint(cls, path, mmap=False):
        """Accepts legacy sequential / 2-way Y checkpoints, Branchy4 inference files and BranchyN (E5) inference files."""
        # mmap: the weights stay in the file and are paged in when used (float32 inference files only)
        ck = torch.load(path, map_location='cpu', weights_only=True, mmap=True) if mmap else torch.load(path, map_location='cpu', weights_only=True)
        cfg, state = ck['config'], ck['model']
        width, heads = cfg['width'], cfg['heads']

        def stack(prefix, w=width, n=heads):
            sub = {k[len(prefix):]: v.float() for k, v in state.items() if k.startswith(prefix)}
            count = len({k.split('.')[0] for k in sub})
            s = stacks.build_stack(w, n, count, empty=bool(mmap))
            s.load_state_dict(sub, strict=True, assign=bool(mmap))
            return s
        common = dict(width=width, heads=heads, vocab=cfg['vocab'], context=cfg['context'], tok=state['tok.weight'].float(),
                      pos=state['pos.weight'].float(), final_w=state['final.weight'].float())
        if cfg.get('kind') == 'sequential':
            return cls(shared=stack('blocks.'), label=f"sequential w{width} x{cfg['seq_blocks']} ({path})", **common)
        if 'n_branches' in cfg:                                   # BranchyN: N equal branches, flat fusion (E2 / E5)
            if cfg.get('fusion', 'flat') != 'flat' or cfg.get('delay', 0) not in (0, 1):
                raise ValueError('only flat fusion with delay 0 or 1 is supported for inference')
            n, bw, bh = cfg['n_branches'], cfg['branch_width'], cfg['branch_heads']
            inproj = None
            if 'inproj.0.weight' in state:
                inproj = [(state[f'inproj.{i}.weight'].float(), state[f'inproj.{i}.bias'].float()) for i in range(n)]
            model = cls(shared=stack('shared.'), branches=[stack(f'branches.{i}.', bw, bh) for i in range(n)],
                        merge_w=state['merge.weight'].float(), merge_b=state['merge.bias'].float(), fused=stack('fused.'),
                        label=f'{n}-way w{width}, branches w{bw}' + (', delay 1' if cfg.get('delay') else '') + f' ({path})',
                        inproj=inproj, branch_width=bw, branch_heads=bh, **common)
            model.delay, model.local_branches = int(cfg.get('delay', 0)), int(cfg.get('local_branches', 1))
            return model
        if 'branch_a.0.qkv.weight' in state:
            branches = [stack('branch_a.'), stack('branch_b.')]
        else:
            branches = [stack(f'branches.{i}.') for i in range(4)]
        return cls(shared=stack('shared.'), branches=branches, merge_w=state['merge.weight'].float(),
                   merge_b=state['merge.bias'].float(), fused=stack('fused.'),
                   label=f'{len(branches)}-way w{width} ({path})', **common)

    # ------------------------------------------------------------ pieces of a forward step
    @property
    def n_branches(self):
        return len(self.branches)

    def parameters(self):
        total = self.tok.numel() + self.pos.numel() + self.final_w.numel()
        for s in [self.shared, self.fused, *self.branches]:
            if s is not None:
                total += sum(p.numel() for p in s.parameters())
        if self.merge_w is not None:
            total += self.merge_w.numel() + self.merge_b.numel()
        for w, b in self.inproj or []:
            total += w.numel() + b.numel()
        return total

    def trunk(self, ids, cache):
        start = 0 if cache is None else cache[0][0].shape[2]
        if start + ids.shape[1] > self.context:
            raise ValueError('sequence exceeds context')
        h = self.tok[ids] + self.pos[start:start + ids.shape[1]]
        return stacks.run_stack(self.shared, h, cache)

    def branch_input(self, i, h):
        """What branch i reads: the trunk output, through the branch's input projection if it has one."""
        if self.inproj is None:
            return h
        w, b = self.inproj[i]
        if stacks.LINEAR_MODE == 'mv' and h.shape[1] == 1:
            return torch.addmv(b, w, h.reshape(-1)).reshape(1, 1, -1)
        return F.linear(h, w, b)

    def head(self, outputs, trunk_out, cache):
        """Fusion blocks + last-position logits. `outputs` is the list of branch outputs (empty for sequential).
        A missing branch (None) counts as zeros, as in training with branch dropout."""
        if self.branches:
            ref = next(o for o in outputs if o is not None)
            outputs = [torch.zeros_like(ref) if o is None else o for o in outputs]
            cat = torch.cat(outputs, dim=-1)
            if stacks.LINEAR_MODE == 'mv' and cat.shape[1] == 1:
                h = torch.addmv(self.merge_b, self.merge_w, cat.reshape(-1)).reshape(1, 1, -1)
            else:
                h = F.linear(cat, self.merge_w, self.merge_b)
            h, new = stacks.run_stack(self.fused, h, cache)
        else:
            h, new = trunk_out, None
        last = h[0, -1]
        last = last * torch.rsqrt(last.pow(2).mean() + 1e-5) * self.final_w
        return torch.mv(self.tok, last), new


class Lag:
    """One-token delay of the non-local branches, as in training (branchyn_model, delay=1): at position t the fusion
    sees branch i's output of position t-1 (zeros at position 0). `shift` takes the fresh outputs of this step and
    returns what the fusion must use; the last fresh position is kept for the next step."""

    def __init__(self, model):
        self.on = bool(model.delay)
        self.first = model.local_branches
        self.held = {}

    def delayed(self, i):
        return self.on and i >= self.first

    def shift(self, i, fresh, reset):
        if not self.delayed(i) or fresh is None:
            return fresh
        previous = torch.zeros_like(fresh[:, :1]) if reset or i not in self.held else self.held[i]
        self.held[i] = fresh[:, -1:].clone()
        return torch.cat([previous, fresh[:, :-1]], dim=1)


class LocalExecutor:
    """Everything on this machine. Also the reference for parity checks."""

    def __init__(self, model: Model):
        self.model = model
        self.cache = None
        self.lag = Lag(model)

    def close(self):
        pass

    @torch.inference_mode()
    def step(self, ids, reset=False):
        m = self.model
        cache = None if reset else self.cache
        t0 = perf()
        h, shared = m.trunk(ids, None if cache is None else cache['shared'])
        t1 = perf()
        outs, branch_caches = [], []
        for i, b in enumerate(m.branches):
            y, c = stacks.run_stack(b, m.branch_input(i, h), None if cache is None else cache['branches'][i])
            outs.append(self.lag.shift(i, y, cache is None))
            branch_caches.append(c)
        t2 = perf()
        logits, fused = m.head(outs, h, None if cache is None else cache['fused'])
        t3 = perf()
        self.cache = dict(shared=shared, branches=branch_caches, fused=fused)
        return logits, dict(total_s=t3 - t0, shared_s=t1 - t0, local_s=t2 - t1, wait_s=0.0, fuse_s=t3 - t2, remote=[])


def summarize(values):
    ordered = sorted(values)
    return dict(p50=statistics.median(ordered), p95=ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
                min=ordered[0], max=ordered[-1], mean=statistics.fmean(ordered))


@torch.inference_mode()
def run_trial(executor, prompt, decode, forced=None):
    """Prefill + `decode` cached steps. With `forced` (token list) the steps are teacher-forced.

    Returns timings, the greedy tokens and the last-position logits of every step.
    """
    logits, stat = executor.step(prompt, reset=True)
    prefill = stat
    tokens, all_logits, steps = [], [logits.clone()], []
    for i in range(decode):
        greedy = int(logits.argmax())
        tokens.append(greedy)
        nxt = forced[i] if forced is not None else greedy
        logits, stat = executor.step(torch.tensor([[nxt]]), reset=False)
        steps.append(stat)
        all_logits.append(logits.clone())
    return dict(prefill=prefill, steps=steps, tokens=tokens, logits=torch.stack(all_logits))


def time_local(model: Model, prompts, decodes, runs=3, seed='prompt'):
    """Single-device timing table for a model held entirely on this machine."""
    rows = []
    exe = LocalExecutor(model)
    for p in prompts:
        ids = (stacks.synthetic_tensor([1, p], f'{seed}/{p}', 1.0).abs() * 1e6).long() % model.vocab
        for d in decodes:
            if p + d > model.context:
                continue
            run_trial(exe, ids, min(2, d))
            trials = [run_trial(exe, ids, d) for _ in range(runs)]
            step_ms = [s['total_s'] * 1000 for t in trials for s in t['steps']]
            prefill_s = statistics.median(t['prefill']['total_s'] for t in trials)
            decode_s = statistics.median(sum(s['total_s'] for s in t['steps']) for t in trials)
            rows.append(dict(prompt=p, decode=d, prefill_s=prefill_s, prefill_tps=p / prefill_s, decode_tps=d / decode_s,
                             total_s=prefill_s + decode_s, step_ms=summarize(step_ms)))
    return rows
