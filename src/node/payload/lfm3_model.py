"""Coordinator side for LFM3-1.7B on Branchy nodes: sequential, layer split and branch layouts with one executor.

The 24 layers are cut into STACKS of consecutive layers (lfm3_blocks). A PLAN lists the stages one token goes through:

  ['seq', where, first, end]                       the stack of layers first..end-1 on `where` ('local' or a node name)
  ['par', [[where, first, end], ...]]              parallel branches: every stack gets the same input h0, and the stage
                                                   returns h0 + sum(branch output - h0)   (the conversion used in E6)

  sequential on one machine   [['seq', 'local', 0, 12], ['seq', 'local', 12, 24]]
  layer split (llama.cpp RPC) [['seq', 'local', 0, 12], ['seq', 'phone', 12, 24]]
  branched                    [['seq', 'local', 0, 4], ['par', [['local', 4, 8], ['phone', 8, 12], ...]], ['seq', 'local', 20, 24]]

Remote stacks are addressed by name (f'{tag}L{first}-{end}') and must be loaded on their node beforehand.
"""
from __future__ import annotations

import concurrent.futures
import uuid

import torch

import lfm3_blocks
import stacks
from cluster_model import perf
from peerclient import NodeError


def stack_name(tag, first, end):
    return f'{tag}L{first}-{end}'


def plan_stacks(plan):
    """[(where, first, end)] of every stack a plan uses."""
    out = []
    for stage in plan:
        out += [tuple(stage[1:])] if stage[0] == 'seq' else [tuple(s) for s in stage[1]]
    return out


class Lfm3:
    """The parts the coordinator holds itself: token table (also the output layer), final normalisation, local stacks."""

    def __init__(self, embed, final_w, kinds, context=4096):
        self.embed, self.final_w, self.kinds = embed, final_w, list(kinds)
        self.vocab, self.context, self.width = embed.shape[0], context, embed.shape[1]
        self.local = {}                 # (first, end) -> ModuleList
        self.label = f'LFM3-1.7B ({len(self.kinds)} layers, width {self.width})'

    def add_stack(self, first, end, raw=None, state=None, path=None, mmap=False):
        stack = lfm3_blocks.build_stack(self.kinds[first:end])
        if path is not None:                # from a blob file; with mmap the weights stay in the file (low-memory devices)
            stacks.load_blob_file(stack, path, mmap=mmap)
        elif raw is not None:
            stacks.load_blob_into(stack, raw)
        elif state is not None:
            stack.load_state_dict(state)
        self.local[(first, end)] = stack
        return stack

    def head(self, h):
        if stacks.BUDGET is not None:           # the output layer reads the whole token table for every token
            stacks.BUDGET.touch('embed', [self.embed])
        last = lfm3_blocks.rms(h[0, -1], self.final_w)
        return torch.mv(self.embed, last)


class Lfm3Executor:
    def __init__(self, model: Lfm3, plan, nodes: dict, wire='f4', tag='lfm'):
        self.model, self.plan, self.nodes, self.wire, self.tag = model, plan, nodes, wire, tag
        self.session = uuid.uuid4().hex[:12]
        self.cache, self.position = {}, 0
        merged = []                     # consecutive stacks on the same remote node become one chained call (one round trip)
        for stage in plan:
            if stage[0] == 'seq' and stage[1] != 'local' and merged and merged[-1][0] == 'chain' and merged[-1][1] == stage[1]:
                merged[-1][2].append((stage[2], stage[3]))
            elif stage[0] == 'seq' and stage[1] != 'local':
                merged.append(['chain', stage[1], [(stage[2], stage[3])]])
            else:
                merged.append(stage)
        self.stages = merged
        remote = sorted({where for where, _, _ in plan_stacks(plan) if where != 'local'})
        self.pools = {name: concurrent.futures.ThreadPoolExecutor(max_workers=1) for name in remote}
        for where, first, end in plan_stacks(plan):
            if where == 'local' and (first, end) not in model.local:
                raise ValueError(f'stack {first}-{end} is not loaded on the coordinator')

    def close(self):
        for name, pool in self.pools.items():
            pool.shutdown(wait=True)
            try:
                self.nodes[name].call('drop_session', session=self.session)
            except NodeError:
                pass

    def _local(self, first, end, h):
        y, self.cache[(first, end)] = stacks.run_stack(self.model.local[(first, end)], h, self.cache.get((first, end)))
        return y

    @torch.inference_mode()
    def step(self, ids, reset=False):
        m = self.model
        if reset:
            self.cache, self.position = {}, 0
        t0 = perf()
        h = m.embed[ids]
        self.position += ids.shape[1]
        local_s = wait_s = 0.0
        remote_stats = []
        for stage in self.stages:
            t = perf()
            if stage[0] == 'seq':
                h = self._local(stage[2], stage[3], h)
                local_s += perf() - t
                continue
            if stage[0] == 'chain':
                _, where, parts = stage
                names = [stack_name(self.tag, a, b) for a, b in parts]
                result, stats = self.nodes[where].forward(names, h, reset, self.session, self.wire, self.wire, chain=True)
                h = result[names[-1]]
                remote_stats.append(stats)
                wait_s += perf() - t
                continue
            groups = {}
            for where, first, end in stage[1]:
                groups.setdefault(where, []).append((first, end))
            futures = {where: self.pools[where].submit(self.nodes[where].forward, [stack_name(self.tag, a, b) for a, b in parts],
                                                       h, reset, self.session, self.wire, self.wire)
                       for where, parts in groups.items() if where != 'local'}
            fused = h.clone()
            for first, end in groups.get('local', []):
                fused += self._local(first, end, h) - h
            t_local = perf()
            local_s += t_local - t
            for where, future in futures.items():
                result, stats = future.result()
                remote_stats.append(stats)
                for first, end in groups[where]:
                    fused += result[stack_name(self.tag, first, end)] - h
            wait_s += perf() - t_local
            h = fused
        t1 = perf()
        logits = m.head(h)
        t2 = perf()
        return logits, dict(total_s=t2 - t0, shared_s=local_s, local_s=0.0, wait_s=wait_s, fuse_s=t2 - t1, remote=remote_stats)
