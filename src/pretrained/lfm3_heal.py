#!/usr/bin/env python3
"""E6b / E7: "healing" fine-tune of a trained sequential model after its middle layers were turned into parallel branches.

The original weights stay frozen. Small low-rank adapters (LoRA) are added to every linear layer of the branch
layers and of the layers after them, and trained so that the converted model reproduces the ORIGINAL model's
next-token distribution (KL distillation). Teacher and student are the same weights in memory: the teacher pass
runs the layers in sequence with the adapters switched off.

  lfm3_heal.py --region 8 16 --branch-layers 2 --steps 1000          (4 branches x 2 layers)
  lfm3_heal.py --region 8 16 --skip --steps 1000                     (control: the same layers deleted, then healed)

With --full (E7, for models small enough: Qwen3.5-0.8B) the branch layers themselves are retrained (float32 master
weights under float16 autocast); a frozen copy of the original layers serves the teacher pass, and only the layers
after the region get adapters.

Healing data: WikiText-103 train. Evaluation (same windows as the E6 sweep): WikiText-2 test (same kind of text,
held out) and TinyStories valid (a kind of text the healing never saw).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lfm3_parallel import MODEL_DIR, datasets, evaluate, load

ROOT_DIR = Path(__file__).resolve().parents[2]      # the repository

WIKI103 = (ROOT_DIR / 'runs/pretrained/wikitext103/wikitext-103-raw-v1/train-00000-of-00002.parquet')


class Switch:
    on = True


class LoRALinear(nn.Module):
    """base(x) + scale * B(A(x)); B starts at zero, so the adapter is a no-op until trained."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, switch: Switch):
        super().__init__()
        self.base, self.switch, self.scale = base, switch, alpha / rank
        bound = 1 / math.sqrt(base.in_features)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device).uniform_(-bound, bound))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('base'), name)

    def forward(self, x):
        y = self.base(x)
        if self.switch.on:
            y = y + self.scale * F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))
        return y


def add_adapters(model, first_layer, rank, alpha, switch):
    for p in model.parameters():
        p.requires_grad_(False)
    added = 0
    for wrapped in model.model.layers[first_layer:]:
        targets = [(parent, name, child) for parent in wrapped.layer.modules()
                   for name, child in parent.named_children() if isinstance(child, nn.Linear)]
        for parent, name, child in targets:
            setattr(parent, name, LoRALinear(child, rank, alpha, switch))
            added += 1
    params = [p for p in model.parameters() if p.requires_grad]
    return params, added


def training_windows(tok, count, context, seed, kinds=('wiki',)):
    """`count` healing windows, an equal share from each kind of text, shuffled together."""
    import textsets
    head = [] if tok.bos_token_id is None else [tok.bos_token_id]
    body = context - len(head)
    share = -(-count // len(kinds))
    rows = []
    for kind in kinds:
        ids = tok(textsets.heal_text(kind, share * context * 8), add_special_tokens=False)['input_ids']
        assert len(ids) // body >= share, f'{kind}: only {len(ids) // body} healing windows, {share} needed'
        rows += [head + ids[i * body:(i + 1) * body] for i in range(share)]
    out = torch.tensor(rows, dtype=torch.long)
    return out[torch.randperm(len(out), generator=torch.Generator().manual_seed(seed))][:count]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=Path, default=MODEL_DIR)
    ap.add_argument('--full', action='store_true', help='retrain the branch layers themselves (adapters only on later layers)')
    ap.add_argument('--full-lr', type=float, default=2e-5, help='learning rate of the fully retrained layers')
    ap.add_argument('--loss-chunk', type=int, default=None,
                    help='compute the distillation loss over this many positions at a time (keeps the vocabulary-sized '
                         'tensors small; default 64 with --full, off otherwise)')
    ap.add_argument('--heal-mix', default='wiki', help='comma-separated kinds of healing text (textsets.py), equal shares')
    ap.add_argument('--eval-sets', default='tinystories,wikitext2', help='comma-separated kinds of evaluation text')
    ap.add_argument('--region', type=int, nargs=2, default=[8, 16], help='first layer and one past the last layer')
    ap.add_argument('--branch-layers', type=int, default=2, help='layers per branch')
    ap.add_argument('--skip', action='store_true', help='control: delete the region instead of branching it')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--context', type=int, default=512)
    ap.add_argument('--rank', type=int, default=32)
    ap.add_argument('--alpha', type=float, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--hidden-weight', type=float, default=0.0,
                    help='weight of an extra loss: relative squared error between the hidden state after the region '
                         'in the converted model and in the original (0 = logit distillation only)')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--eval-windows', type=int, default=48)
    ap.add_argument('--quick-windows', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default=None)
    ap.add_argument('--name')
    ap.add_argument('--out', type=Path, default=(ROOT_DIR / 'docs/results/E6_lfm3_parallel/heal'))
    ap.add_argument('--ckpt', type=Path, default=(ROOT_DIR / 'runs/pretrained/heal'))
    args = ap.parse_args()
    start, end = args.region
    if args.skip:
        name = args.name or f'skip_{start}-{end - 1}'
    else:
        assert (end - start) % args.branch_layers == 0
        ranges = [(s, s + args.branch_layers) for s in range(start, end, args.branch_layers)]
        name = args.name or f'par_{start}-{end - 1}_{len(ranges)}x{args.branch_layers}'
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype or ('float16' if device.type == 'cuda' else 'float32'))
    torch.manual_seed(args.seed)
    t0 = time.time()
    tok, model, plan = load(device, dtype, args.model)
    switch = Switch()
    full = args.full and not args.skip
    lora_from = end if full else start
    lora_params, added = add_adapters(model, lora_from, args.rank, args.alpha, switch)
    full_params = []
    if full:
        for wrapped in model.model.layers[start:end]:
            wrapped.alt = copy.deepcopy(wrapped.layer)          # frozen original, used by the teacher pass
            wrapped.layer.float()                                # float32 master weights for the optimizer
            for p in wrapped.layer.parameters():
                p.requires_grad_(True)
                full_params.append(p)
    params = lora_params + full_params
    trainable = sum(p.numel() for p in params)
    use_autocast = full and device.type == 'cuda'
    print(f'{name} on {args.model.name}: {added} adapters on layers {lora_from}-{len(model.model.layers) - 1} (rank {args.rank})'
          + (f', layers {start}-{end - 1} fully retrained' if full else '')
          + f', {trainable / 1e6:.1f}M trainable parameters ({time.time() - t0:.0f}s)', flush=True)

    def forward(ids):
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_autocast):
            return model(input_ids=ids, use_cache=False).logits

    loss_chunk = args.loss_chunk if args.loss_chunk is not None else (64 if full else 0)
    assert not (loss_chunk and args.hidden_weight), 'the hidden-state loss is only implemented without --loss-chunk'

    def body(ids):
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_autocast):
            return model.model(input_ids=ids, use_cache=False).last_hidden_state

    def chunked_kl(h_student, h_teacher, scale):
        """KL(teacher || student) over all positions, computed chunk by chunk through the (frozen) output layer.
        Returns the value and d(scale * KL)/d(h_student)."""
        hs = h_student.detach().requires_grad_(True)
        positions = hs.shape[0] * hs.shape[1]
        value = 0.0
        for a in range(0, hs.shape[1], loss_chunk):
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_autocast):
                with torch.no_grad():
                    target = F.log_softmax(model.lm_head(h_teacher[:, a:a + loss_chunk]).float(), dim=-1)
                logits = model.lm_head(hs[:, a:a + loss_chunk])
            logp = F.log_softmax(logits.float(), dim=-1)
            kl = (target.exp() * (target - logp)).sum() / positions
            (kl * scale).backward()
            value += float(kl.detach())
            del target, logits, logp, kl
        return value, hs.grad

    def teacher():
        plan.clear()
        plan.teacher = True
        switch.on = False

    def student():
        if args.skip:
            plan.set_skip(start, end)
        else:
            plan.set_branches(ranges, 'sum')
        plan.teacher = False
        switch.on = True

    captured = {}
    model.model.layers[end - 1].register_forward_hook(lambda module, inputs, output: captured.__setitem__('h', output))
    data = datasets(tok, args.eval_windows, args.context, args.eval_sets.split(','))
    train = training_windows(tok, args.steps * args.batch * args.accum, args.context, args.seed, args.heal_mix.split(','))
    assert len(train) >= args.steps * args.batch * args.accum, f'only {len(train)} training windows'
    print(f'  healing data: {len(train)} windows x {args.context} tokens, kinds: {args.heal_mix}', flush=True)

    teacher()
    original, reference = {}, {}
    for n, w in data.items():
        original[n], reference[n] = evaluate(model, w, device, args.batch, autocast=use_autocast)
        print(f'  original          {n:12s} NLL {original[n]["nll"]:.4f}', flush=True)

    def full_eval(count=None):
        student()
        res = {n: evaluate(model, w[:count] if count else w, device, args.batch, reference[n], autocast=use_autocast)[0] for n, w in data.items()}
        return res

    def show(tag, res):
        print(f'  {tag:17s} ' + ' | '.join(f"{n}: NLL {m['nll']:.3f} (+{m['nll'] - original[n]['nll']:.3f}) "
                                          f"agree {m['top1_agreement'] * 100:.0f}%" for n, m in res.items()), flush=True)

    before = full_eval()
    show('before healing', before)
    groups = [dict(params=lora_params, base=args.lr)] + ([dict(params=full_params, base=args.full_lr)] if full_params else [])
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)
    scaler.scale(torch.zeros(1, device=device))            # initialise the scale (the chunked loss scales by hand)
    curve = []
    running = None
    cursor = 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    for step in range(1, args.steps + 1):
        factor = step / args.warmup if step <= args.warmup else \
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup)))
        for group in opt.param_groups:
            group['lr'] = group['base'] * factor
        opt.zero_grad(set_to_none=True)
        step_kl = step_hid = 0.0
        for _ in range(args.accum):
            ids = train[cursor:cursor + args.batch].to(device)
            cursor += args.batch
            if loss_chunk:
                with torch.no_grad():
                    teacher()
                    h_teacher = body(ids)
                student()
                h_student = body(ids)
                value, grad = chunked_kl(h_student, h_teacher, scaler.get_scale() / args.accum)
                h_student.backward(grad.to(h_student.dtype))
                step_kl += value / args.accum
                del h_teacher, h_student, grad
                continue
            with torch.no_grad():
                teacher()
                target = F.log_softmax(forward(ids).float(), dim=-1)
                target_hidden = captured['h'].float()
            student()
            logp = F.log_softmax(forward(ids).float(), dim=-1)
            kl = (target.exp() * (target - logp)).sum(-1).mean()
            loss = kl
            if args.hidden_weight:
                hid = (captured['h'].float() - target_hidden).pow(2).mean() / target_hidden.pow(2).mean()
                loss = kl + args.hidden_weight * hid
                step_hid += float(hid.detach()) / args.accum
            scaler.scale(loss / args.accum).backward()
            step_kl += float(kl.detach()) / args.accum
            del target, logp, kl, loss, target_hidden
        if not math.isfinite(step_kl):
            raise SystemExit(f'loss is not finite at step {step}')
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        running = step_kl if running is None else 0.95 * running + 0.05 * step_kl
        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            quick = full_eval(args.quick_windows)
            elapsed = time.time() - t1
            vram = torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else 0
            curve.append(dict(step=step, train_kl=running, elapsed_s=elapsed,
                              **{f'{n}_nll': m['nll'] for n, m in quick.items()}))
            print(f'  step {step}/{args.steps} train KL {running:.4f}' + (f' hidden {step_hid:.3f}' if args.hidden_weight else '') + ' | ' +
                  ' '.join(f"{n} +{m['nll'] - original[n]['nll']:.3f}" for n, m in quick.items()) +
                  f' (first {args.quick_windows} windows) | {step * args.batch * args.accum * args.context / elapsed:,.0f} tok/s | '
                  f'VRAM {vram:.0f} MiB | {elapsed / 60:.1f} min', flush=True)
    train_min = (time.time() - t1) / 60
    after = full_eval()
    show('after healing', after)
    result = dict(name=name, model=str(args.model), revision=(args.model / 'REVISION').read_text().strip(), dtype=str(dtype),
                  full=full, full_lr=args.full_lr if full else None, loss_chunk=loss_chunk, heal_mix=args.heal_mix.split(','),
                  region=[start, end], kind='skip' if args.skip else 'parallel',
                  branches=None if args.skip else len(ranges), layers_per_branch=None if args.skip else args.branch_layers,
                  rank=args.rank, alpha=args.alpha, hidden_weight=args.hidden_weight, adapters=added, trainable_parameters=trainable, steps=args.steps,
                  batch=args.batch, accum=args.accum, context=args.context, lr=args.lr, seed=args.seed,
                  train_tokens=args.steps * args.batch * args.accum * args.context, train_minutes=train_min,
                  peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else 0,
                  eval_windows=args.eval_windows, original=original, before=before, after=after, curve=curve)
    (out / 'result.json').write_text(json.dumps(result, indent=1))
    args.ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().half().cpu() for n, p in model.named_parameters() if p.requires_grad}, args.ckpt / f'{name}.pt')
    print(f'  saved {out / "result.json"} ({train_min:.1f} min training, {(time.time() - t0) / 60:.1f} min total)', flush=True)


if __name__ == '__main__':
    main()
