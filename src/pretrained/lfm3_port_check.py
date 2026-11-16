#!/usr/bin/env python3
"""Numerical check of the plain-PyTorch LFM3 layers (src/node/payload/lfm3_blocks.py) against the original model.

1. Whole sequence of 320 tokens (longer than the 256-token sliding window): hidden state after every layer and logits.
2. Cached decoding as the workers do it: 290 tokens in one call, then 30 single tokens; logits against (1).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'node' / 'payload'))
import lfm3_blocks  # noqa: E402
import stacks  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = str(ROOT / 'runs/pretrained/LFM3-1.7B-Base')


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32).eval()
    text = (ROOT / 'runs/rocm_300m_train/data/TinyStoriesV2-GPT4-valid.txt').read_text(errors='ignore')[:6000]
    ids = tok(text, return_tensors='pt').input_ids[:, :320]
    ref = hf(ids, output_hidden_states=True)
    kinds = list(hf.config.layer_types)
    state = hf.state_dict()
    ours = lfm3_blocks.build_stack(kinds)
    for i, layer in enumerate(ours):
        layer.load_state_dict(lfm3_blocks.layer_state(f'model.layers.{i}.', state, kinds[i]))
    embed, final = state['model.embed_tokens.weight'].float(), state['model.embedding_norm.weight'].float()

    def head(h):
        return lfm3_blocks.rms(h, final) @ embed.T
    h = embed[ids]
    worst = 0.0
    for i, layer in enumerate(ours):
        h, _ = layer(h, None)
        diff = float((h - ref.hidden_states[i + 1]).abs().max())
        scale = float(ref.hidden_states[i + 1].abs().max())
        worst = max(worst, diff / scale)
        print(f'layer {i:2d} {kinds[i]:17s} max abs diff {diff:.2e} (values up to {scale:.1f})', flush=True)
    full = head(h)
    d1 = float((full - ref.logits).abs().max())
    print(f'1. whole sequence: logits max abs diff {d1:.2e}, same top token {float((full.argmax(-1) == ref.logits.argmax(-1)).float().mean()) * 100:.1f}%')
    for mode in ('gemm', 'mv'):
        stacks.set_linear_mode(mode)
        h, cache = stacks.run_stack(ours, embed[ids[:, :290]], None)
        outs = [head(h)]
        for t in range(290, 320):
            h, cache = stacks.run_stack(ours, embed[ids[:, t:t + 1]], cache)
            outs.append(head(h))
        step = torch.cat(outs, dim=1)
        d2 = float((step - ref.logits).abs().max())
        print(f'2. cached decode ({mode}): logits max abs diff {d2:.2e}, same top token {float((step.argmax(-1) == ref.logits.argmax(-1)).float().mean()) * 100:.1f}%')
    print('PASS' if max(d1, d2) < 2e-2 else 'FAIL')


if __name__ == '__main__':
    main()
