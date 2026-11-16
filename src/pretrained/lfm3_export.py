#!/usr/bin/env python3
"""Write LFM3-1.7B as weight blobs for the Branchy nodes (float32, the node's blob format):

  runs/pretrained/lfm3_stacks/L<first>-<end>.bin   one stack of consecutive layers (lfm3_blocks naming)
  runs/pretrained/lfm3_stacks/embed.bin            token table (also the output layer) followed by the final norm weight
  runs/pretrained/lfm3_stacks/index.json           layer kinds, sha256 and size of every file

  lfm3_export.py --step 4          # six stacks of four layers
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'node' / 'payload'))
import lfm3_blocks  # noqa: E402
import stacks  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'runs/pretrained/LFM3-1.7B-Base'
OUT = ROOT / 'runs/pretrained/lfm3_stacks'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--step', type=int, default=4)
    args = ap.parse_args()
    from safetensors.torch import load_file
    state = load_file(str(MODEL / 'model.safetensors'))
    kinds = json.loads((MODEL / 'config.json').read_text())['layer_types']
    OUT.mkdir(parents=True, exist_ok=True)
    index = dict(kinds=kinds, files={})

    def save(name, raw):
        (OUT / name).write_bytes(raw)
        index['files'][name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        print(f'{name}: {len(raw) / 2**20:.0f} MiB', flush=True)
    for first in range(0, len(kinds), args.step):
        end = min(len(kinds), first + args.step)
        stack = lfm3_blocks.build_stack(kinds[first:end])
        for i, layer in enumerate(stack):
            layer.load_state_dict(lfm3_blocks.layer_state(f'model.layers.{first + i}.', state, kinds[first + i]))
        save(f'L{first}-{end}.bin', stacks.state_to_blob(stack.state_dict()))
    embed = state['model.embed_tokens.weight'].float().contiguous().numpy().astype('<f4').tobytes()
    final = state['model.embedding_norm.weight'].float().contiguous().numpy().astype('<f4').tobytes()
    save('embed.bin', embed + final)
    index.update(vocab=int(state['model.embed_tokens.weight'].shape[0]), width=int(state['model.embed_tokens.weight'].shape[1]))
    (OUT / 'index.json').write_text(json.dumps(index, indent=1))


if __name__ == '__main__':
    main()
