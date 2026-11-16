#!/usr/bin/env python3
"""Tokenise TinyStories (TinyStoriesV2-GPT4-train.txt) with the project's tokenizer, the same way as the inherited
96 MiB prefix (stories split at <|endoftext|>, stripped, EOS after each, uint16).

  prepare_tinystories_full.py --text FILE --out train_full.bin            tokenise FILE (downloads the full file if missing)
  prepare_tinystories_full.py --check                                     tokenise the local prefix and compare with train.bin

The validation file and the tokenizer stay the inherited ones, so losses remain comparable with E5 and Q1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from array import array
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / 'runs/rocm_300m_train/data'
URL = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt'


def iter_stories(path):
    parts = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('<|endoftext|>'):
                story = ''.join(parts).strip()
                if story:
                    yield story
                parts = []
            else:
                parts.append(line)          # a last unfinished story is not used, as in the inherited preparation


def encode(text, out, tokenizer_file):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_file))
    eos = tok.token_to_id('[EOS]')
    tokens = stories = 0
    t0 = time.time()
    batch = []
    with open(out, 'wb') as f:
        def flush():
            nonlocal tokens
            for enc in tok.encode_batch(batch, add_special_tokens=False):
                array('H', enc.ids + [eos]).tofile(f)
                tokens += len(enc.ids) + 1
            batch.clear()
        for story in iter_stories(text):
            batch.append(story)
            stories += 1
            if len(batch) == 1024:
                flush()
                if stories % 204800 == 0:
                    print(f'  {stories:,} stories, {tokens:,} tokens, {time.time() - t0:.0f}s', flush=True)
        if batch:
            flush()
    return dict(stories=stories, tokens=tokens)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--text', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--tokenizer', type=Path, default=DATA / 'tokenizer.json')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.check:
        out = Path(sys.argv[0]).resolve().parents[2] / 'runs' / 'prefix_check.bin'
        info = encode(DATA / 'TinyStoriesV2-GPT4-train.txt', out, args.tokenizer)
        same = out.read_bytes() == (DATA / 'train.bin').read_bytes()
        out.unlink()
        print(info, 'identical to the inherited train.bin' if same else 'DIFFERENT from the inherited train.bin')
        sys.exit(0 if same else 1)
    if not args.text.exists():
        print(f'downloading {URL}', flush=True)
        urllib.request.urlretrieve(URL, args.text)
    print(f'{args.text}: {args.text.stat().st_size / 2**20:.0f} MiB', flush=True)
    info = encode(args.text, args.out, args.tokenizer)
    args.out.with_suffix('.json').write_text(json.dumps(dict(source=URL, bytes=args.text.stat().st_size, **info), indent=1))
    print(info, flush=True)


if __name__ == '__main__':
    main()
