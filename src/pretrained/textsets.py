"""Text of different kinds for the healing experiments (E8): each kind gives disjoint evaluation and healing text.

kind        evaluation text                         healing text
wikitext2   WikiText-2 test (as in E6/E7)           WikiText-103 train          (healing name: wiki)
tinystories TinyStories valid (as in E6/E7)         TinyStories train           (healing name: stories)
web         first documents of an OpenWebText shard  the rest of the shard
news        first CNN/DailyMail test articles        the rest
dialogue    first DialogSum train dialogues          the rest
code        Python instruction/solution pairs        the rest (not used for healing in E8: stays unseen)
science     PubMed articles                          the rest (not used for healing in E8: stays unseen)
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]      # the repository

T = (ROOT_DIR / 'runs/pretrained/text')
TINY = (ROOT_DIR / 'runs/rocm_300m_train/data')
ALIASES = {'wiki': 'wikitext2', 'stories': 'tinystories'}
EVAL_CHARS = 600_000          # reserved for evaluation at the start of each single-file kind


def _column(path, column):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=[column]).column(column).to_pylist()


def _documents(kind):
    if kind == 'web':
        return _column(T / 'web/plain_text/train-00000-of-00080.parquet', 'text')
    if kind == 'news':
        return _column(T / 'news/1.0.0/test-00000-of-00001.parquet', 'article')
    if kind == 'science':
        return _column(T / 'science/document/test-00000-of-00001.parquet', 'article')
    if kind == 'code':
        import pyarrow.parquet as pq
        t = pq.read_table(T / 'code/data/train-00000-of-00001-8b6e212f3e1ece96.parquet', columns=['instruction', 'output'])
        return [f'# {a}\n{b}' for a, b in zip(t.column('instruction').to_pylist(), t.column('output').to_pylist())]
    if kind == 'dialogue':
        with open(T / 'dialogue/train.csv', newline='') as f:
            return [row['dialogue'] for row in csv.DictReader(f)]
    raise ValueError(kind)


def _take(docs, chars, sep='\n\n'):
    out, used = [], 0
    for i, d in enumerate(docs):
        if used >= chars:
            return sep.join(out), i
        out.append(d)
        used += len(d)
    return sep.join(out), len(docs)


def eval_text(kind, chars):
    kind = ALIASES.get(kind, kind)
    if kind == 'tinystories':
        return (TINY / 'TinyStoriesV2-GPT4-valid.txt').read_text(errors='ignore')[:chars]
    if kind == 'wikitext2':
        return ''.join(_column(str(ROOT_DIR / 'runs/pretrained/wikitext2/wikitext-2-raw-v1/test-00000-of-00001.parquet'), 'text'))[:chars]
    assert chars <= EVAL_CHARS, 'evaluation text would run into the healing text'
    return _take(_documents(kind), chars)[0][:chars]


def heal_text(kind, chars):
    kind = ALIASES.get(kind, kind)
    if kind == 'tinystories':
        with open(TINY / 'TinyStoriesV2-GPT4-train.txt', errors='ignore') as f:
            return f.read(chars)
    if kind == 'wikitext2':
        rows = _column(str(ROOT_DIR / 'runs/pretrained/wikitext103/wikitext-103-raw-v1/train-00000-of-00002.parquet'), 'text')
        return _take(rows, chars, sep='')[0]
    docs = _documents(kind)
    _, first = _take(docs, EVAL_CHARS)
    return _take(docs[first:], chars)[0]
