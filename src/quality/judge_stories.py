#!/usr/bin/env python3
"""Blind pairwise story comparison by a local judge model (llama-swap on this server).

  judge_stories.py --dir docs/results/Q1 --reference seq --models flat8 seq_s2 --count 100

For each story beginning the judge sees the two continuations (reference model and candidate) as Story A and Story B,
once in each order, and answers which is better (grammar, fits the beginning, stays consistent) or a tie.
Reported per candidate: wins, ties, losses over all judgements, the win share among decided ones with a 95% interval,
and how often the judge picked position A (its position bias). Comparing two seeds of the same model gives the noise floor.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.request
from pathlib import Path

PROMPT = """You are grading two short children's stories written by two different students. Both continue the same beginning.

Beginning:
{beginning}

Story A continues:
{a}

Story B continues:
{b}

Which continuation is better? Judge three things: correct grammar and complete sentences, whether it follows the beginning,
and whether it stays consistent (same characters, events that make sense). Both are cut off at the end; ignore that.
Answer with one short sentence of reasoning and then a last line that is exactly one of:
WINNER: A
WINNER: B
WINNER: TIE"""


def ask(url, model, prompt, retries=3):
    body = json.dumps(dict(model=model, temperature=0, max_tokens=300, messages=[dict(role='user', content=prompt)],
                           chat_template_kwargs=dict(enable_thinking=False))).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, body, {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=300) as r:
                text = json.loads(r.read())['choices'][0]['message'].get('content') or ''
            found = re.findall(r'WINNER:\s*(A|B|TIE)', text.upper())
            if found:
                return found[-1], text
        except Exception as exc:  # noqa: BLE001
            text = f'error: {exc}'
            time.sleep(5)
    return 'INVALID', text


def wait_until_free(url, model):
    """Do not push another model out of llama-swap: wait while it has a model other than the judge loaded."""
    base = url.split('/v1/')[0]
    waited = False
    while True:
        try:
            with urllib.request.urlopen(base + '/running', timeout=5) as r:
                running = json.loads(r.read()).get('running', [])
        except Exception:  # noqa: BLE001
            running = []
        if not [r for r in running if r.get('model') != model]:
            return waited
        waited = True
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', type=Path, required=True, help='folder with stories_<model>.jsonl from quality_bench.py')
    ap.add_argument('--reference', required=True)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--judge', default='Qwen3.8-9B-MTP_65k')
    ap.add_argument('--url', default='http://127.0.0.1:8080/v1/chat/completions')
    args = ap.parse_args()
    read = lambda n: [json.loads(l) for l in open(args.dir / f'stories_{n}.jsonl')][:args.count]     # noqa: E731
    ref = read(args.reference)
    out_path = args.dir / 'judge.json'
    report = json.loads(out_path.read_text()) if out_path.exists() else dict(judge=args.judge, reference=args.reference, pairs={})
    for name in args.models:
        if name in report['pairs'] and report['pairs'][name].get('done'):
            continue
        cand = read(name)
        win = tie = loss = invalid = picked_a = 0
        partial = args.dir / f'judge_{name}.partial.jsonl'            # progress survives a stop; finished pairs are not asked again
        log = [json.loads(l) for l in partial.read_text().splitlines()] if partial.exists() else []
        done = {(x['story'], x['candidate_is_a']) for x in log}
        for x in log:
            picked_a += x['verdict'] == 'A'
            tie += x['verdict'] == 'TIE'
            invalid += x['verdict'] == 'INVALID'
            win += x['verdict'] in ('A', 'B') and (x['verdict'] == 'A') == x['candidate_is_a']
            loss += x['verdict'] in ('A', 'B') and (x['verdict'] == 'A') != x['candidate_is_a']
        if log:
            print(f'  {name}: resuming after {len(log)} judgements', flush=True)
        t0 = time.time()
        for i, (r, c) in enumerate(zip(ref, cand)):
            for cand_is_a in (True, False):
                if (i, cand_is_a) in done:
                    continue
                if wait_until_free(args.url, args.judge):
                    print(f'  {time.strftime("%H:%M:%S")} llama-swap was busy with another model; waited', flush=True)
                a, b = (c, r) if cand_is_a else (r, c)
                verdict, text = ask(args.url, args.judge, PROMPT.format(beginning=r['beginning'], a=a['continuation'], b=b['continuation']))
                picked_a += verdict == 'A'
                if verdict == 'TIE':
                    tie += 1
                elif verdict == 'INVALID':
                    invalid += 1
                elif (verdict == 'A') == cand_is_a:
                    win += 1
                else:
                    loss += 1
                log.append(dict(story=i, candidate_is_a=cand_is_a, verdict=verdict, text=text[-300:]))
                with partial.open('a') as f:
                    f.write(json.dumps(log[-1]) + '\n')
            if (i + 1) % 10 == 0:
                print(f'  {name}: {i + 1}/{len(ref)} beginnings, candidate wins {win}, ties {tie}, reference wins {loss} ({(time.time() - t0) / 60:.1f} min)', flush=True)
        decided = win + loss
        share = win / decided if decided else float('nan')
        half = 1.96 * math.sqrt(share * (1 - share) / decided) if decided else float('nan')
        report['pairs'][name] = dict(done=True, judgements=len(log), candidate_wins=win, ties=tie, reference_wins=loss, invalid=invalid,
                                     win_share_of_decided=share, ci95=half, picked_position_a=picked_a / max(1, len(log) - invalid - tie))
        log.sort(key=lambda x: (x['story'], not x['candidate_is_a']))
        (args.dir / f'judge_{name}.jsonl').write_text('\n'.join(json.dumps(x) for x in log) + '\n')
        partial.unlink(missing_ok=True)
        out_path.write_text(json.dumps(report, indent=1))
        print(f'{name} against {args.reference}: wins {win}, ties {tie}, losses {loss}; win share {share * 100:.1f}% ±{half * 100:.1f}', flush=True)
    md = ['# Judged story comparison', '', f"Judge: `{report['judge']}` (local). Reference: `{report['reference']}`. Each beginning is judged twice, "
          'with the two stories in both orders. 50% means the judge cannot tell the models apart.', '',
          '| Candidate | Judgements | Candidate better | Tie | Reference better | Win share of decided | Judge picked position A |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, p in report['pairs'].items():
        md.append(f"| {name} | {p['judgements']} | {p['candidate_wins']} | {p['ties']} | {p['reference_wins']} | "
                  f"{p['win_share_of_decided'] * 100:.1f}% ±{p['ci95'] * 100:.1f} | {p['picked_position_a'] * 100:.0f}% |")
    (args.dir / 'judge.md').write_text('\n'.join(md) + '\n')
    print(f'saved {args.dir / "judge.md"}', flush=True)


if __name__ == '__main__':
    main()
