#!/usr/bin/env python3
"""Placement planner: decide how many equal branch units each device runs, from measured data.

Inputs (all measured, nothing assumed about the cluster):
  - concurrency probes (docs/results/E0b_concurrency/<node>.json): total weight-reading rate of a device when it
    runs k branch streams side by side;
  - calibration pings (docs/results/E0_calibration/*/calibration.json): round trip for one token each way;
  - free memory per device (optional cap).

Model:  T_token = coordinator_ms + max over devices ( rtt_i + time_i(n_i) )
        time_i(n) = n * unit_mib / rate_i(min(n, k_max))     (units beyond the probed stream count queue at the ceiling)
Units are identical, so the greedy rule "give the next unit to the device that would finish it earliest" is optimal.

  planner.py --units 16 --unit-mib 62 --coordinator server
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / 'docs' / 'results'


def load_measurements():
    devices = {}
    for path in sorted(glob.glob(str(RESULTS / 'E0b_concurrency' / '*.json'))):
        for name, rep in json.load(open(path)).items():
            if isinstance(rep, dict) and 'rows' in rep:
                devices[name] = dict(rate={r['k']: r['gib_per_s_total'] for r in rep['rows']}, rtt_ms=0.0)
    for path in sorted(glob.glob(str(RESULTS / 'E0_calibration' / '*' / 'calibration.json'))):
        for name, rows in json.load(open(path)).get('ping', {}).items():
            if name in devices and isinstance(rows, list):
                for r in rows:
                    if r['up_bytes'] == 3520 and r['down_bytes'] == 3520:
                        devices[name]['rtt_ms'] = r['rtt_ms_p50']
    return devices


def rate(dev, k):
    ks = sorted(dev['rate'])
    k = min(k, ks[-1])
    if k in dev['rate']:
        return dev['rate'][k]
    lo = max(x for x in ks if x < k)
    hi = min(x for x in ks if x > k)
    return dev['rate'][lo] + (dev['rate'][hi] - dev['rate'][lo]) * (k - lo) / (hi - lo)


def leg_ms(dev, n, unit_mib, is_coordinator):
    if n == 0:
        return 0.0
    return (0.0 if is_coordinator else dev['rtt_ms']) + n * unit_mib / 1024 / rate(dev, n) * 1000


def plan(devices, units, unit_mib, coordinator, coordinator_ms, exclude=()):
    counts = {name: 0 for name in devices if name not in exclude}
    for _ in range(units):
        best = min(counts, key=lambda d: leg_ms(devices[d], counts[d] + 1, unit_mib, d == coordinator))
        counts[best] += 1
    legs = {d: leg_ms(devices[d], n, unit_mib, d == coordinator) for d, n in counts.items()}
    total = coordinator_ms + max(legs.values())
    return counts, legs, total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--units', type=int, default=16)
    ap.add_argument('--unit-mib', type=float, default=62.0, help='weights of one branch unit')
    ap.add_argument('--coordinator', default='server')
    ap.add_argument('--coordinator-ms', type=float, default=12.0, help='trunk + fusion + logits on the coordinator')
    ap.add_argument('--exclude', default='')
    args = ap.parse_args()
    devices = load_measurements()
    counts, legs, total = plan(devices, args.units, args.unit_mib, args.coordinator, args.coordinator_ms,
                               [x for x in args.exclude.split(',') if x])
    print(f'{args.units} units of {args.unit_mib:.0f} MiB, coordinator {args.coordinator} (+{args.coordinator_ms:.0f} ms)')
    for d, n in counts.items():
        print(f"  {d:8s} {n:>2} units  leg {legs[d]:6.1f} ms  (round trip {0.0 if d == args.coordinator else devices[d]['rtt_ms']:.1f} ms, "
              f"reading at {rate(devices[d], max(n, 1)):.1f} GiB/s)")
    print(f'  predicted {total:.1f} ms per token = {1000 / total:.1f} tok/s')
    return total


if __name__ == '__main__':
    main()
