#!/usr/bin/env bash
# Devices with too little RAM for the model (article sections 4.8 and 4.9).
#   bash scripts/bench_low_memory.sh devices COORDINATOR WORKER [BUDGET_MB]   real devices: a worker node coordinates,
#        both get a RAM budget for weights that the process enforces itself (no root needed, works on Android)
#   bash scripts/bench_low_memory.sh simulated                                two memory-limited processes on this
#        Linux machine (needs root for systemd-run memory limits), sweep over memory sizes
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if [ "$1" = simulated ]; then
  exec "$PY" -u src/lowmem/run_lowmem.py --out results/low_memory_simulated --weights-mb 0,1100,900,800,700,600,500,400 --seq models/sequential-313M/inference.pt --branched models/branched8-309M/inference.pt ${CPUS_A:+--cpus-a $CPUS_A} ${CPUS_B:+--cpus-b $CPUS_B}
fi
C="${2:?coordinator node name}"; W="${3:?worker node name}"; RAM="${4:-800}"
cd src/node
for ram in 0 "$RAM"; do
  extra=""; [ "$ram" != 0 ] && extra="--ram-mb $ram --peer-ram-mb $ram"   # reference: ordinary loading; add --mmap to map without a budget
  "$PY" -u remote_coord.py --coordinator "$C" --peers "$W" --checkpoint ../../models/sequential-313M/inference.pt --prompts 32 --decodes 24 --runs 2 \
    $extra --placement "alone=local:0-18" --placement "split=local:0-9;$W:10-18" --out ../../results/low_memory_devices/sequential_$ram
  "$PY" -u remote_coord.py --coordinator "$C" --peers "$W" --checkpoint ../../models/branched8-309M/inference.pt --prompts 32 --decodes 24 --runs 2 \
    $extra --placement "alone=local:0,1,2,3,4,5,6,7" --placement "split=local:0,1,2,3;$W:4,5,6,7" --out ../../results/low_memory_devices/branched_$ram
done
