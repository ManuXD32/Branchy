#!/usr/bin/env bash
# Branch split with delayed execution (article section 4.6, experiment E12/E13), this machine coordinating.
#   bash scripts/bench_branch_split.sh workerA workerB        (names from src/node/cluster.json; one or two workers)
set -e
cd "$(dirname "$0")/../src/node"
PY="${PYTHON:-python3}"; A="${1:?worker name}"; B="$2"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
M=../../models/branched8-309M/inference.pt; S=../../models/sequential-313M/inference.pt
P=(--placement "alone=local:0,1,2,3,4,5,6,7" --placement "4+4=local:0,1,2,3;$A:4,5,6,7" --placement "6+2=local:0,1,2,3,4,5;$A:6,7")
[ -n "$B" ] && P+=(--placement "4+2+2=local:0,1,2,3;$A:4,5;$B:6,7" --placement "2+3+3=local:0,1;$A:2,3,4;$B:5,6,7" --placement "4+2+1,one-missing=local:0,1,2,3;$A:4,5;$B:6;none:7")
"$PY" -u bench_cluster.py --checkpoint $M --sequential $S --master-threads 8 --runs 3 --prompts 32,256 --decodes 64 "${P[@]}" --out ../../results/branch_split
