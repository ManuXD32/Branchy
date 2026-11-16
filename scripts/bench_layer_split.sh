#!/usr/bin/env bash
# Layer split of the sequential model, the llama.cpp RPC layout (article section 4.7, experiment E14).
#   bash scripts/bench_layer_split.sh workerA [workerB]
set -e
cd "$(dirname "$0")/../src/node"
PY="${PYTHON:-python3}"; A="${1:?worker name}"; B="$2"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
S=../../models/sequential-313M/inference.pt
P=(--placement "alone=local:0-18" --placement "10+9=local:0-9;$A:10-18" --placement "14+5=local:0-13;$A:14-18")
[ -n "$B" ] && P+=(--placement "5+5+9=local:0-4;$A:5-9;$B:10-18")
"$PY" -u bench_cluster.py --checkpoint $S --master-threads 8 --runs 3 --prompts 32,256 --decodes 64 "${P[@]}" --out ../../results/layer_split
