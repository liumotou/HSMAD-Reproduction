#!/usr/bin/env bash
set -uo pipefail

cd /root/autodl-tmp/HSMAD
out="methods/curvgad_hsmad/audit/weibo_exact_precompute_proc128"
mkdir -p "$out"
date -Is > "$out/started_at.txt"
free -h > "$out/memory_before.txt"
sha256sum \
  methods/curvgad_hsmad/src/precompute.py \
  methods/curvgad_hsmad/src/precompute_dataset.py \
  > "$out/code_sha256.txt"

set +e
timeout 7200 .venvs/curvgad_candidate_conda/bin/python \
  -m methods.curvgad_hsmad.src.precompute_dataset \
  --dataset weibo \
  --dataset-path datasets/weibo \
  --cache-dir cache/curvgad_exact \
  --preflight "$out/preflight.json" \
  --method exact_orc \
  --alpha 0.5 \
  --proc 128 \
  2>&1 | tee "$out/terminal.log"
code=${PIPESTATUS[0]}
set -e

printf '%s\n' "$code" > "$out/exit_code.txt"
date -Is > "$out/finished_at.txt"
free -h > "$out/memory_after.txt"
exit "$code"
