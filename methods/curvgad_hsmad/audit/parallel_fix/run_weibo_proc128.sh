#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${CURVGAD_PYTHON:-${HSMAD_PYTHON:-python}}"

cd "$PROJECT_ROOT"
out="methods/curvgad_hsmad/audit/weibo_exact_precompute_proc128"
mkdir -p "$out"
date -Is > "$out/started_at.txt"
free -h > "$out/memory_before.txt"
sha256sum \
  methods/curvgad_hsmad/src/precompute.py \
  methods/curvgad_hsmad/src/precompute_dataset.py \
  > "$out/code_sha256.txt"

set +e
timeout 7200 "$PYTHON" \
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
