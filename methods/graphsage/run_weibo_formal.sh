#!/usr/bin/env bash
set -u
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${HSMAD_PYTHON:-python}"

ROOT="$PROJECT_ROOT"
METHOD_DIR="$ROOT/methods/graphsage"
CONFIG="$METHOD_DIR/configs/weibo_graphsage_gadbench_h64_formal.json"
BASE="$ROOT/results/experiments/graphsage/weibo/graphsage_gadbench_h64_candidate/formal"

for seed in 0 1 2 3 4 5 6 7 8 9; do
  temporary_log="$BASE/terminal_seed_${seed}.tmp.log"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
    "$PYTHON" -m methods.graphsage.src.run_formal --config "$CONFIG" --seed "$seed" > "$temporary_log" 2>&1
  exit_code=$?
  if [[ -d "$BASE/seed_${seed}" ]]; then
    mv -- "$temporary_log" "$BASE/seed_${seed}/terminal.log"
  fi
  printf 'seed=%s exit_code=%s completed_at_utc=%s\n' "$seed" "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

"$PYTHON" "$METHOD_DIR/src/build_summary.py" --formal-dir "$BASE"
