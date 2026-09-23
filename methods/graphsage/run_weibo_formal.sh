#!/usr/bin/env bash
set -u

ROOT=/root/autodl-tmp/HSMAD
PYTHON=/root/miniconda3/bin/python
METHOD_DIR="$ROOT/methods/graphsage"
CONFIG="$METHOD_DIR/configs/weibo_graphsage_gadbench_h64_formal.json"
BASE="$ROOT/results/experiments/graphsage/weibo/graphsage_gadbench_h64_candidate/formal"

for seed in 0 1 2 3 4 5 6 7 8 9; do
  temporary_log="$BASE/terminal_seed_${seed}.tmp.log"
  "$PYTHON" "$METHOD_DIR/src/run_formal.py" --config "$CONFIG" --seed "$seed" > "$temporary_log" 2>&1
  exit_code=$?
  if [[ -d "$BASE/seed_${seed}" ]]; then
    mv -- "$temporary_log" "$BASE/seed_${seed}/terminal.log"
  fi
  printf 'seed=%s exit_code=%s completed_at_utc=%s\n' "$seed" "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

"$PYTHON" "$METHOD_DIR/src/build_summary.py" --formal-dir "$BASE"
