#!/usr/bin/env bash
set -u
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${HSMAD_PYTHON:-python}"
cd "$PROJECT_ROOT"
mkdir -p results/experiments/mlp/weibo/formal
for seed in 0 1 2 3 4 5 6 7 8 9; do
  temporary_log="results/experiments/mlp/weibo/formal/.seed_${seed}.terminal.tmp"
  "$PYTHON" -u -m methods.mlp.src.formal_runner --seed "$seed" --config methods/mlp/configs/weibo_formal.json > "$temporary_log" 2>&1
  status=$?
  terminal_log="results/experiments/mlp/weibo/formal/seed_${seed}/terminal.log"
  if [ -f "$terminal_log" ]; then cat "$temporary_log" >> "$terminal_log"; else mv "$temporary_log" "$terminal_log"; fi
  rm -f "$temporary_log"
  if [ "$status" -ne 0 ]; then exit "$status"; fi
done
"$PYTHON" -m methods.mlp.src.summarize_formal > results/experiments/mlp/weibo/formal/summary_command.log 2>&1
