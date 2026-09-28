#!/usr/bin/env bash
set -u
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${HSMAD_PYTHON:-python}"
cd "$PROJECT_ROOT"
mkdir -p results/experiments/mlp/weibo/protocol_v3_dropout0
for seed in 0 1 2 3 4 5 6 7 8 9; do
  tmp="results/experiments/mlp/weibo/protocol_v3_dropout0/.seed_${seed}.tmp"
  "$PYTHON" -u -m methods.mlp.src.v3_dropout0_candidate_runner --seed "$seed" --config methods/mlp/configs/weibo_protocol_v3_dropout0_candidate.json > "$tmp" 2>&1
  status=$?
  log="results/experiments/mlp/weibo/protocol_v3_dropout0/seed_${seed}/terminal.log"
  if [ -f "$log" ]; then cat "$tmp" >> "$log"; else mv "$tmp" "$log"; fi
  rm -f "$tmp"
  if [ "$status" -ne 0 ]; then exit "$status"; fi
done
"$PYTHON" -m methods.mlp.src.summarize_v3_dropout0_candidate > results/experiments/mlp/weibo/protocol_v3_dropout0/summary_command.log 2>&1
