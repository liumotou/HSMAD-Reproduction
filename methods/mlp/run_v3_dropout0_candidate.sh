#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/HSMAD
mkdir -p results/experiments/mlp/weibo/protocol_v3_dropout0
for seed in 0 1 2 3 4 5 6 7 8 9; do
  tmp="results/experiments/mlp/weibo/protocol_v3_dropout0/.seed_${seed}.tmp"
  /root/miniconda3/bin/python -u methods/mlp/src/v3_dropout0_candidate_runner.py --seed "$seed" --config methods/mlp/configs/weibo_protocol_v3_dropout0_candidate.json > "$tmp" 2>&1
  status=$?
  log="results/experiments/mlp/weibo/protocol_v3_dropout0/seed_${seed}/terminal.log"
  if [ -f "$log" ]; then cat "$tmp" >> "$log"; else mv "$tmp" "$log"; fi
  rm -f "$tmp"
  if [ "$status" -ne 0 ]; then exit "$status"; fi
done
/root/miniconda3/bin/python methods/mlp/src/summarize_v3_dropout0_candidate.py > results/experiments/mlp/weibo/protocol_v3_dropout0/summary_command.log 2>&1
