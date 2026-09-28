#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/HSMAD
mkdir -p results/experiments/mlp/weibo/formal
for seed in 0 1 2 3 4 5 6 7 8 9; do
  temporary_log="results/experiments/mlp/weibo/formal/.seed_${seed}.terminal.tmp"
  /root/miniconda3/bin/python -u methods/mlp/src/formal_runner.py --seed "$seed" --config methods/mlp/configs/weibo_formal.json > "$temporary_log" 2>&1
  status=$?
  terminal_log="results/experiments/mlp/weibo/formal/seed_${seed}/terminal.log"
  if [ -f "$terminal_log" ]; then cat "$temporary_log" >> "$terminal_log"; else mv "$temporary_log" "$terminal_log"; fi
  rm -f "$temporary_log"
  if [ "$status" -ne 0 ]; then exit "$status"; fi
done
/root/miniconda3/bin/python methods/mlp/src/summarize_formal.py > results/experiments/mlp/weibo/formal/summary_command.log 2>&1
