#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/HSMAD
root=results/experiments/mlp/amazon/protocol_v3_dropout0/formal
mkdir -p "$root"
/root/miniconda3/bin/python methods/mlp/src/summarize_amazon_v3_dropout0_formal.py --precheck-only > "$root/seed_0_precheck.log" 2>&1
for seed in 1 2 3 4 5 6 7 8 9; do
  tmp="$root/.seed_${seed}.tmp"
  /root/miniconda3/bin/python -u methods/mlp/src/amazon_v3_dropout0_formal_runner.py --seed "$seed" --config methods/mlp/configs/amazon_protocol_v3_dropout0_seed0_diagnostic.json > "$tmp" 2>&1
  status=$?
  log="$root/seed_${seed}/terminal.log"
  if [ -f "$log" ]; then cat "$tmp" >> "$log"; else mv "$tmp" "$log"; fi
  rm -f "$tmp"
  if [ "$status" -ne 0 ]; then exit "$status"; fi
done
/root/miniconda3/bin/python methods/mlp/src/summarize_amazon_v3_dropout0_formal.py > "$root/summary_command.log" 2>&1
