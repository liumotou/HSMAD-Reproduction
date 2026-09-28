#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/HSMAD
base=results/experiments/mlp/amazon/protocol_v5_val_auprc_selection
mkdir -p "$base"
tmp="$base/.seed3_verify.tmp"
/root/miniconda3/bin/python -u methods/mlp/src/amazon_v5_val_auprc_runner.py --seed 3 --mode seed3_verify --config methods/mlp/configs/amazon_protocol_v5_val_auprc_selection.json > "$tmp" 2>&1
status=$?
mv "$tmp" "$base/seed3_reproduction_check/seed_3/terminal.log"
if [ "$status" -ne 0 ]; then exit "$status"; fi
mkdir -p "$base/formal"
for seed in 0 1 2 3 4 5 6 7 8 9; do
 tmp="$base/formal/.seed_${seed}.tmp"
 /root/miniconda3/bin/python -u methods/mlp/src/amazon_v5_val_auprc_runner.py --seed "$seed" --mode formal --config methods/mlp/configs/amazon_protocol_v5_val_auprc_selection.json > "$tmp" 2>&1
 status=$?
 log="$base/formal/seed_${seed}/terminal.log"
 if [ -f "$log" ]; then cat "$tmp" >> "$log"; else mv "$tmp" "$log"; fi
 rm -f "$tmp"
 if [ "$status" -ne 0 ]; then exit "$status"; fi
done
/root/miniconda3/bin/python methods/mlp/src/summarize_amazon_v5_val_auprc.py > "$base/formal/summary_command.log" 2>&1
