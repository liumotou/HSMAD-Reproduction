#!/usr/bin/env bash
set -u
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${HSMAD_PYTHON:-python}"
cd "$PROJECT_ROOT"
base=results/experiments/mlp/amazon/protocol_v5_val_auprc_selection
mkdir -p "$base"
tmp="$base/.seed3_verify.tmp"
"$PYTHON" -u -m methods.mlp.src.amazon_v5_val_auprc_runner --seed 3 --mode seed3_verify --config methods/mlp/configs/amazon_protocol_v5_val_auprc_selection.json > "$tmp" 2>&1
status=$?
mv "$tmp" "$base/seed3_reproduction_check/seed_3/terminal.log"
if [ "$status" -ne 0 ]; then exit "$status"; fi
mkdir -p "$base/formal"
for seed in 0 1 2 3 4 5 6 7 8 9; do
 tmp="$base/formal/.seed_${seed}.tmp"
 "$PYTHON" -u -m methods.mlp.src.amazon_v5_val_auprc_runner --seed "$seed" --mode formal --config methods/mlp/configs/amazon_protocol_v5_val_auprc_selection.json > "$tmp" 2>&1
 status=$?
 log="$base/formal/seed_${seed}/terminal.log"
 if [ -f "$log" ]; then cat "$tmp" >> "$log"; else mv "$tmp" "$log"; fi
 rm -f "$tmp"
 if [ "$status" -ne 0 ]; then exit "$status"; fi
done
"$PYTHON" -m methods.mlp.src.summarize_amazon_v5_val_auprc > "$base/formal/summary_command.log" 2>&1
