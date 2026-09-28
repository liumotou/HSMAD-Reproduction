#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/HSMAD
base=results/experiments/mlp/amazon/protocol_v5_val_auprc_selection
mkdir -p "$base"
tmp="$base/.seed3_verify.tmp"
/root/miniconda3/bin/python -u methods/mlp/src/amazon_v5_val_auprc_runner.py --seed 3 --mode seed3_verify --config methods/mlp/configs/amazon_protocol_v5_val_auprc_selection.json > "$tmp" 2>&1
status=$?
if [ -d "$base/seed3_reproduction_check/seed_3" ]; then mv "$tmp" "$base/seed3_reproduction_check/seed_3/terminal.log"; fi
exit "$status"
