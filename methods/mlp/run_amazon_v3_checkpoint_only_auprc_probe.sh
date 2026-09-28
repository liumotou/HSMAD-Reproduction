#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/HSMAD
base=results/experiments/mlp/amazon/v3_checkpoint_only_auprc_probe
mkdir -p "$base"
tmp="$base/.terminal.tmp"
/root/miniconda3/bin/python -u methods/mlp/src/amazon_v3_checkpoint_only_auprc_probe.py --config methods/mlp/configs/amazon_protocol_v3_dropout0_seed0_diagnostic.json > "$tmp" 2>&1
status=$?
if [ -d "$base/seed_3" ]; then mv "$tmp" "$base/seed_3/terminal.log"; fi
exit "$status"
