#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/HSMAD
base=results/experiments/mlp/tolokers/protocol_v5a_f1_earlystop_auprc_checkpoint/diagnostic
mkdir -p "$base"
tmp="$base/.terminal.tmp"
/root/miniconda3/bin/python -u methods/mlp/src/tolokers_v5a_seed0_diagnostic.py --config methods/mlp/configs/tolokers_protocol_v5a_seed0_diagnostic.json > "$tmp" 2>&1
status=$?
if [ -d "$base/seed_0" ]; then mv "$tmp" "$base/seed_0/terminal.log"; fi
exit "$status"
