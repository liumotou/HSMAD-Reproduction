#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/HSMAD
root=results/experiments/mlp/amazon/protocol_v3_dropout0/checkpoint_selection_probe/seed_3
tmp=results/experiments/mlp/amazon/protocol_v3_dropout0/checkpoint_selection_probe/.terminal.tmp
/root/miniconda3/bin/python -u methods/mlp/src/amazon_v3_seed3_checkpoint_selection_probe.py --config methods/mlp/configs/amazon_protocol_v3_dropout0_seed0_diagnostic.json > "$tmp" 2>&1
status=$?
if [ -d "$root" ]; then mv "$tmp" "$root/terminal.log"; fi
exit "$status"
