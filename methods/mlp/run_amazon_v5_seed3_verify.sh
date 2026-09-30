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
if [ -d "$base/seed3_reproduction_check/seed_3" ]; then mv "$tmp" "$base/seed3_reproduction_check/seed_3/terminal.log"; fi
exit "$status"
