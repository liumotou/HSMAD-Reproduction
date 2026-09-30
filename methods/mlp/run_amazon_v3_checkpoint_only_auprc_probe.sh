#!/usr/bin/env bash
set -u
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${HSMAD_PYTHON:-python}"
cd "$PROJECT_ROOT"
base=results/experiments/mlp/amazon/v3_checkpoint_only_auprc_probe
mkdir -p "$base"
tmp="$base/.terminal.tmp"
"$PYTHON" -u -m methods.mlp.src.amazon_v3_checkpoint_only_auprc_probe --config methods/mlp/configs/amazon_protocol_v3_dropout0_seed0_diagnostic.json > "$tmp" 2>&1
status=$?
if [ -d "$base/seed_3" ]; then mv "$tmp" "$base/seed_3/terminal.log"; fi
exit "$status"
