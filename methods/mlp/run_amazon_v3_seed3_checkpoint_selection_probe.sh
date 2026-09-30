#!/usr/bin/env bash
set -u
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${HSMAD_PYTHON:-python}"
cd "$PROJECT_ROOT"
root=results/experiments/mlp/amazon/protocol_v3_dropout0/checkpoint_selection_probe/seed_3
tmp=results/experiments/mlp/amazon/protocol_v3_dropout0/checkpoint_selection_probe/.terminal.tmp
"$PYTHON" -u -m methods.mlp.src.amazon_v3_seed3_checkpoint_selection_probe --config methods/mlp/configs/amazon_protocol_v3_dropout0_seed0_diagnostic.json > "$tmp" 2>&1
status=$?
if [ -d "$root" ]; then mv "$tmp" "$root/terminal.log"; fi
exit "$status"
