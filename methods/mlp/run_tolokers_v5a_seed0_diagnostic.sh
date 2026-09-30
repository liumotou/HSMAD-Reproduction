#!/usr/bin/env bash
set -u
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${HSMAD_PYTHON:-python}"
cd "$PROJECT_ROOT"
base=results/experiments/mlp/tolokers/protocol_v5a_f1_earlystop_auprc_checkpoint/diagnostic
mkdir -p "$base"
tmp="$base/.terminal.tmp"
"$PYTHON" -u -m methods.mlp.src.tolokers_v5a_seed0_diagnostic --config methods/mlp/configs/tolokers_protocol_v5a_seed0_diagnostic.json > "$tmp" 2>&1
status=$?
if [ -d "$base/seed_0" ]; then mv "$tmp" "$base/seed_0/terminal.log"; fi
exit "$status"
