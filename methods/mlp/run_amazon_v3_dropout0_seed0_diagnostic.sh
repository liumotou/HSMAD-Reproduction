#!/usr/bin/env bash
set -u
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${HSMAD_PYTHON:-python}"
cd "$PROJECT_ROOT"
root=results/experiments/mlp/amazon/protocol_v3_dropout0_seed0_diagnostic
mkdir -p "$root"
tmp="$root/.seed_0.tmp"
"$PYTHON" -u -m methods.mlp.src.amazon_v3_dropout0_seed0_diagnostic --seed 0 --config methods/mlp/configs/amazon_protocol_v3_dropout0_seed0_diagnostic.json > "$tmp" 2>&1
status=$?
log="$root/seed_0/terminal.log"
if [ -f "$log" ]; then cat "$tmp" >> "$log"; else mv "$tmp" "$log"; fi
rm -f "$tmp"
exit "$status"
