#!/usr/bin/env bash
set -u
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${HSMAD_PYTHON:-python}"
cd "$PROJECT_ROOT"
root=results/experiments/mlp/amazon/protocol_v3_dropout0/formal
mkdir -p "$root"
"$PYTHON" -m methods.mlp.src.summarize_amazon_v3_dropout0_formal --precheck-only > "$root/seed_0_precheck.log" 2>&1
for seed in 1 2 3 4 5 6 7 8 9; do
  tmp="$root/.seed_${seed}.tmp"
  "$PYTHON" -u -m methods.mlp.src.amazon_v3_dropout0_formal_runner --seed "$seed" --config methods/mlp/configs/amazon_protocol_v3_dropout0_seed0_diagnostic.json > "$tmp" 2>&1
  status=$?
  log="$root/seed_${seed}/terminal.log"
  if [ -f "$log" ]; then cat "$tmp" >> "$log"; else mv "$tmp" "$log"; fi
  rm -f "$tmp"
  if [ "$status" -ne 0 ]; then exit "$status"; fi
done
"$PYTHON" -m methods.mlp.src.summarize_amazon_v3_dropout0_formal > "$root/summary_command.log" 2>&1
