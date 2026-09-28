#!/usr/bin/env bash
set -u
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${HSMAD_PYTHON:-python}"
cd "$PROJECT_ROOT"
root=results/experiments/mlp/weibo/protocol_v5a_f1_earlystop_auprc_checkpoint/formal_restart_01
export MLP_V5A_FORMAL_ROOT="$root"
mkdir -p "$root"
for seed in 0 1 2 3 4 5 6 7 8 9; do
 tmp="$root/.seed_${seed}.tmp"
 "$PYTHON" -u -m methods.mlp.src.weibo_v5a_f1stop_auprc_checkpoint_runner --seed "$seed" --config methods/mlp/configs/weibo_protocol_v5a_f1_earlystop_auprc_checkpoint.json > "$tmp" 2>&1
 status=$?
 log="$root/seed_${seed}/terminal.log"
 if [ -f "$log" ]; then cat "$tmp" >> "$log"; else mv "$tmp" "$log"; fi
 rm -f "$tmp"
 if [ "$status" -ne 0 ]; then exit "$status"; fi
done
"$PYTHON" -m methods.mlp.src.summarize_weibo_v5a > "$root/summary_command.log" 2>&1
