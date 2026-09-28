#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/HSMAD
root=results/experiments/mlp/weibo/protocol_v5a_f1_earlystop_auprc_checkpoint/formal_restart_01
export MLP_V5A_FORMAL_ROOT="$root"
mkdir -p "$root"
for seed in 0 1 2 3 4 5 6 7 8 9; do
 tmp="$root/.seed_${seed}.tmp"
 /root/miniconda3/bin/python -u methods/mlp/src/weibo_v5a_f1stop_auprc_checkpoint_runner.py --seed "$seed" --config methods/mlp/configs/weibo_protocol_v5a_f1_earlystop_auprc_checkpoint.json > "$tmp" 2>&1
 status=$?
 log="$root/seed_${seed}/terminal.log"
 if [ -f "$log" ]; then cat "$tmp" >> "$log"; else mv "$tmp" "$log"; fi
 rm -f "$tmp"
 if [ "$status" -ne 0 ]; then exit "$status"; fi
done
/root/miniconda3/bin/python methods/mlp/src/summarize_weibo_v5a.py > "$root/summary_command.log" 2>&1
