#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/HSMAD
OUT="$ROOT/results/experiments/mlp/tfinance/protocol_v5a_f1_earlystop_auprc_checkpoint/diagnostic"
PY=/root/miniconda3/bin/python
cd "$ROOT"
mkdir -p "$OUT"
test ! -e "$OUT/seed_0" || { echo "refusing to overwrite existing seed_0" >&2; exit 2; }
CMD="$PY methods/mlp/src/tfinance_v5a_seed0_diagnostic.py --config methods/mlp/configs/tfinance_protocol_v5a_seed0_diagnostic.json"
printf '%s\n' "$CMD" > "$OUT/launch_command.txt"
{ date -Is; pwd; "$PY" -V; "$PY" -c 'import torch,dgl; print({"torch":torch.__version__,"cuda":torch.version.cuda,"dgl":dgl.__version__,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})'; } > "$OUT/launch_environment.txt" 2>&1
date -Is > "$OUT/launch_time.txt"
nohup "$PY" methods/mlp/src/tfinance_v5a_seed0_diagnostic.py --config methods/mlp/configs/tfinance_protocol_v5a_seed0_diagnostic.json > "$OUT/run.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$OUT/run.pid"
kill -0 "$pid"
ps -p "$pid" -o pid,ppid,stat,lstart,cmd > "$OUT/process_start.txt"
echo "$pid"
