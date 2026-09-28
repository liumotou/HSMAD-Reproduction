#!/usr/bin/env bash
set -o pipefail
cd /root/autodl-tmp/HSMAD
OUT=results/experiments/cgadm_hsmad/weibo/cgadm_hsmad_candidate/smoke/seed_0
mkdir -p "$(dirname "$OUT")"
timeout 7200 .venvs/pyg_standard_baselines_conda/bin/python -m methods.cgadm_hsmad.src.runner \
  --root /root/autodl-tmp/HSMAD \
  --config /root/autodl-tmp/HSMAD/methods/cgadm_hsmad/configs/weibo_candidate.json \
  --seed 0 --run-type smoke --max-epoch 5 --patience 50 \
  2>&1 | tee /tmp/cgadm_weibo_smoke_terminal.log
code=${PIPESTATUS[0]}
if [[ -d "$OUT" ]]; then
  cp /tmp/cgadm_weibo_smoke_terminal.log "$OUT/terminal.log"
  echo "$code" > "$OUT/exit_code.txt"
else
  mkdir -p "$OUT"
  cp /tmp/cgadm_weibo_smoke_terminal.log "$OUT/terminal.log"
  echo "$code" > "$OUT/exit_code.txt"
fi
exit "$code"
