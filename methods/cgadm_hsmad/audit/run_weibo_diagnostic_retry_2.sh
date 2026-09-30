#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${HSMAD_PYTHON:-python}"
cd "$PROJECT_ROOT"
log=results/experiments/cgadm_hsmad_weibo_diagnostic_seed0_retry2_launcher.log
PYTHONPATH=. timeout --signal=TERM --kill-after=30s 1800 \
  "$PYTHON" -m methods.cgadm_hsmad.src.runner \
  --root "$PROJECT_ROOT" \
  --config methods/cgadm_hsmad/configs/weibo_candidate.json \
  --seed 0 --run-type diagnostic --max-epoch 1000 --patience 200 \
  --output-suffix retry_2 >"$log" 2>&1
code=$?
printf '%s\n' "$code" > results/experiments/cgadm_hsmad_weibo_diagnostic_seed0_retry2_exit_code.txt
exit "$code"
