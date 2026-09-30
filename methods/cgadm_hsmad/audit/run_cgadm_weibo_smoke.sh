#!/usr/bin/env bash
set -o pipefail
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${HSMAD_PYTHON:-python}"
cd "$PROJECT_ROOT"
OUT=results/experiments/cgadm_hsmad/weibo/cgadm_hsmad_candidate/smoke/seed_0
mkdir -p "$(dirname "$OUT")"
timeout 7200 "$PYTHON" -m methods.cgadm_hsmad.src.runner \
  --root "$PROJECT_ROOT" \
  --config "$PROJECT_ROOT/methods/cgadm_hsmad/configs/weibo_candidate.json" \
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
