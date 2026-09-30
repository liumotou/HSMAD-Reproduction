#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
PYTHON="${HSMAD_PYTHON:-python}"
cd "$PROJECT_ROOT"
formal=results/experiments/cgadm_hsmad/amazon/cgadm_hsmad_candidate/formal
mkdir -p "$formal"
for seed in 0 1 2 3 4 5 6 7 8 9; do
  tmp="/tmp/cgadm_amazon_formal_seed_${seed}.log"
  PYTHONPATH=. timeout --signal=TERM --kill-after=30s 1800 \
    "$PYTHON" -m methods.cgadm_hsmad.src.runner \
    --root "$PROJECT_ROOT" \
    --config methods/cgadm_hsmad/configs/amazon_candidate.json \
    --seed "$seed" --run-type formal --max-epoch 1000 --patience 200 >"$tmp" 2>&1
  code=$?
  if test -d "$formal/seed_${seed}"; then
    cp "$tmp" "$formal/seed_${seed}/terminal.log"
    printf '%s\n' "$code" > "$formal/seed_${seed}/exit_code.txt"
  fi
  if test "$code" -eq 0; then
    PYTHONPATH=. timeout --signal=TERM --kill-after=30s 300 \
      "$PYTHON" -m methods.cgadm_hsmad.audit.recompute_checkpoint \
      --root "$PROJECT_ROOT" --run-dir "$formal/seed_${seed}" \
      --output-dir "$formal/seed_${seed}/audit_recompute" >"/tmp/cgadm_amazon_formal_seed_${seed}_audit.log" 2>&1
    audit_code=$?
    cp "/tmp/cgadm_amazon_formal_seed_${seed}_audit.log" "$formal/seed_${seed}/audit_recompute/terminal.log" 2>/dev/null || true
    printf '%s\n' "$audit_code" > "$formal/seed_${seed}/audit_recompute/exit_code.txt" 2>/dev/null || true
  fi
done
PYTHONPATH=. "$PYTHON" -m methods.cgadm_hsmad.audit.aggregate_formal --formal-dir "$formal" --dataset amazon > "$formal/aggregate.log" 2>&1
