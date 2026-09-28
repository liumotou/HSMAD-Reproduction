#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/HSMAD
formal_root="results/experiments/graphconsis/tfinance/graphconsis_dgfraud_single_relation_h64_candidate/formal"

for seed in 0 1 2 3 4 5 6 7 8 9; do
  run_dir="${formal_root}/seed_${seed}"
  if [[ -e "${run_dir}" ]]; then
    echo "REFUSE_EXISTING_${run_dir}"
    exit 20
  fi
done

for seed in 0 1 2 3 4 5 6 7 8 9; do
  echo "START_SEED_${seed}"
  .venvs/pyg_standard_baselines_conda/bin/python -m methods.graphconsis.src.run \
    --config "methods/graphconsis/configs/tfinance_formal_seed_${seed}.json"
  .venvs/pyg_standard_baselines_conda/bin/python -m methods.graphconsis.audit.recompute \
    --output "${formal_root}/seed_${seed}"
done

.venvs/pyg_standard_baselines_conda/bin/python -m methods.graphconsis.audit.summarize \
  --formal "${formal_root}"
