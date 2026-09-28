#!/usr/bin/env bash
set -u
dataset="$1"
tolerance="${2:-0.000000000001}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
logs="results/experiments/amnet_hsmad/${dataset}/candidate_official_adapted/formal/_launcher_logs"
mkdir -p "$logs"
for seed in $(seq 0 9); do
  config="methods/amnet_hsmad/configs/${dataset}_formal_seed${seed}.json"
  output="results/experiments/amnet_hsmad/${dataset}/candidate_official_adapted/formal/seed_${seed}"
  log="$logs/seed_${seed}.terminal.log"
  if [ -e "$output" ]; then echo "REFUSE_EXISTING $output" | tee "$log"; echo 99 > "$logs/seed_${seed}.exit_code.txt"; continue; fi
  timeout 1800 methods/amnet_hsmad/run_in_official_env.sh -m methods.amnet_hsmad.run_full --config "$config" > "$log" 2>&1
  code=$?; echo "$code" > "$logs/seed_${seed}.exit_code.txt"
  if [ "$code" -eq 0 ]; then
    cp "$log" "$output/terminal.log"
    timeout 300 methods/amnet_hsmad/run_in_official_env.sh -m methods.amnet_hsmad.recompute_checkpoint --run-dir "$output" --audit-name audit_recompute_tolerance --absolute-tolerance "$tolerance" > "$logs/seed_${seed}.audit.log" 2>&1
    audit_code=$?; echo "$audit_code" > "$logs/seed_${seed}.audit_exit_code.txt"
  fi
  echo "SEED_COMPLETE dataset=$dataset seed=$seed exit=$code"
done
