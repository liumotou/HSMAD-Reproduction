#!/usr/bin/env bash
set -euo pipefail
project_root="/root/autodl-tmp/HSMAD"
prefix="$project_root/.venvs/amnet_official_torch111_cu113"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$prefix/bin/python" "$@"
