#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${HSMAD_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
project_root="$PROJECT_ROOT"
prefix="$project_root/.venvs/amnet_official_torch111_cu113"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$prefix/bin/python" "$@"
