#!/usr/bin/env python3
"""Static validation for the public HSMAD reproduction snapshot."""
from __future__ import annotations

import json
import py_compile
import re
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "main.py",
    "model.py",
    "dataset.py",
    "docs/DATASETS.md",
    "docs/BASELINE_SCRIPTS.md",
    "methods/gcn/src/run_kipf_v2.py",
    "methods/gat_v2_gadbench/src/run_formal.py",
    "methods/graphsage/src/run_formal.py",
    "methods/bwgnn/src/run_formal.py",
    "methods/sparsegad/src/run_formal.py",
    "methods/project_paths.py",
    "methods/mlp/src/formal_runner.py",
    "methods/gcn/src/kipf_two_layer.py",
    "methods/caregnn/src/run.py",
    "methods/chebnet/src/run.py",
    "methods/gin/src/run.py",
    "methods/graphconsis/src/run.py",
    "methods/gwnn/src/run.py",
    "methods/pcgnn/src/run.py",
    "methods/spacegnn_hsmad/src/runner.py",
    "methods/amnet_hsmad/run_smoke.py",
    "methods/pmp_hsmad/src/runner.py",
    "methods/sparsegad/audit/recompute_run.py",
    "methods/nrgl/audit/recompute_checkpoint.py",
]
EXCLUDED_PARTS = {"audit", "_source_audit_refs", "__pycache__", ".venvs", ".venv"}
MAX_PUBLISHED_FILE = 10 * 1024 * 1024
FORBIDDEN_TRACKED_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".whl", ".pyc"}
MACHINE_SPECIFIC_PATH = re.compile(
    r"/root/(?:autodl-tmp|miniconda3)(?:/|\b)|[A-Za-z]:\\"
)
def published(path: Path) -> bool:
    return not any(part in EXCLUDED_PARTS for part in path.parts)


def main() -> int:
    errors: list[str] = []
    checked_python = checked_json = 0

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    executable_paths = [
        *sorted((ROOT / "methods").rglob("*.py")),
        *sorted((ROOT / "methods").rglob("*.sh")),
    ]
    for path in executable_paths:
        relative = path.relative_to(ROOT)
        if not published(relative):
            continue
        if MACHINE_SPECIFIC_PATH.search(path.read_text(encoding="utf-8", errors="replace")):
            errors.append(f"machine-specific path pinned by executable code: {relative}")

    for path in sorted((ROOT / "methods").rglob("*.json")):
        if not published(path.relative_to(ROOT)):
            continue
        checked_json += 1
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:  # report path and parser error
            errors.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")

    python_files = [
        *[ROOT / name for name in ("main.py", "model.py", "dataset.py", "manifold_update.py", "utils.py")],
        *sorted((ROOT / "methods").rglob("*.py")),
    ]
    for path in python_files:
        if not path.is_file() or not published(path.relative_to(ROOT)):
            continue
        checked_python += 1
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            errors.append(f"Python syntax error {path.relative_to(ROOT)}: {exc}")

    tracked_output = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    for raw_name in tracked_output.split(b"\0"):
        if not raw_name:
            continue
        relative = Path(raw_name.decode("utf-8"))
        path = ROOT / relative
        if path.is_file() and path.stat().st_size > MAX_PUBLISHED_FILE:
            errors.append(f"tracked file over 10 MiB: {relative} ({path.stat().st_size} bytes)")
        if path.suffix.lower() in FORBIDDEN_TRACKED_SUFFIXES:
            errors.append(f"tracked generated/binary artifact: {relative}")

    print(f"checked_python={checked_python}")
    print(f"checked_json={checked_json}")
    if errors:
        print("status=FAILED")
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("status=STATIC_VALIDATION_OK")
    print("note=This does not validate CUDA, DGL/PyG imports, datasets, or full training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
