"""Promote one audited PC-GNN smoke config to diagnostic and immutable formal configs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from methods.project_paths import project_root

ROOT = project_root()


def write_new(path: Path, value: object) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--create-formal", action="store_true")
    args = parser.parse_args()
    source = Path(args.smoke_config)
    base = json.loads(source.read_text(encoding="utf-8"))
    if base["dataset"] != args.dataset or base["run_type"] != "smoke":
        raise RuntimeError("source must be the matching audited smoke config")
    protocol = base["protocol_version"]
    diagnostic = dict(base)
    diagnostic.update({
        "seed": 0,
        "run_type": "diagnostic",
        "max_epoch": 51,
        "result_dir": f"results/experiments/pcgnn/{args.dataset}/{protocol}/diagnostic/seed_0",
    })
    destination = ROOT / "methods/pcgnn/configs" / f"{args.dataset}_diagnostic.json"
    if destination.exists() and args.create_formal:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != diagnostic:
            raise RuntimeError("existing diagnostic config differs from audited promotion")
        manifest = {destination.name: hashlib.sha256(destination.read_bytes()).hexdigest()}
    else:
        manifest = {destination.name: write_new(destination, diagnostic)}
    if args.create_formal:
        for seed in range(10):
            config = dict(diagnostic)
            config.update({
                "seed": seed,
                "run_type": "formal",
                "result_dir": f"results/experiments/pcgnn/{args.dataset}/{protocol}/formal/seed_{seed}",
            })
            path = ROOT / "methods/pcgnn/configs/formal" / f"{args.dataset}_seed_{seed}.json"
            manifest[path.name] = write_new(path, config)
        write_new(ROOT / "methods/pcgnn/configs/formal" / f"{args.dataset}_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
