"""Generate immutable PC-GNN Tolokers formal configs from the audited diagnostic config."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from methods.project_paths import project_root

ROOT = project_root()
BASE = ROOT / "methods/pcgnn/configs/tolokers_diagnostic.json"
DEST = ROOT / "methods/pcgnn/configs/formal"


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for seed in range(10):
        config = dict(base)
        config["seed"] = seed
        config["run_type"] = "formal"
        config["result_dir"] = (
            "results/experiments/pcgnn/tolokers/"
            "pcgnn_official_single_flattened_relation_h64_candidate/formal/"
            f"seed_{seed}"
        )
        path = DEST / f"tolokers_seed_{seed}.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        payload = json.dumps(config, indent=2, sort_keys=True) + "\n"
        path.write_text(payload, encoding="utf-8")
        manifest[path.name] = hashlib.sha256(payload.encode()).hexdigest()
    output = DEST / "tolokers_manifest.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
