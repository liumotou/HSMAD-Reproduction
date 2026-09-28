"""Generate auditable CARE-GNN Weibo formal configs from the frozen diagnostic config."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/root/autodl-tmp/HSMAD")
SOURCE = ROOT / "methods/caregnn/configs/weibo_diagnostic.json"
TARGET = ROOT / "methods/caregnn/configs"
PROTOCOL = "caregnn_official_single_flattened_relation_h64_candidate"


def main() -> None:
    base = json.loads(SOURCE.read_text(encoding="utf-8"))
    for seed in range(10):
        config = dict(base)
        config["seed"] = seed
        config["run_type"] = "formal"
        config["result_dir"] = (
            f"results/experiments/caregnn/weibo/{PROTOCOL}/formal/seed_{seed}"
        )
        path = TARGET / f"weibo_formal_seed_{seed}.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
