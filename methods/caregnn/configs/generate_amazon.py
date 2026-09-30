"""Generate CARE-GNN Amazon configs from the frozen protocol."""
from __future__ import annotations

import json
from pathlib import Path
from methods.project_paths import project_root


ROOT = project_root()
SOURCE = ROOT / "methods/caregnn/configs/weibo_diagnostic.json"
TARGET = ROOT / "methods/caregnn/configs"
PROTOCOL = "caregnn_official_single_flattened_relation_h64_candidate"
HASHES = {
    "dataset_file_sha256": "25f77cdd9f5991c575ce5efd385aec86e30123585e9b87c1da0bcb25d5f67872",
    "feature_sha256": "063d12c2db83b93229574fff15c7c65c8b2f6e1a686f7c7e0212722c583742ad",
    "label_sha256": "d5e3de05cedaaabbde98665fd6ba21de71cea2831d0998ea766f8f8773cd83c5",
    "test_mask_sha256": "bc42f2677dbf8357e949a30330b0236fbe6435c91583aa0232a249a20f95e9d1",
    "train_mask_sha256": "fb95bd68eda65b33435b2214bd1ceff41dfa324b5fbed8bcfeb72a1950ca0c4a",
    "val_mask_sha256": "2175e7133a0b272f26416cf46b11e08b771d899af50d222724036ed090d9acfb",
}


def write_new(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    base = json.loads(SOURCE.read_text(encoding="utf-8"))
    base.update({
        "dataset": "amazon",
        "dataset_file": "datasets/amazon",
        "expected": {"nodes": 11944, "training_edges": 8808728},
        "expected_hashes": HASHES,
        "batch_size": 256,
        "batch_size_source": "official_amazon_default",
        "uncovered_prefix_nodes": 3305,
    })
    for run_type, epochs in (("smoke", 5), ("diagnostic", 31)):
        config = dict(base)
        config.update({
            "seed": 0, "run_type": run_type, "max_epoch": epochs,
            "result_dir": f"results/experiments/caregnn/amazon/{PROTOCOL}/{run_type}/seed_0",
        })
        write_new(TARGET / f"amazon_{run_type}.json", config)
    for seed in range(10):
        config = dict(base)
        config.update({
            "seed": seed, "run_type": "formal", "max_epoch": 31,
            "result_dir": f"results/experiments/caregnn/amazon/{PROTOCOL}/formal/seed_{seed}",
        })
        write_new(TARGET / f"amazon_formal_seed_{seed}.json", config)


if __name__ == "__main__":
    main()
