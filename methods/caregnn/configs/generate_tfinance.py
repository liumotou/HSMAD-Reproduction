"""Generate CARE-GNN T-Finance configs from the frozen protocol."""
from __future__ import annotations

import json
from pathlib import Path
from methods.project_paths import project_root


ROOT = project_root()
SOURCE = ROOT / "methods/caregnn/configs/weibo_diagnostic.json"
TARGET = ROOT / "methods/caregnn/configs"
PROTOCOL = "caregnn_official_single_flattened_relation_h64_candidate"
HASHES = {
    "dataset_file_sha256": "051b27bd8d32086a2f39b629d84a1e900d47e75d5726c7dc2d6a8dd54de4db98",
    "feature_sha256": "b330e6183b8e7439fb2634b92fe14f9090089d20beed06b09ca48fdee9011c70",
    "label_sha256": "343637bcc2e93dfcf1dccde170c87bcf0c8abb70957764b4eeb6473b86dab2f2",
    "test_mask_sha256": "34234f45acfca50e465087b99634b5ef8bb2143f9edb8d7478ed837f2dfe1364",
    "train_mask_sha256": "88727debf7b9f9e9dba73095a8706dc175e830012220fb38b2ff39a2e21fe0f1",
    "val_mask_sha256": "654bddb549a4f0a72950c796b5feccd712656ba89d866484d0c585c932b5c7f2",
}


def write_new(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    base = json.loads(SOURCE.read_text(encoding="utf-8"))
    base.update({
        "dataset": "tfinance",
        "dataset_file": "datasets/tfinance",
        "expected": {"nodes": 39357, "training_edges": 42484443},
        "expected_hashes": HASHES,
        "batch_size": 1024,
        "batch_size_source": "project_choice_using_official_yelp_default_because_tfinance_is_unpublished",
    })
    for run_type, epochs in (("smoke", 5), ("diagnostic", 31)):
        config = dict(base)
        config.update({
            "seed": 0, "run_type": run_type, "max_epoch": epochs,
            "result_dir": f"results/experiments/caregnn/tfinance/{PROTOCOL}/{run_type}/seed_0",
        })
        write_new(TARGET / f"tfinance_{run_type}.json", config)
    for seed in range(10):
        config = dict(base)
        config.update({
            "seed": seed, "run_type": "formal", "max_epoch": 31,
            "result_dir": f"results/experiments/caregnn/tfinance/{PROTOCOL}/formal/seed_{seed}",
        })
        write_new(TARGET / f"tfinance_formal_seed_{seed}.json", config)


if __name__ == "__main__":
    main()
