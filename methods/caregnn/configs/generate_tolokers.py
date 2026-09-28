"""Generate CARE-GNN Tolokers configs from the frozen Weibo protocol."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/root/autodl-tmp/HSMAD")
SOURCE = ROOT / "methods/caregnn/configs/weibo_diagnostic.json"
TARGET = ROOT / "methods/caregnn/configs"
PROTOCOL = "caregnn_official_single_flattened_relation_h64_candidate"
HASHES = {
    "dataset_file_sha256": "1d400b4cdced54b25c394cc3d258921f443f04b2ae2e09bbb70fbc7073ed0b65",
    "feature_sha256": "d70a2b3d04774f11e8d8e94f659b493c7cb86eae0c22bdfbecd2e36fc0577809",
    "label_sha256": "808c63a620abae774a7614459d9bfe6a258ee3460f2e39ca0511c59dbf553518",
    "test_mask_sha256": "5fc331d7846b8f113c954295f8c9503209157d8a6393e5520de4ff029a6c1662",
    "train_mask_sha256": "4f298168b1a4d72eedf0f4594d1deb196ddb6e1c80ee95058dd8c05da1bb10af",
    "val_mask_sha256": "bf4c7797ce28dc6a47a51c2389ad56f86538c5fa852113de76a17eaaaba9bb36",
}


def write_new(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    base = json.loads(SOURCE.read_text(encoding="utf-8"))
    base.update({
        "dataset": "tolokers",
        "dataset_file": "datasets/tolokers",
        "expected": {"nodes": 11758, "training_edges": 1049758},
        "expected_hashes": HASHES,
        "batch_size_source": "project_choice_using_official_yelp_default_because_tolokers_is_unpublished",
    })
    for run_type, epochs in (("smoke", 5), ("diagnostic", 31)):
        config = dict(base)
        config.update({
            "seed": 0, "run_type": run_type, "max_epoch": epochs,
            "result_dir": f"results/experiments/caregnn/tolokers/{PROTOCOL}/{run_type}/seed_0",
        })
        write_new(TARGET / f"tolokers_{run_type}.json", config)
    for seed in range(10):
        config = dict(base)
        config.update({
            "seed": seed, "run_type": "formal", "max_epoch": 31,
            "result_dir": f"results/experiments/caregnn/tolokers/{PROTOCOL}/formal/seed_{seed}",
        })
        write_new(TARGET / f"tolokers_formal_seed_{seed}.json", config)


if __name__ == "__main__":
    main()
