"""Generate immutable Tolokers GIN diagnostic/formal configs from audited smoke."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from methods.project_paths import project_root

ROOT = project_root()
SOURCE = ROOT / "methods/gin/configs/tolokers_gin_h64_smoke.json"

def write_new(path, value):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--formal", action="store_true"); args = parser.parse_args()
    base = json.loads(SOURCE.read_text(encoding="utf-8")); base["max_epoch"] = 200
    diagnostic = dict(base, run_type="diagnostic", seed=0, result_dir="results/experiments/gin/tolokers/gin_h64_candidate/diagnostic/seed_0")
    diagnostic_path = ROOT / "methods/gin/configs/tolokers_gin_h64_diagnostic.json"
    if diagnostic_path.exists():
        if json.loads(diagnostic_path.read_text()) != diagnostic: raise RuntimeError("diagnostic mismatch")
    else: write_new(diagnostic_path, diagnostic)
    if args.formal:
        manifest = {}
        for seed in range(10):
            cfg = dict(diagnostic, run_type="formal", seed=seed, result_dir=f"results/experiments/gin/tolokers/gin_h64_candidate/formal/seed_{seed}")
            path = ROOT / f"methods/gin/configs/tolokers_gin_h64_formal_seed_{seed}.json"
            manifest[path.name] = write_new(path, cfg)
        write_new(ROOT / "methods/gin/configs/tolokers_gin_h64_formal_manifest.json", manifest)
    print(diagnostic_path)

if __name__ == "__main__": main()
