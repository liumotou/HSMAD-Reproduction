from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicRepositoryContractTests(unittest.TestCase):
    def test_recovered_method_entry_points_exist(self) -> None:
        required = (
            "methods/mlp/src/formal_runner.py",
            "audit/mlp_reference/GADBench/models/gnn.py",
            "methods/gcn/src/kipf_two_layer.py",
            "methods/caregnn/src/run.py",
            "methods/chebnet/src/run.py",
            "methods/gin/src/run.py",
            "methods/graphconsis/src/run.py",
            "methods/pcgnn/src/run.py",
            "methods/spacegnn_hsmad/src/runner.py",
            "methods/amnet_hsmad/run_smoke.py",
            "methods/pmp_hsmad/src/runner.py",
            "methods/sparsegad/audit/recompute_run.py",
            "methods/nrgl/audit/recompute_checkpoint.py",
        )
        missing = [relative for relative in required if not (ROOT / relative).is_file()]
        self.assertEqual([], missing, f"missing published entry points: {missing}")

    def test_published_python_does_not_pin_original_server_workspace(self) -> None:
        offenders: list[str] = []
        primary_entry_points = (
            "methods/bwgnn/src/run_formal.py",
            "methods/bwgnn/src/run_smoke.py",
            "methods/caregnn/src/run.py",
            "methods/chebnet/src/run.py",
            "methods/dsgad/src/runner.py",
            "methods/ghrn/src/protocol.py",
            "methods/gin/src/run.py",
            "methods/graphconsis/src/run.py",
            "methods/gwnn/src/run.py",
            "methods/nrgl/src/data.py",
            "methods/pcgnn/src/run.py",
            "methods/pmp_hsmad/src/runner.py",
            "methods/sec_gfd/src/runner.py",
            "methods/spacegnn_hsmad/src/runner.py",
            "methods/sparsegad/src/run_formal.py",
            "methods/sparsegad/src/run_smoke.py",
        )
        for relative_text in primary_entry_points:
            path = ROOT / relative_text
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"/root/autodl-tmp/HSMAD", text):
                offenders.append(relative.as_posix())
        self.assertEqual([], offenders, f"server-specific absolute roots: {offenders}")

    def test_no_publishable_large_or_generated_artifacts(self) -> None:
        forbidden_parts = {"__pycache__", ".pytest_cache", ".venv", ".venvs", "cache", "results"}
        forbidden_suffixes = {".pt", ".pth", ".ckpt", ".npz", ".npy", ".whl", ".pyc"}
        offenders: list[str] = []
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "methods"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        for raw_relative in tracked:
            if not raw_relative:
                continue
            relative = Path(raw_relative.decode("utf-8"))
            path = ROOT / relative
            if not path.exists():
                continue
            is_backup = ".before_" in path.name
            if forbidden_parts.intersection(relative.parts) or path.suffix.lower() in forbidden_suffixes or is_backup:
                offenders.append(relative.as_posix())
        self.assertEqual([], offenders, f"generated/binary artifacts must not be published: {offenders}")

    def test_mlp_sources_use_package_qualified_imports(self) -> None:
        offenders: list[str] = []
        pattern = re.compile(r"^from (model|train|utils) import ", re.MULTILINE)
        for path in (ROOT / "methods" / "mlp" / "src").glob("*.py"):
            if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual([], offenders, f"ambiguous MLP imports: {offenders}")

    def test_mlp_formal_runner_does_not_require_nested_git_metadata(self) -> None:
        path = ROOT / "methods" / "mlp" / "src" / "formal_runner.py"
        text = path.read_text(encoding="utf-8", errors="replace")
        self.assertNotIn("rev-parse", text)
        self.assertNotIn("ref_file.parents[1]", text)

    def test_mlp_frozen_code_hashes_match_published_sources(self) -> None:
        config = json.loads((ROOT / "methods/mlp/configs/weibo_formal.json").read_text(encoding="utf-8"))
        actual = {
            relative: hashlib.sha256(
                (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for relative in config["frozen_code_sha256"]
        }
        self.assertEqual(config["frozen_code_sha256"], actual)


if __name__ == "__main__":
    unittest.main()
