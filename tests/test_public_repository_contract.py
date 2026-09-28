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

    def test_published_executable_code_has_no_machine_specific_paths(self) -> None:
        offenders: list[str] = []
        machine_specific = re.compile(
            r"/root/(?:autodl-tmp|miniconda3)(?:/|\b)|[A-Za-z]:\\"
        )
        executable_paths = [
            *ROOT.glob("methods/**/*.py"),
            *ROOT.glob("methods/**/*.sh"),
        ]
        for path in executable_paths:
            relative = path.relative_to(ROOT)
            text = path.read_text(encoding="utf-8", errors="replace")
            if machine_specific.search(text):
                offenders.append(relative.as_posix())
        self.assertEqual([], offenders, f"machine-specific paths in executable code: {offenders}")

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

    def test_nrgl_tests_use_package_qualified_imports(self) -> None:
        offenders: list[str] = []
        pattern = re.compile(r"^\s*from (src|audit)\.", re.MULTILINE)
        for path in (ROOT / "methods" / "nrgl" / "tests").glob("*.py"):
            if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual([], offenders, f"ambiguous NRGL test imports: {offenders}")

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
