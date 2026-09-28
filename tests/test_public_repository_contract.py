from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicRepositoryContractTests(unittest.TestCase):
    def test_recovered_method_entry_points_exist(self) -> None:
        required = (
            "methods/mlp/src/formal_runner.py",
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
            if forbidden_parts.intersection(relative.parts) or path.suffix.lower() in forbidden_suffixes:
                offenders.append(relative.as_posix())
        self.assertEqual([], offenders, f"generated/binary artifacts must not be published: {offenders}")


if __name__ == "__main__":
    unittest.main()
