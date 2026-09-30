from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicRepositoryContractTests(unittest.TestCase):
    TABLE1_BASELINES = (
        "MLP",
        "GCN",
        "GAT",
        "GraphSAGE",
        "AMNet",
        "BWGNN",
        "GHRN",
        "SparseGAD",
        "SEC-GFD",
        "NRGL",
        "PC-GNN",
        "ConsisGAD",
        "PMP",
        "DSGAD",
        "CurvGAD",
        "SpaceGNN",
        "CGADM",
    )
    SUPPLEMENTARY_METHODS = (
        "ChebNet",
        "GIN",
        "GWNN",
        "SVM",
        "CARE-GNN",
        "GraphConsis",
    )

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

    def test_tracked_markdown_local_links_resolve(self) -> None:
        missing: list[str] = []
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "*.md"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        for raw_relative in tracked:
            if not raw_relative:
                continue
            relative = Path(raw_relative.decode("utf-8"))
            if "official_snapshot" in relative.parts:
                continue
            path = ROOT / relative
            text = path.read_text(encoding="utf-8", errors="replace")
            for target in link_pattern.findall(text):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                local_target = target.split("#", 1)[0]
                if local_target and not (path.parent / local_target).exists():
                    missing.append(f"{relative.as_posix()} -> {target}")
        self.assertEqual([], missing, f"broken local Markdown links: {missing}")

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

    def test_documented_module_entry_points_use_package_qualified_local_imports(self) -> None:
        entry_points = (
            "methods/gcn/src/run_kipf_v2.py",
            "methods/gat_v2_gadbench/src/run_formal.py",
            "methods/gat_v2_gadbench/src/run_diagnostic_full.py",
            "methods/gat_v2_gadbench/src/run_smoke.py",
            "methods/gat_v2_gadbench/src/model.py",
            "methods/graphsage/src/run_formal.py",
            "methods/graphsage/src/run_smoke.py",
            "methods/svm/src/runner.py",
        )
        ambiguous = re.compile(
            r"^\s*from (?:contracts|kipf_two_layer|model|protocol|run_diagnostic_full|run_smoke|selection|utils) import ",
            re.MULTILINE,
        )
        offenders = [
            relative
            for relative in entry_points
            if ambiguous.search((ROOT / relative).read_text(encoding="utf-8", errors="replace"))
        ]
        self.assertEqual([], offenders, f"ambiguous imports break documented python -m entry points: {offenders}")

    def test_graphsage_formal_wrapper_supplies_required_determinism_environment(self) -> None:
        wrapper = (ROOT / "methods/graphsage/run_weibo_formal.sh").read_text(encoding="utf-8")
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", wrapper)
        self.assertIn("PYTHONHASHSEED=0", wrapper)

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

    def test_readmes_separate_table1_baselines_from_supplementary_methods(self) -> None:
        def table_methods(text: str, start: str, end: str) -> list[str]:
            section = text.split(start, 1)[1].split(end, 1)[0]
            methods: list[str] = []
            for line in section.splitlines():
                if not line.startswith("|") or line.startswith("|---"):
                    continue
                name = line.split("|", 2)[1].strip().replace("**", "")
                if name not in {"方法", "Method"}:
                    methods.append(name)
            return methods

        chinese = (ROOT / "README.md").read_text(encoding="utf-8")
        english = (ROOT / "README_EN.md").read_text(encoding="utf-8")
        index = (ROOT / "docs/BASELINE_SCRIPTS.md").read_text(encoding="utf-8")

        self.assertNotIn("> **复现定位**", chinese)
        self.assertEqual(
            list(self.TABLE1_BASELINES),
            table_methods(
                chinese,
                "### 6.1 Table 1 baseline（17 个）",
                "### 6.2 补充方法（不计入 Table 1 baseline）",
            ),
        )
        self.assertEqual(
            list(self.TABLE1_BASELINES),
            table_methods(
                english,
                "### 6.1 Table 1 baselines (17)",
                "### 6.2 Supplementary methods (not counted as Table 1 baselines)",
            ),
        )
        self.assertEqual(
            list(self.SUPPLEMENTARY_METHODS),
            table_methods(
                chinese,
                "### 6.2 补充方法（不计入 Table 1 baseline）",
                "详细入口见",
            ),
        )
        self.assertEqual(
            list(self.SUPPLEMENTARY_METHODS),
            table_methods(
                english,
                "### 6.2 Supplementary methods (not counted as Table 1 baselines)",
                "See [docs/BASELINE_SCRIPTS.md]",
            ),
        )
        self.assertEqual(
            list(self.TABLE1_BASELINES),
            table_methods(index, "## Table 1 baselines (17)", "## Supplementary methods"),
        )
        self.assertEqual(
            list(self.SUPPLEMENTARY_METHODS),
            table_methods(index, "## Supplementary methods", "## General execution notes"),
        )
        self.assertIn("ConsisGAD", chinese)
        self.assertIn("尚未收录", chinese)

    def test_known_batch_import_duplicates_are_removed(self) -> None:
        removed_duplicates = (
            "methods/cgadm/official_snapshot/diffuse.py",
            "methods/gat/run_smoke.py",
            "methods/gat_v2_gadbench/run_amazon_diagnostic.py",
            "methods/gcn/configs/preflight_protocol_v2.py",
            "methods/bwgnn/src/test_runner_contract.py",
            "methods/graphconsis/configs/weibo_single_relation_smoke.json",
        )
        remaining = [relative for relative in removed_duplicates if (ROOT / relative).exists()]
        self.assertEqual([], remaining, f"redundant imported copies remain: {remaining}")

        curvgad_readme = (
            ROOT / "methods/curvgad/official_snapshot/README.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Intentional vendored overlap", curvgad_readme)
        self.assertIn("audit/mlp_reference/GADBench/models/gnn.py", curvgad_readme)

    def test_project_plan_uses_neutral_docs_directory(self) -> None:
        plan = ROOT / "docs/plans/2026-09-28-repository-portability-fixes.md"
        self.assertTrue(plan.is_file())
        self.assertFalse((ROOT / "docs/superpowers").exists())

    def test_third_party_notices_cover_retained_source_snapshots(self) -> None:
        notice_path = ROOT / "THIRD_PARTY_NOTICES.md"
        self.assertTrue(notice_path.is_file())
        notice = notice_path.read_text(encoding="utf-8")
        retained = [
            path.relative_to(ROOT).as_posix()
            for pattern in ("methods/**/official_snapshot", "methods/**/official_source")
            for path in ROOT.glob(pattern)
            if path.is_dir()
        ]
        retained.extend(
            (
                "audit/mlp_reference/GADBench",
                "methods/gat/audit/reference/PetarV-GAT",
            )
        )
        missing = sorted(relative for relative in retained if f"`{relative}`" not in notice)
        self.assertEqual([], missing, f"vendored source missing from THIRD_PARTY_NOTICES: {missing}")
        self.assertIn("upstream license", notice.lower())
        self.assertIn("redistribution", notice.lower())
        self.assertIn("MIT", notice)
        bundled_licenses = (
            "methods/curvgad/official_snapshot/LICENSE",
            "methods/gat/audit/reference/PetarV-GAT/LICENSE",
        )
        for relative in bundled_licenses:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_minimal_ci_runs_contract_tests_and_static_validator(self) -> None:
        workflow_path = ROOT / ".github/workflows/repository-validation.yml"
        self.assertTrue(workflow_path.is_file())
        workflow = workflow_path.read_text(encoding="utf-8")
        self.assertRegex(workflow, r"(?m)^\s*push:\s*$")
        self.assertRegex(workflow, r"(?m)^\s*pull_request:\s*$")
        self.assertIn("python -m pytest tests/ -q", workflow)
        self.assertIn("python scripts/validate_repository.py", workflow)


if __name__ == "__main__":
    unittest.main()
