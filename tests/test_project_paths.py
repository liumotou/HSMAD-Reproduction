from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch


class ProjectPathTests(unittest.TestCase):
    def test_default_root_is_repository_checkout(self) -> None:
        from methods.project_paths import project_root

        expected = Path(__file__).resolve().parents[1]
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(expected, project_root())

    def test_environment_override_selects_external_workspace(self) -> None:
        from methods.project_paths import project_root

        external = Path(__file__).resolve().parent / "external-root"
        with patch.dict(os.environ, {"HSMAD_ROOT": str(external)}, clear=True):
            self.assertEqual(external.resolve(), project_root())


if __name__ == "__main__":
    unittest.main()
