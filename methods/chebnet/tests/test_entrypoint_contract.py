import subprocess
import sys
import unittest
from pathlib import Path


class ChebNetEntrypointContractTest(unittest.TestCase):
    def test_runner_script_can_resolve_project_package(self):
        root = Path('/root/autodl-tmp/HSMAD')
        result = subprocess.run([sys.executable, str(root / 'methods/chebnet/src/run.py'), '--help'], cwd=root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--config', result.stdout)


if __name__ == '__main__':
    unittest.main()
