import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA runtime")
class GraphSAGERunnerEnvironmentContractTest(unittest.TestCase):
    def test_framework_info_reads_free_memory_from_explicit_gpu_zero(self):
        from run_smoke import framework_info

        value = framework_info()

        self.assertEqual(value["gpu"], torch.cuda.get_device_name(0))
        self.assertGreater(value["gpu_free_mb_at_start"], 0)


if __name__ == "__main__":
    unittest.main()
