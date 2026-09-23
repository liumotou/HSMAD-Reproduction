import unittest
import sys
import tempfile
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

    def test_failed_preflight_metadata_directory_is_recoverable_but_result_directory_is_not(self):
        from run_smoke import recoverable_output_dir

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "config_snapshot.json").write_text("{}", encoding="utf-8")
            (output / "terminal_attempt_1_environment_error.log").write_text("error", encoding="utf-8")
            self.assertTrue(recoverable_output_dir(output))
            (output / "metrics.json").write_text("{}", encoding="utf-8")
            self.assertFalse(recoverable_output_dir(output))


if __name__ == "__main__":
    unittest.main()
