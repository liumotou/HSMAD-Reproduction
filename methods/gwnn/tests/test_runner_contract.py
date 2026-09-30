import json
import tempfile
import unittest
from pathlib import Path


class GWNNRunnerContractTests(unittest.TestCase):
    def test_frozen_smoke_config_and_run_spec(self):
        from methods.gwnn.src.run import build_run_spec, load_config

        config_path = Path("methods/gwnn/configs/weibo_gwnn_paper_formula_h64_smoke.json")
        config = load_config(config_path)
        spec = build_run_spec(config)
        self.assertEqual(spec.dataset, "weibo")
        self.assertEqual(spec.seed, 0)
        self.assertEqual(spec.run_type, "smoke")
        self.assertEqual(spec.max_epoch, 5)
        self.assertEqual(config["hidden_dim"], 64)
        self.assertEqual(config["wavelet_scale"], 1.0)
        self.assertEqual(config["wavelet_threshold"], 1e-4)
        self.assertEqual(config["wavelet_inverse_semantics"], "paper_formula_exp_positive_sL")
        self.assertTrue(config["candidate_protocol_not_author_exact"])

    def test_existing_output_is_refused(self):
        from methods.gwnn.src.run import ensure_new_output

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(FileExistsError):
                ensure_new_output(path)


if __name__ == "__main__":
    unittest.main()
