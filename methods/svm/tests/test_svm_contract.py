import sys
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT.parents[1]))


class SVMContractTests(unittest.TestCase):
    def test_metric_comparison_requires_exact_core_values(self):
        from methods.svm.audit.recompute import compare_metrics

        original = {"f1_macro": 0.8, "auroc": 0.9, "threshold": 0.5, "predicted_anomaly_count": 3}
        self.assertEqual(compare_metrics(original, dict(original))["status"], "recompute_match")
        changed = dict(original, auroc=0.89)
        self.assertEqual(compare_metrics(original, changed)["status"], "recompute_mismatch")

    def test_formal_summary_uses_sample_standard_deviation(self):
        from methods.svm.audit.summarize_formal import summarize_rows

        rows = [
            {"status": "OK", "audit_status": "recompute_match", "f1_macro": 0.8, "auroc": 0.9},
            {"status": "OK", "audit_status": "recompute_match", "f1_macro": 1.0, "auroc": 0.7},
        ]
        summary = summarize_rows(rows, "weibo")
        self.assertEqual(summary["n_formal_ok"], 2)
        self.assertAlmostEqual(summary["f1_macro_mean"], 0.9)
        self.assertAlmostEqual(summary["f1_macro_std_sample"], 2 ** -0.5 / 5)

    def test_artifact_writer_creates_isolated_run_directory(self):
        from runner import write_run_artifacts

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "seed_0"
            write_run_artifacts(output, {"seed": 0}, {"nodes": 4}, {"status": "smoke"})
            self.assertEqual(json.loads((output / "config_snapshot.json").read_text())["seed"], 0)
            self.assertEqual(json.loads((output / "preflight.json").read_text())["nodes"], 4)
            self.assertEqual(json.loads((output / "metrics.json").read_text())["status"], "smoke")
            with self.assertRaises(FileExistsError):
                write_run_artifacts(output, {}, {}, {})

    def test_preflight_records_persisted_mask_counts(self):
        from types import SimpleNamespace
        import numpy as np
        from runner import build_preflight

        graph = SimpleNamespace(
            ndata={
                "feature": np.zeros((3, 2), dtype=np.float32),
                "label": np.array([0, 1, 0]),
                "train_mask": np.array([True, False, False]),
                "val_mask": np.array([False, True, False]),
                "test_mask": np.array([False, False, True]),
            },
            num_nodes=lambda: 3,
            num_edges=lambda: 4,
        )
        preflight = build_preflight(graph)
        self.assertEqual(preflight["nodes"], 3)
        self.assertEqual(preflight["stored_edges"], 4)
        self.assertEqual(preflight["mask_counts"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(preflight["edge_access"], "none")

    def test_fit_helper_returns_the_fitted_feature_only_protocol(self):
        import numpy as np
        from runner import fit_with_protocol

        x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1, 1, 0])
        masks = {
            "train": np.array([True, True, False, False]),
            "val": np.array([False, False, True, False]),
            "test": np.array([False, False, False, True]),
        }
        result, protocol = fit_with_protocol(x, y, masks, seed=0)
        self.assertEqual(result["edge_access"], "none")
        self.assertEqual(protocol.model.predict_proba(x[:1]).shape, (1, 2))

    def test_embedded_frozen_masks_are_read_without_regeneration(self):
        """A persisted DGL graph is the authoritative frozen-split source."""
        from types import SimpleNamespace
        import numpy as np
        from runner import frozen_masks_from_graph

        graph = SimpleNamespace(ndata={
            "train_mask": np.array([True, False, False]),
            "val_mask": np.array([False, True, False]),
            "test_mask": np.array([False, False, True]),
        })
        masks = frozen_masks_from_graph(graph)
        self.assertEqual(masks["train"].tolist(), [True, False, False])
        self.assertEqual(masks["val"].tolist(), [False, True, False])
        self.assertEqual(masks["test"].tolist(), [False, False, True])

    def test_feature_only_protocol_and_masks(self):
        from protocol import FeatureSVMProtocol
        import numpy as np

        x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1, 1, 0])
        masks = {
            "train": np.array([True, True, False, False]),
            "val": np.array([False, False, True, False]),
            "test": np.array([False, False, False, True]),
        }
        p = FeatureSVMProtocol(random_state=0)
        result = p.fit_evaluate(x, y, masks)
        self.assertEqual(result["edge_access"], "none")
        self.assertEqual(result["train_count"], 2)
        self.assertIn("threshold", result)
        self.assertIn("auroc", result)

    def test_test_metrics_are_not_used_for_threshold(self):
        from protocol import choose_threshold
        import numpy as np

        threshold = choose_threshold(np.array([0.1, 0.8]), np.array([0, 1]))
        self.assertIn(threshold, [0.05 * i for i in range(1, 20)])


if __name__ == "__main__":
    unittest.main()
