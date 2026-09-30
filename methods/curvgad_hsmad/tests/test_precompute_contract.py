import json
import tempfile
import unittest
from pathlib import Path

import dgl
import numpy as np
import torch


class CurvaturePrecomputeContract(unittest.TestCase):
    def _graph(self, label_value=0):
        graph = dgl.graph((torch.tensor([0, 1, 1, 2, 2, 0]), torch.tensor([1, 0, 2, 1, 0, 2])), num_nodes=3)
        graph.ndata["label"] = torch.tensor([label_value, 1, 0])
        graph.ndata["train_mask"] = torch.tensor([True, False, False])
        graph.ndata["val_mask"] = torch.tensor([False, True, False])
        graph.ndata["test_mask"] = torch.tensor([False, False, True])
        return graph

    def test_graph_hash_uses_topology_not_labels_or_masks(self):
        from methods.curvgad_hsmad.src.precompute import graph_topology_sha256

        left = self._graph(label_value=0)
        right = self._graph(label_value=1)
        right.ndata["train_mask"] = ~right.ndata["train_mask"]
        self.assertEqual(graph_topology_sha256(left), graph_topology_sha256(right))

    def test_cache_key_changes_with_topology_or_exact_parameters(self):
        from methods.curvgad_hsmad.src.precompute import cache_key, graph_topology_sha256

        graph = self._graph()
        graph_hash = graph_topology_sha256(graph)
        base = cache_key(graph_hash, method="exact_orc", alpha=0.5, implementation_sha256="abc")
        self.assertNotEqual(base, cache_key(graph_hash, method="exact_orc", alpha=0.4, implementation_sha256="abc"))
        self.assertNotEqual(base, cache_key(graph_hash, method="exact_orc", alpha=0.5, implementation_sha256="def"))

    def test_exact_precompute_is_cached_and_contains_no_label_material(self):
        from methods.curvgad_hsmad.src.precompute import compute_or_load_exact_curvature

        with tempfile.TemporaryDirectory() as directory:
            first = compute_or_load_exact_curvature(self._graph(), Path(directory), alpha=0.5)
            second = compute_or_load_exact_curvature(self._graph(label_value=1), Path(directory), alpha=0.5)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.matrix_sha256, second.matrix_sha256)
            self.assertEqual(first.matrix.shape, (3, 3))
            self.assertEqual(first.matrix.nnz, 3)
            metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["method"], "exact_orc")
            self.assertEqual(metadata["alpha"], 0.5)
            self.assertEqual(metadata["label_or_mask_access"], "none")
            self.assertNotIn("label_sha256", metadata)
            self.assertNotIn("mask_sha256", metadata)

    def test_approximate_method_is_rejected(self):
        from methods.curvgad_hsmad.src.precompute import compute_or_load_exact_curvature

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exact_orc"):
                compute_or_load_exact_curvature(self._graph(), Path(directory), method="accelerated")


if __name__ == "__main__":
    unittest.main()
