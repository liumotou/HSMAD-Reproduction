import unittest

import torch

from methods.graphconsis.src.run import build_padded_adjacency
from methods.graphconsis.src.scalable_adjacency import build_padded_adjacency_stable_csr


class ScalableAdjacencyContractTest(unittest.TestCase):
    def test_stable_csr_is_exactly_legacy_for_unsorted_edges_and_isolated_node(self):
        source = torch.tensor([2, 0, 2, 1, 0, 3, 2, 1], dtype=torch.int64)
        destination = torch.tensor([0, 1, 3, 2, 2, 3, 1, 0], dtype=torch.int64)
        for seed in (0, 1, 9):
            expected = build_padded_adjacency(source, destination, 5, 7, seed)
            actual = build_padded_adjacency_stable_csr(
                source, destination, 5, 7, seed
            )
            self.assertTrue(torch.equal(actual, expected))

    def test_does_not_mutate_inputs(self):
        source = torch.tensor([1, 0, 1, 0], dtype=torch.int64)
        destination = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
        source_before = source.clone()
        destination_before = destination.clone()
        build_padded_adjacency_stable_csr(source, destination, 2, 4, 3)
        self.assertTrue(torch.equal(source, source_before))
        self.assertTrue(torch.equal(destination, destination_before))


if __name__ == "__main__":
    unittest.main()
