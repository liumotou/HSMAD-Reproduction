import hashlib
import json
import unittest
from pathlib import Path
from methods.project_paths import project_root

import dgl

from methods.gin.src.run import prepare_training_graph, sha256_edges, sha256_tensor


class GINAmazonContractTest(unittest.TestCase):
    def test_smoke_config_matches_frozen_amazon_input(self):
        root = project_root()
        config_path = root / 'methods/gin/configs/amazon_gin_h64_smoke.json'
        self.assertTrue(config_path.exists(), 'Amazon smoke config is not implemented')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        raw_path = root / config['dataset_file']
        raw = dgl.load_graphs(str(raw_path))[0][0]
        graph = prepare_training_graph(raw)
        self.assertEqual(config['dataset'], 'amazon')
        self.assertEqual((graph.num_nodes(), graph.num_edges()), (11944, 8808728))
        self.assertEqual(tuple(graph.ndata['feature'].shape), (11944, 25))
        counts = {
            name: int(graph.ndata[name].sum())
            for name in ('train_mask', 'val_mask', 'test_mask')
        }
        self.assertEqual(
            counts, {'train_mask': 3455, 'val_mask': 1710, 'test_mask': 3474}
        )
        for name in ('train_mask', 'val_mask', 'test_mask'):
            self.assertEqual(int(graph.ndata[name][:3305].sum()), 0)
        self.assertEqual(
            hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            '25f77cdd9f5991c575ce5efd385aec86e30123585e9b87c1da0bcb25d5f67872',
        )
        self.assertEqual(
            sha256_tensor(graph.ndata['train_mask']),
            'fb95bd68eda65b33435b2214bd1ceff41dfa324b5fbed8bcfeb72a1950ca0c4a',
        )
        self.assertEqual(
            sha256_tensor(graph.ndata['val_mask']),
            '2175e7133a0b272f26416cf46b11e08b771d899af50d222724036ed090d9acfb',
        )
        self.assertEqual(
            sha256_tensor(graph.ndata['test_mask']),
            'bc42f2677dbf8357e949a30330b0236fbe6435c91583aa0232a249a20f95e9d1',
        )
        self.assertEqual(
            sha256_edges(graph),
            'c37edec12527c4fdf74695be7d6b2cad57f65d23f86764c62a08ccddf5cff008',
        )


if __name__ == '__main__':
    unittest.main()
