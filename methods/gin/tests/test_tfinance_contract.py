import hashlib
import json
import unittest
from pathlib import Path
from methods.project_paths import project_root

import dgl

from methods.gin.src.run import prepare_training_graph, sha256_edges, sha256_tensor


class GINTFinanceContractTest(unittest.TestCase):
    def test_smoke_config_matches_frozen_tfinance_input(self):
        root = project_root()
        config_path = root / 'methods/gin/configs/tfinance_gin_h64_smoke.json'
        self.assertTrue(config_path.exists(), 'T-Finance smoke config is not implemented')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        raw_path = root / config['dataset_file']
        raw = dgl.load_graphs(str(raw_path))[0][0]
        graph = prepare_training_graph(raw)
        self.assertEqual(config['dataset'], 'tfinance')
        self.assertEqual((graph.num_nodes(), graph.num_edges()), (39357, 42484443))
        self.assertEqual(tuple(graph.ndata['feature'].shape), (39357, 10))
        self.assertEqual(
            {name: int(graph.ndata[name].sum()) for name in ('train_mask', 'val_mask', 'test_mask')},
            {'train_mask': 15742, 'val_mask': 7792, 'test_mask': 15823},
        )
        self.assertEqual(
            hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            '051b27bd8d32086a2f39b629d84a1e900d47e75d5726c7dc2d6a8dd54de4db98',
        )
        expected_masks = {
            'train_mask': '88727debf7b9f9e9dba73095a8706dc175e830012220fb38b2ff39a2e21fe0f1',
            'val_mask': '654bddb549a4f0a72950c796b5feccd712656ba89d866484d0c585c932b5c7f2',
            'test_mask': '34234f45acfca50e465087b99634b5ef8bb2143f9edb8d7478ed837f2dfe1364',
        }
        for name, expected in expected_masks.items():
            self.assertEqual(sha256_tensor(graph.ndata[name]), expected)
        self.assertEqual(
            sha256_edges(graph),
            'ec2d12e833240b59f017ad644741536042f82d5d589d1b467391b1bfd8390a82',
        )


if __name__ == '__main__':
    unittest.main()
