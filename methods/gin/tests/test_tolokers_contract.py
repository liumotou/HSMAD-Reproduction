import hashlib
import json
import unittest
from pathlib import Path
from methods.project_paths import project_root

import dgl

from methods.gin.src.run import prepare_training_graph, sha256_edges, sha256_tensor


class GINTolokersContractTest(unittest.TestCase):
    def test_smoke_config_matches_frozen_tolokers_input(self):
        root = project_root()
        config_path = root / 'methods/gin/configs/tolokers_gin_h64_smoke.json'
        self.assertTrue(config_path.exists(), 'Tolokers smoke config is not implemented')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        raw_path = root / config['dataset_file']
        raw = dgl.load_graphs(str(raw_path))[0][0]
        graph = prepare_training_graph(raw)
        self.assertEqual(config['dataset'], 'tolokers')
        self.assertEqual((graph.num_nodes(), graph.num_edges()), (11758, 1049758))
        self.assertEqual(tuple(graph.ndata['feature'].shape), (11758, 10))
        self.assertEqual(
            {name: int(graph.ndata[name].sum()) for name in ('train_mask', 'val_mask', 'test_mask')},
            {'train_mask': 4703, 'val_mask': 2328, 'test_mask': 4727},
        )
        self.assertEqual(
            hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            '1d400b4cdced54b25c394cc3d258921f443f04b2ae2e09bbb70fbc7073ed0b65',
        )
        self.assertEqual(
            sha256_tensor(graph.ndata['train_mask']),
            '4f298168b1a4d72eedf0f4594d1deb196ddb6e1c80ee95058dd8c05da1bb10af',
        )
        self.assertEqual(
            sha256_tensor(graph.ndata['val_mask']),
            'bf4c7797ce28dc6a47a51c2389ad56f86538c5fa852113de76a17eaaaba9bb36',
        )
        self.assertEqual(
            sha256_tensor(graph.ndata['test_mask']),
            '5fc331d7846b8f113c954295f8c9503209157d8a6393e5520de4ff029a6c1662',
        )
        self.assertEqual(
            sha256_edges(graph),
            'cd31edb3f4895e21e7e55f5938bb23109906306965fa095252d25605add69365',
        )


if __name__ == '__main__':
    unittest.main()
