import unittest
from pathlib import Path


class GraphConsisAmazonContractTest(unittest.TestCase):
    def test_frozen_amazon_config_masks_and_uncovered_prefix(self):
        from methods.graphconsis.src.run import ROOT, load_config, prepare_training_graph, sha256_file, sha256_tensor
        import dgl
        config=load_config(Path("methods/graphconsis/configs/amazon_single_relation_smoke.json"))
        self.assertEqual(config["expected"],{"nodes":11944,"training_edges":8808728})
        raw_path=ROOT/config["dataset_file"]; graph=prepare_training_graph(dgl.load_graphs(str(raw_path))[0][0])
        self.assertEqual(sha256_file(raw_path),"25f77cdd9f5991c575ce5efd385aec86e30123585e9b87c1da0bcb25d5f67872")
        expected={"train_mask":(3455,"fb95bd68eda65b33435b2214bd1ceff41dfa324b5fbed8bcfeb72a1950ca0c4a"),
                  "val_mask":(1710,"2175e7133a0b272f26416cf46b11e08b771d899af50d222724036ed090d9acfb"),
                  "test_mask":(3474,"bc42f2677dbf8357e949a30330b0236fbe6435c91583aa0232a249a20f95e9d1")}
        for name,(count,digest) in expected.items():
            mask=graph.ndata[name].bool(); self.assertEqual(int(mask.sum()),count); self.assertEqual(sha256_tensor(mask),digest); self.assertFalse(bool(mask[:3305].any()))


if __name__=="__main__": unittest.main()
