import unittest
from pathlib import Path


class GraphConsisTolokersContractTest(unittest.TestCase):
    def test_frozen_tolokers_config_and_masks(self):
        from methods.graphconsis.src.run import ROOT, load_config, prepare_training_graph, sha256_file, sha256_tensor
        import dgl
        config=load_config(Path("methods/graphconsis/configs/tolokers_single_relation_smoke.json"))
        self.assertEqual(config["expected"], {"nodes":11758,"training_edges":1049758})
        raw_path=ROOT/config["dataset_file"]; graph=prepare_training_graph(dgl.load_graphs(str(raw_path))[0][0])
        self.assertEqual(sha256_file(raw_path),"1d400b4cdced54b25c394cc3d258921f443f04b2ae2e09bbb70fbc7073ed0b65")
        expected={"train_mask":(4703,"4f298168b1a4d72eedf0f4594d1deb196ddb6e1c80ee95058dd8c05da1bb10af"),
                  "val_mask":(2328,"bf4c7797ce28dc6a47a51c2389ad56f86538c5fa852113de76a17eaaaba9bb36"),
                  "test_mask":(4727,"5fc331d7846b8f113c954295f8c9503209157d8a6393e5520de4ff029a6c1662")}
        for name,(count,digest) in expected.items():
            mask=graph.ndata[name].bool(); self.assertEqual(int(mask.sum()),count); self.assertEqual(sha256_tensor(mask),digest)


if __name__=="__main__": unittest.main()
