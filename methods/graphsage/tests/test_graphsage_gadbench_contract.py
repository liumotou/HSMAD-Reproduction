import unittest


class GraphSAGEGADBenchContractTest(unittest.TestCase):
    def test_pool_two_layer_model_exposes_logits_contract(self):
        from src.model import GraphSAGEGADBench

        model = GraphSAGEGADBench(
            in_feats=400,
            h_feats=64,
            num_classes=2,
            num_layers=2,
            agg="pool",
            dropout_rate=0,
            activation="ReLU",
        )

        self.assertEqual(len(model.layers), 2)
        self.assertEqual(model.layers[0]._aggre_type, "pool")
        self.assertEqual(model.layers[0]._out_feats, 64)
        self.assertEqual(model.layers[1]._out_feats, 64)
        self.assertEqual(model.output_linear.in_features, 64)
        self.assertEqual(model.output_linear.out_features, 2)


if __name__ == "__main__":
    unittest.main()
