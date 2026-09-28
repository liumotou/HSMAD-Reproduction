import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'methods/gcn/src'))
from kipf_two_layer import KipfTwoLayerGCN


def test_kipf_topology_contract():
    model = KipfTwoLayerGCN(400, 64, 2, 0.0)
    assert len(model.layers) == 2
    assert model.layers[0]._in_feats == 400 and model.layers[0]._out_feats == 64
    assert model.layers[1]._in_feats == 64 and model.layers[1]._out_feats == 2
    assert model.layers[0].bias is None and model.layers[1].bias is None
    assert model.layers[0]._norm == 'both' and model.layers[1]._norm == 'both'
    assert model.layers[1]._activation is None
    assert not any(isinstance(module, torch.nn.Linear) for module in model.modules())


if __name__ == '__main__':
    test_kipf_topology_contract()
    print('PASS: Kipf two-layer topology contract')
