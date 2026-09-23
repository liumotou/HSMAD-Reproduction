# GraphSAGE-GADBench-h64

This is an isolated candidate baseline: **HSMAD frozen data protocol + GADBench GraphSAGE training protocol**. It is not claimed to be author-exact HSMAD, byte-identical GADBench, or a paper-level reproduction.

Model source: GADBench commit `f9aa021ce9b6c6580427fb633b596843be76ddc6`, `models/gnn.py::GraphSAGE` (`6f81e05c4f924e8b8a047e7d052bee7b358b9473dee4b2ddfac3153eba53704d`).

The candidate uses two full-graph DGL `SAGEConv` layers, `pool` aggregation, ReLU, `h_feats=64`, then a 2-logit linear head. It uses the GADBench train-mask class weight `[1.0, normal_count/anomaly_count]`. Frozen HSMAD masks, graph preprocessing, seeds, validation threshold selection, test F1-Macro/AUROC, and result isolation are project protocol.
