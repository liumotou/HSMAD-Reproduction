# GCN baseline

Reference evidence: GADBench commit `f9aa021ce9b6c6580427fb633b596843be76ddc6`, `models/gnn.py::GCN` (SHA256 `6f81e05c4f924e8b8a047e7d052bee7b358b9473dee4b2ddfac3153eba53704d`). The original Kipf two-layer topology is separately audited in `audit/gcn_source_equivalence_audit.md`.

## Amazon protocol_v2_kipf_two_layer_hidden64

Model: `GraphConv(input_dim→64, ReLU, norm='both', bias=False) → GraphConv(64→2, activation=None, norm='both', bias=False)`. There is no MLP classifier, `GraphConv(64→64)`, second-layer ReLU, or class weighting.

Fixed evaluation metadata:

- `checkpoint_protocol=GADBench_style_AUPRC_best`: retain the validation-AUPRC-best checkpoint.
- `early_stop_protocol=validation_F1_macro`: patience is driven only by validation F1-Macro.
- `threshold_protocol=validation_F1_macro_grid_0.05_to_0.95`: choose the validation F1-Macro threshold from 0.05…0.95 at the retained checkpoint.

The test set is only used once for final F1-Macro and `softmax(logits)[:, 1]` AUROC. These are project execution-protocol choices; this result is not claimed as an author-byte-identical HSMAD checkpoint protocol. Data, frozen masks, graph preprocessing, seeds, logs, and summary formatting remain shared project controls.
