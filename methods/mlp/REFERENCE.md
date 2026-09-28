# MLP network-definition reference

## Frozen external reference

- Repository: GADBench, <https://github.com/squareroot3/GADBench>
- Remote URL: `https://github.com/squareroot3/GADBench.git`
- Fixed commit: `f9aa021ce9b6c6580427fb633b596843be76ddc6`
- Retrieved: 2026-08-13 (Asia/Shanghai)
- Reference file: `audit/mlp_reference/GADBench/models/gnn.py`
- Reference class: `models/gnn.py::MLP`
- Reference-file SHA256: `6f81e05c4f924e8b8a047e7d052bee7b358b9473dee4b2ddfac3153eba53704d`

## Provenance statement

- The network definition is referenced from GADBench's `MLP` class.
- The HSMAD official repository does not provide an MLP/baseline implementation.
- `methods/mlp/` is this project's independent, auditable equivalent implementation.
- It must not be described as “HSMAD official MLP code”.
- The HSMAD paper's unified settings are Adam, learning rate `0.01`, and Weibo `hidden_dim=64`.
- All other MLP formal hyperparameters are this project's frozen choices, to be proposed and approved separately before a formal run.

## Scope boundary

GADBench is used only as the frozen network-definition reference. The data source, frozen masks, threshold selection, training/result logging, and CSV handling are governed separately by this project's reproduction protocol; they are not claimed to originate from GADBench.
