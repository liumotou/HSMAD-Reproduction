# Baseline code and run-script index

The code below is the project-adapted implementation used for audited HSMAD-data experiments. Unless explicitly stated otherwise, these runs are marked `candidate_protocol_not_author_exact`; they are not claimed to be byte-identical author reproductions.

| Method | Main implementation / runner | Configuration location | Status note |
|---|---|---|---|
| HSMAD | `main.py`, `model.py`, `manifold_update.py`, `dataset.py`, `utils.py` | command-line arguments in `main.py` | Project root implementation. |
| GCN | `methods/gcn/src/run_kipf_v2.py` | `methods/gcn/configs/` | Kipf two-layer project candidate. |
| GAT (original topology diagnostic) | `methods/gat/src/run_smoke.py`, `run_diagnostic_full.py` | `methods/gat/configs/` | Diagnostic/archived candidate. |
| GAT-v2 (GADBench-adapted) | `methods/gat_v2_gadbench/src/run_*.py` | `methods/gat_v2_gadbench/configs/` | Residual GAT + FFN project candidate. |
| GraphSAGE | `methods/graphsage/src/run_smoke.py`, `run_formal.py` | `methods/graphsage/configs/` | GADBench pool GraphSAGE project candidate. |
| BWGNN | `methods/bwgnn/src/run_smoke.py`, `run_formal.py` | `methods/bwgnn/configs/` | Project-adapted BWGNN runner. |
| SparseGAD | `methods/sparsegad/src/run_smoke.py`, `run_formal.py` | `methods/sparsegad/configs/` | Candidate protocol. |
| NRGL | `methods/nrgl/src/runner.py` | `methods/nrgl/configs/` | Candidate protocol. |
| SEC-GFD | `methods/sec_gfd/src/runner.py` | method source directory | Candidate protocol. |
| GHRN | `methods/ghrn/src/runner.py` | method source directory | Candidate protocol. |
| SVM | `methods/svm/src/runner.py` | `methods/svm/configs/` | Feature-only candidate. |
| CGADM | `methods/cgadm_hsmad/src/runner.py` | `methods/cgadm_hsmad/configs/` | HSMAD-data adapter candidate. |
| DSGAD | `methods/dsgad/src/runner.py` | method source directory | Candidate components/runner. |
| CurvGAD | `methods/curvgad_hsmad/src/` | method source directory | Precomputation/adaptation code; formal execution remained blocked by architecture/shape issues. |
| AMNet | `methods/amnet_hsmad/` | implementation preflight only | Environment/adaptation work is retained; no complete public runner in this snapshot. |
| PMP | `methods/pmp_hsmad/src/` | method source directory | Model/protocol/audit components; no complete standalone runner in this snapshot. |

## General execution notes

- Run commands should be launched from the repository root so relative paths resolve consistently.
- Each method's JSON configuration is the authoritative record for dataset, hidden size, optimizer, early stopping, checkpoint selection, and output path.
- Raw data and results are intentionally excluded from Git. Generated outputs belong under `results/experiments/<method>/<dataset>/<protocol>/`.
- Smoke and diagnostic artifacts must never be mixed into formal ten-seed summaries.
