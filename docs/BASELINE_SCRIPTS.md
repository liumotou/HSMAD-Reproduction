# Baseline code and run-script index

The code below is the project-adapted implementation used for audited HSMAD-data experiments. Unless explicitly stated otherwise, these runs are marked `candidate_protocol_not_author_exact`; they are not claimed to be byte-identical author reproductions.

| Method | Main implementation / runner | Configuration location | Status note |
|---|---|---|---|
| HSMAD | `main.py`, `model.py`, `manifold_update.py`, `dataset.py`, `utils.py` | command-line arguments in `main.py` | Project root implementation. |
| MLP | `methods/mlp/src/formal_runner.py` | `methods/mlp/configs/` | Feature-only candidate recovered from the experiment workspace. |
| GCN | `methods/gcn/src/run_kipf_v2.py` | `methods/gcn/configs/` | Kipf two-layer project candidate. |
| ChebNet | `methods/chebnet/src/run.py` | `methods/chebnet/configs/` | PyG candidate. |
| GIN | `methods/gin/src/run.py` | `methods/gin/configs/` | PyG candidate. |
| GWNN | `methods/gwnn/src/run.py` | `methods/gwnn/configs/` | Paper-formula candidate. |
| GAT (original topology diagnostic) | `methods/gat/src/run_smoke.py`, `run_diagnostic_full.py` | `methods/gat/configs/` | Diagnostic/archived candidate. |
| GAT-v2 (GADBench-adapted) | `methods/gat_v2_gadbench/src/run_*.py` | `methods/gat_v2_gadbench/configs/` | Residual GAT + FFN project candidate. |
| GraphSAGE | `methods/graphsage/src/run_smoke.py`, `run_formal.py` | `methods/graphsage/configs/` | GADBench pool GraphSAGE project candidate. |
| GraphConsis | `methods/graphconsis/src/run.py` | `methods/graphconsis/configs/` | Single-relation adapted candidate. |
| CARE-GNN | `methods/caregnn/src/run.py` | `methods/caregnn/configs/` | Single-relation adapted candidate. |
| PC-GNN | `methods/pcgnn/src/run.py` | `methods/pcgnn/configs/` | Single-relation adapted candidate. |
| BWGNN | `methods/bwgnn/src/run_smoke.py`, `run_formal.py` | `methods/bwgnn/configs/` | Project-adapted BWGNN runner. |
| SparseGAD | `methods/sparsegad/src/run_smoke.py`, `run_formal.py` | `methods/sparsegad/configs/` | Candidate protocol. |
| NRGL | `methods/nrgl/src/runner.py` | `methods/nrgl/configs/` | Candidate protocol. |
| SEC-GFD | `methods/sec_gfd/src/runner.py` | method source directory | Candidate protocol. |
| GHRN | `methods/ghrn/src/runner.py` | method source directory | Candidate protocol. |
| SVM | `methods/svm/src/runner.py` | `methods/svm/configs/` | Feature-only candidate. |
| CGADM | `methods/cgadm_hsmad/src/runner.py` | `methods/cgadm_hsmad/configs/` | HSMAD-data adapter candidate. |
| DSGAD | `methods/dsgad/src/runner.py` | method source directory | Candidate components/runner. |
| CurvGAD | `methods/curvgad_hsmad/src/` | method source directory | Precomputation/adaptation code; formal execution remained blocked by architecture/shape issues. |
| SpaceGNN | `methods/spacegnn_hsmad/src/runner.py` | generated candidate configuration | Adapted runner plus fixed official snapshot. |
| AMNet | `methods/amnet_hsmad/run_smoke.py`, `run_full.py` | `methods/amnet_hsmad/configs/` | Requires the isolated legacy CUDA 11/PyG environment. |
| PMP | `methods/pmp_hsmad/src/runner.py` | generated candidate configuration | Adapted frozen-mask candidate. |

## General execution notes

- Run commands should be launched from the repository root so relative paths resolve consistently.
- Each method's JSON configuration is the authoritative record for dataset, hidden size, optimizer, early stopping, checkpoint selection, and output path.
- Raw data and results are intentionally excluded from Git. Generated outputs belong under `results/experiments/<method>/<dataset>/<protocol>/`.
- Smoke and diagnostic artifacts must never be mixed into formal ten-seed summaries.
- Use module invocation from the repository root. If datasets/results live in another verified workspace, set `HSMAD_ROOT` only for that command.

## Exact invocation examples

```bash
export HSMAD_ROOT=/absolute/path/to/verified/workspace  # optional

python -m methods.mlp.src.train --dataset weibo --seed 0 --run-type smoke --config methods/mlp/configs/weibo_smoke.json
python -m methods.mlp.src.formal_runner --config methods/mlp/configs/weibo_formal.json --seed 0 --preflight-only
python -m methods.caregnn.src.run --config methods/caregnn/configs/weibo_smoke.json
python -m methods.chebnet.src.run --config methods/chebnet/configs/weibo_chebnet_h64_smoke.json
python -m methods.gin.src.run --config methods/gin/configs/weibo_gin_h64_smoke.json
python -m methods.gwnn.src.run --config methods/gwnn/configs/weibo_gwnn_paper_formula_h64_smoke.json
python -m methods.graphconsis.src.run --config methods/graphconsis/configs/weibo_graphconsis_single_relation_smoke.json
python -m methods.pcgnn.src.run --config methods/pcgnn/configs/weibo_smoke.json
python -m methods.spacegnn_hsmad.src.runner --dataset weibo --run-type smoke --seeds 0
python -m methods.pmp_hsmad.src.runner --dataset weibo --run-type smoke --seeds 0
python -m methods.amnet_hsmad.run_smoke --config methods/amnet_hsmad/configs/weibo_hsmad_candidate.json
```

AMNet must use its dedicated environment wrapper when the fixed PyG binary stack is required. Run the MLP smoke command first: it creates the fingerprint baseline consumed by `--preflight-only`. The preflight command then validates configuration, frozen inputs, and the pinned GADBench reference without starting formal training.
