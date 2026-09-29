# Baseline code and run-script index

The code below is the project-adapted implementation used for audited HSMAD-data experiments. Unless explicitly stated otherwise, these runs are marked `candidate_protocol_not_author_exact`; they are not claimed to be byte-identical author reproductions.

## HSMAD

The main project method uses `main.py`, `model.py`, `manifold_update.py`, `dataset.py`, and `utils.py`. It is not part of the 17-baseline count.

## Table 1 baselines (17)

| Method | Main implementation / runner | Configuration location | Status note |
|---|---|---|---|
| MLP | `methods/mlp/src/formal_runner.py` | `methods/mlp/configs/` | Feature-only project candidate. |
| GCN | `methods/gcn/src/run_kipf_v2.py` | `methods/gcn/configs/` | Kipf two-layer project candidate. |
| GAT | `methods/gat_v2_gadbench/src/run_formal.py` | `methods/gat_v2_gadbench/configs/` | Primary candidate; `methods/gat/` is an earlier diagnostic implementation of the same baseline. |
| GraphSAGE | `methods/graphsage/src/run_formal.py` | `methods/graphsage/configs/` | GADBench pool GraphSAGE project candidate. |
| AMNet | `methods/amnet_hsmad/run_full.py` | `methods/amnet_hsmad/configs/` | Requires the isolated legacy CUDA 11/PyG environment. |
| BWGNN | `methods/bwgnn/src/run_formal.py` | `methods/bwgnn/configs/` | Project-adapted candidate. |
| GHRN | `methods/ghrn/src/runner.py` | method source directory | Project-adapted candidate. |
| SparseGAD | `methods/sparsegad/src/run_formal.py` | `methods/sparsegad/configs/` | Project-adapted candidate. |
| SEC-GFD | `methods/sec_gfd/src/runner.py` | method source directory | Selected-dataset candidate. |
| NRGL | `run(...)` in `methods/nrgl/src/runner.py` | `methods/nrgl/configs/` | Code/API is present, but this snapshot has no standalone CLI main. |
| PC-GNN | `methods/pcgnn/src/run.py` | `methods/pcgnn/configs/` | Single-relation adapted candidate. |
| ConsisGAD | Not included | — | No ConsisGAD implementation is currently present; GraphConsis is not a substitute. |
| PMP | `methods/pmp_hsmad/src/runner.py` | generated candidate configuration | Adapted frozen-mask candidate. |
| DSGAD | `methods/dsgad/src/runner.py` | method source directory | Project-adapted candidate. |
| CurvGAD | `methods/curvgad_hsmad/src/` | method source directory | Partial precomputation/adaptation code; no executable formal runner. |
| SpaceGNN | `methods/spacegnn_hsmad/src/runner.py` | generated candidate configuration | Project-adapted candidate. |
| CGADM | `methods/cgadm_hsmad/src/runner.py` | `methods/cgadm_hsmad/configs/` | HSMAD-data adapter candidate. |

## Supplementary methods

These six implementations are useful repository additions but are not among the 17 baselines in the paper's Table 1.

| Method | Main implementation / runner | Configuration location | Status note |
|---|---|---|---|
| ChebNet | `methods/chebnet/src/run.py` | `methods/chebnet/configs/` | Supplementary PyG candidate. |
| GIN | `methods/gin/src/run.py` | `methods/gin/configs/` | Supplementary PyG candidate. |
| GWNN | `methods/gwnn/src/run.py` | `methods/gwnn/configs/` | Supplementary paper-formula candidate. |
| SVM | `methods/svm/src/runner.py` | `methods/svm/configs/` | Supplementary feature-only candidate. |
| CARE-GNN | `methods/caregnn/src/run.py` | `methods/caregnn/configs/` | Supplementary single-relation candidate. |
| GraphConsis | `methods/graphconsis/src/run.py` | `methods/graphconsis/configs/` | Supplementary single-relation candidate; not the same method as ConsisGAD. |

## General execution notes

- Run commands should be launched from the repository root so relative paths resolve consistently.
- Each method's JSON configuration is the authoritative record for dataset, hidden size, optimizer, early stopping, checkpoint selection, and output path.
- Raw data and results are intentionally excluded from Git. Generated outputs belong under `results/experiments/<method>/<dataset>/<protocol>/`.
- Smoke and diagnostic artifacts must never be mixed into formal ten-seed summaries.
- Use module invocation from the repository root. If datasets/results live in another verified workspace, set `HSMAD_ROOT` only for that command.
- `HSMAD_ROOT` applies only to entry points that import `methods.project_paths`; it is not a compatibility switch for every historical runner.
- GraphSAGE formal execution requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED=0`. The ten-seed wrapper supplies both values.

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

For a single GraphSAGE formal seed, use:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
python -m methods.graphsage.src.run_formal \
  --config methods/graphsage/configs/weibo_graphsage_gadbench_h64_formal.json \
  --seed 0
```

AMNet must use its dedicated environment wrapper when the fixed PyG binary stack is required. Run the MLP smoke command first: it creates the fingerprint baseline consumed by `--preflight-only`. The preflight command then validates configuration, frozen inputs, and the pinned GADBench reference without starting formal training.
