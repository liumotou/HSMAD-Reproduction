# HSMAD Unified Reproduction Workspace

[中文](README.md) | [English](README_EN.md)

This repository publishes HSMAD, project-adapted implementations of the paper's Table 1 baselines, and the runners needed for shared splits, evaluation, and audit.

## Repository scope and reproduction policy

- `HSMAD` is the main method. The paper's Table 1 contains **17 baselines**, for 18 in-table methods in total.
- Unless stated otherwise, implementations under `methods/` use the `candidate_protocol_not_author_exact` label: they are auditable project adaptations, not claims of byte-identical author code or exact reproduction of every reported decimal.
- ChebNet, GIN, GWNN, SVM, CARE-GNN, and GraphConsis are supplementary methods retained in this repository and are not counted among the 17 Table 1 baselines.
- Dataset binaries, frozen masks, checkpoints, logs, and formal result artifacts are intentionally excluded from Git. The repository publishes sources, configurations, provenance links, and usage instructions.

## 1. Objective

The project compares HSMAD and its baselines with:

- one fixed split per dataset/method comparison, unaffected by training seed;
- training seeds `0..9` without regenerating splits;
- train-only loss and parameter updates;
- validation-only early stopping, checkpoint selection, and F1 threshold selection;
- test-only final F1-Macro and AUROC after model and threshold are fixed;
- per-run config, log, history, checkpoint, metrics, time, memory, and SHA256 records;
- sample standard deviation (`ddof=1`) for ten-seed summaries.

### Split implementation boundary

- The HSMAD entry point currently rebuilds an approximately 40/20/40 split deterministically in `dataset.py` with `random_state=2`, then overwrites graph masks. This is a fixed reproducible split, not direct consumption of persisted masks.
- Project-adapted baseline runners normally consume `train_mask`, `val_mask`, and `test_mask` from the graph and, where frozen hashes are configured, verify their SHA256 values.
- Before a formal comparison, verify that the HSMAD-generated split and the target baseline split are identical. Equal seeds alone do not prove equal masks.

## 2. Layout

```text
.
├── main.py, model.py, dataset.py, manifold_update.py, utils.py  # HSMAD
├── methods/                                                     # baseline code/config/tests
├── docs/DATASETS.md                                             # dataset links and provenance
├── docs/BASELINE_SCRIPTS.md                                     # runner index
├── scripts/validate_repository.py
└── datasets/                                                     # local only, ignored by Git
```

## 3. Datasets

See [docs/DATASETS.md](docs/DATASETS.md) for complete sources. The primary unified source is:

- GADBench: <https://github.com/squareroot3/GADBench>
- GADBench dataset archive: <https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing>

Place the DGL graph files at:

```text
datasets/weibo
datasets/amazon
datasets/yelp
datasets/tolokers
datasets/tfinance
datasets/tsocial
```

Each graph is expected to expose `feature`, `label`, `train_mask`, `val_mask`, and `test_mask` in `graph.ndata`.

Those masks are the default baseline-runner contract. HSMAD's `dataset.py` rebuilds and overwrites them with the deterministic `random_state=2` split; compare counts and SHA256 before claiming node-for-node split equivalence.

### A source URL is not proof of a runnable experiment

Four separate levels must be distinguished:

1. **Source verified:** the URL comes from a paper, official repository, or maintained benchmark;
2. **Download verified:** the complete file size and SHA256 are checked;
3. **Format verified:** the graph is readable and fields/shapes match the contract;
4. **Execution verified:** dependencies, CUDA, memory, paths, and protocol pass smoke/diagnostic/formal runs.

Always validate downloaded files and run an isolated smoke test before formal training.

### Special rules

- **Amazon:** the first 3305 uncovered nodes remain in the transductive graph for message passing but are excluded from all masks, loss, and metrics.
- **Tolokers:** use the frozen HSMAD masks instead of the built-in 50/25/25 masks.
- **Graph methods:** follow the preprocessing declared by the protocol; most candidates use `to_bidirected -> remove_self_loop -> add_self_loop`.
- **Feature-only methods:** MLP/SVM-style baselines must not read edges.

## 4. Yelp status

Yelp provenance and experimental coverage are separate matters:

1. **Provenance was verified.** Feature and label SHA256 matched the official GADBench Google Drive candidate. The official candidate contained exactly 45,954 additional edges—one self-loop per node. The local graph is explainable as the official graph after `remove_self_loop()`.
2. **Not every baseline completed Yelp.** Later baseline work deliberately froze Yelp. A missing Yelp result therefore does not imply an invalid dataset or an algorithmic inability to process Yelp.

Therefore the local Yelp graph is treated as a verified official GADBench preprocessing variant, while per-baseline Yelp completion must still be reported separately.

## 5. Why are some methods available on only one or two datasets?

The repository preserves actual progress rather than filling an artificial matrix. Common causes are:

- limited dataset/format scope in the upstream author repository;
- incomplete dataset adapters, runners, or checkpoint-recompute audits;
- full-graph memory requirements on large datasets;
- deliberate protocol freezes;
- dependency, ABI, preprocessing, tensor-shape, or determinism blockers;
- smoke/diagnostic completion without an audited ten-seed formal run.

Missing cells mean `INCOMPLETE` or `BLOCKED`; they do not mean low-scoring seeds were removed.

## 6. Method code coverage

The HSMAD entry point is `main.py`; it is not counted among the 17 baselines below. The list follows the paper's Table 1 exactly so supplementary repository experiments do not inflate the baseline count.

### 6.1 Table 1 baselines (17)

| Method | Code status | Main entry point | Note |
|---|---|---|---|
| MLP | Entry provided | `methods/mlp/src/formal_runner.py` | Feature-only project candidate |
| GCN | Entry provided | `methods/gcn/src/run_kipf_v2.py` | Kipf two-layer project candidate |
| GAT | Entry provided | `methods/gat_v2_gadbench/src/run_formal.py` | Current primary candidate; `methods/gat/` retains an earlier diagnostic implementation, but both count as one baseline |
| GraphSAGE | Entry provided | `methods/graphsage/src/run_formal.py` | GADBench pool project candidate |
| AMNet | Entry provided | `methods/amnet_hsmad/run_full.py` | Requires an isolated legacy CUDA 11/PyG environment |
| BWGNN | Entry provided | `methods/bwgnn/src/run_formal.py` | Project-adapted candidate |
| GHRN | Entry provided | `methods/ghrn/src/runner.py` | Project-adapted candidate |
| SparseGAD | Entry provided | `methods/sparsegad/src/run_formal.py` | Project-adapted candidate |
| SEC-GFD | Entry provided | `methods/sec_gfd/src/runner.py` | Selected-dataset candidate |
| NRGL | Code/API provided | `run(...)` in `methods/nrgl/src/runner.py` | No standalone CLI main in this snapshot |
| PC-GNN | Entry provided | `methods/pcgnn/src/run.py` | Single-relation adapted candidate |
| ConsisGAD | Not included | — | No ConsisGAD implementation is currently present; GraphConsis is not a substitute |
| PMP | Entry provided | `methods/pmp_hsmad/src/runner.py` | Frozen HSMAD protocol candidate |
| DSGAD | Entry provided | `methods/dsgad/src/runner.py` | Project-adapted candidate |
| CurvGAD | Partial code | `methods/curvgad_hsmad/src/` | Precomputation/adaptation code is retained, but no executable formal runner is available |
| SpaceGNN | Entry provided | `methods/spacegnn_hsmad/src/runner.py` | Project-adapted candidate |
| CGADM | Entry provided | `methods/cgadm_hsmad/src/runner.py` | HSMAD-data adapter candidate |

### 6.2 Supplementary methods (not counted as Table 1 baselines)

| Method | Main entry point | Note |
|---|---|---|
| ChebNet | `methods/chebnet/src/run.py` | Supplementary PyG candidate |
| GIN | `methods/gin/src/run.py` | Supplementary PyG candidate |
| GWNN | `methods/gwnn/src/run.py` | Supplementary paper-formula candidate |
| SVM | `methods/svm/src/runner.py` | Supplementary feature-only candidate |
| CARE-GNN | `methods/caregnn/src/run.py` | Supplementary single-relation candidate |
| GraphConsis | `methods/graphconsis/src/run.py` | Supplementary single-relation candidate; not the same method as Table 1's ConsisGAD |

See [docs/BASELINE_SCRIPTS.md](docs/BASELINE_SCRIPTS.md) for the detailed index.

## 7. Environment and validation

The repository intentionally has no umbrella root `requirements.txt`: the historical methods require mutually incompatible graph-library stacks. Most DGL candidates require Python 3.10, PyTorch, a compatible DGL build, NumPy, SciPy, pandas, scikit-learn, and SymPy. AMNet uses a separate legacy PyTorch/PyG environment and should not overwrite the main DGL environment. See [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) for recorded versions, isolated-environment templates, and verification steps.

Check the repository snapshot:

```bash
python scripts/validate_repository.py
```

Check the framework:

```bash
python - <<'PY'
import torch, dgl
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('dgl:', dgl.__version__)
PY
```

Check one dataset:

```bash
python - <<'PY'
import dgl
g = dgl.load_graphs('datasets/weibo')[0][0]
print(g.num_nodes(), g.num_edges())
print(tuple(g.ndata['feature'].shape), tuple(g.ndata['label'].shape))
for key in ('train_mask', 'val_mask', 'test_mask'):
    print(key, int(g.ndata[key].bool().sum()))
PY
```

Static validation does not prove CUDA/DGL runtime compatibility or full training success.

## 8. Running the code

Run commands from the repository root. Recommended entry points use repository-relative paths and contain no developer-machine absolute path.

Some newer adapted runners resolve the workspace through `methods.project_paths.project_root()`. For those runners only, datasets and results may be redirected to another verified workspace for one command:

```bash
export HSMAD_ROOT=/absolute/path/to/verified/workspace
```

`HSMAD_ROOT` is not a universal feature of every historical runner. HSMAD itself and runners that still derive `ROOT` from `Path(__file__)` expect data under the current checkout's `datasets/`. Check whether an entry point imports `methods.project_paths` before relying on the override.

### HSMAD, ten seeds

```bash
python main.py --dataset weibo --run 10 --epoch 1000 --patience 100 --hid_dim 64 --order 2 --q 0.5
```

### GCN, one seed

```bash
python -m methods.gcn.src.run_kipf_v2 \
  --config methods/gcn/configs/protocol_v2_weibo_formal.json --seed 0
```

### GAT-v2, one seed

```bash
python -m methods.gat_v2_gadbench.src.run_formal \
  --config methods/gat_v2_gadbench/configs/weibo_protocol_v2_gadbench_hidden64_formal.json --seed 0
```

### GraphSAGE, one seed

GraphSAGE formal execution requires deterministic process variables:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
python -m methods.graphsage.src.run_formal \
  --config methods/graphsage/configs/weibo_graphsage_gadbench_h64_formal.json --seed 0
```

### BWGNN, ten seeds

`--seeds` is comma-separated:

```bash
python -m methods.bwgnn.src.run_formal \
  --config methods/bwgnn/configs/weibo_bwg_h64_smoke.json \
  --seeds 0,1,2,3,4,5,6,7,8,9 --run-type formal
```

The historical config filename contains `smoke`; the formal runner overrides execution controls with `max_epoch=200` and `patience=50`. Verify the generated config snapshot before training.

### SparseGAD diagnostic

```bash
python -m methods.sparsegad.src.run_formal \
  --config methods/sparsegad/configs/weibo_sparsegad_h64_candidate.json \
  --run-type diagnostic --seeds 0
```

### SVM diagnostic

```bash
python -m methods.svm.src.runner --dataset weibo --seed 0 --run-type diagnostic
```

### SEC-GFD smoke

```bash
python -m methods.sec_gfd.src.runner --dataset weibo --run-type smoke --seeds 0
```

### GHRN smoke

```bash
python -m methods.ghrn.src.runner --dataset weibo --run-type smoke --seeds 0
```

## 9. Recommended execution sequence

For each method/dataset pair:

1. Run repository validation;
2. verify graph fields, shapes, mask counts, and SHA256;
3. run an isolated five-epoch smoke;
4. reload its checkpoint and independently reproduce metrics;
5. run a fresh seed-0 full diagnostic;
6. after audit success, run fresh formal seeds `0..9`;
7. summarize only `formal/OK` records with sample standard deviation (`ddof=1`).

Do not mix smoke/diagnostic artifacts into formal summaries, remove low-scoring seeds, or use test metrics to choose checkpoints, thresholds, or hyperparameters.

## 10. Limitations

- Dataset binaries, frozen masks, checkpoints, logs, and historical results are not published in Git;
- some configs are frozen experiment snapshots and must be reviewed before reuse;
- static checks cannot detect CUDA/DGL nondeterminism, OOM, ABI mismatches, or semantic differences from upstream code;
- Actual server sources for MLP, CAREGNN, ChebNet, GIN, GWNN, GraphConsis, PC-GNN, SpaceGNN, AMNet, and PMP are included. Source availability still does not imply that every dataset completed smoke, diagnostic, and audited ten-seed execution;
- only candidates passing checkpoint recomputation, mask isolation, and leakage audits should enter project comparison tables.

## 11. Upstream sources and local provenance

- GADBench: <https://github.com/squareroot3/GADBench>
- BWGNN: <https://github.com/squareroot3/Rethinking-Anomaly-Detection>
- GAT: <https://github.com/PetarV-/GAT>
- GraphSAGE: <https://github.com/williamleif/GraphSAGE>
- AMNet: <https://github.com/Illyasville/AMNet>
- SparseGAD: <https://github.com/KellyGong/SparseGAD>
- SEC-GFD: <https://github.com/Sunxkissed/SEC-GFD>
- NRGL: <https://github.com/Shzuwu/NRGL>
- CGADM: <https://github.com/weicy15/CGADM>

Not every project adaptation maps to an independently verified public official repository. Fixed commits, configuration provenance, and retained source evidence for the remaining methods live under the corresponding `methods/<method>/configs/`, `methods/<method>/audit/`, or `methods/<method>/official_snapshot/` directory. This README does not guess an upstream URL when one has not been verified. ConsisGAD is not present and is not replaced by GraphConsis.

## 12. Release and licensing boundary

- This repository currently has no project-wide `LICENSE` or `CITATION.cff`; absence of a license does not grant unrestricted redistribution rights.
- Third-party code under `official_snapshot/` remains subject to its upstream license. Check every upstream repository before use or redistribution.
- Papers, datasets, and baseline implementations must still be cited even when their code is organized in this repository.

Please cite the corresponding papers and official repositories when using any method or dataset.
