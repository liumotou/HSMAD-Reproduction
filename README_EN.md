# HSMAD Unified Reproduction Workspace

[中文](README.md) | [English](README_EN.md)

This repository contains the HSMAD implementation and project-adapted Table 1 baseline runners under a shared data, frozen-split, and evaluation protocol.

> **Positioning:** unless explicitly stated otherwise, baseline implementations under `methods/` are labelled `candidate_protocol_not_author_exact`. They are auditable project adaptations, not claims of byte-identical author code or exact four-decimal reproduction.
>
> Dataset binaries, frozen masks, checkpoints, logs, and formal result artifacts are intentionally excluded from Git. This repository publishes sources, configurations, provenance links, and usage instructions.

## 1. Objective

The project compares HSMAD and its baselines with:

- fixed dataset versions and persistent train/validation/test masks;
- training seeds `0..9` without regenerating splits;
- train-only loss and parameter updates;
- validation-only early stopping, checkpoint selection, and F1 threshold selection;
- test-only final F1-Macro and AUROC after model and threshold are fixed;
- per-run config, log, history, checkpoint, metrics, time, memory, and SHA256 records;
- sample standard deviation (`ddof=1`) for ten-seed summaries.

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

## 6. Baseline readiness

| Method | Entry point | Note |
|---|---|---|
| HSMAD | `main.py` | Main project method |
| MLP | `methods/mlp/src/formal_runner.py` | Feature-only server experiment source recovered |
| GCN | `methods/gcn/src/run_kipf_v2.py` | Kipf two-layer candidate |
| ChebNet | `methods/chebnet/src/run.py` | PyG candidate protocol |
| GIN | `methods/gin/src/run.py` | PyG candidate protocol |
| GWNN | `methods/gwnn/src/run.py` | Paper-formula candidate protocol |
| GAT v1 | smoke/diagnostic runners | Archived diagnostic candidate |
| GAT-v2 | `methods/gat_v2_gadbench/src/run_*.py` | GADBench-structure adaptation |
| GraphSAGE | `methods/graphsage/src/run_smoke.py`, `run_formal.py` | GADBench pool candidate |
| GraphConsis | `methods/graphconsis/src/run.py` | Single-relation adapted candidate |
| CARE-GNN | `methods/caregnn/src/run.py` | Single-relation adapted candidate |
| PC-GNN | `methods/pcgnn/src/run.py` | Single-relation adapted candidate |
| BWGNN | `methods/bwgnn/src/run_smoke.py`, `run_formal.py` | Project-adapted candidate |
| SparseGAD | `methods/sparsegad/src/run_smoke.py`, `run_formal.py` | Project-adapted candidate |
| SVM | `methods/svm/src/runner.py` | Feature-only candidate |
| SEC-GFD | `methods/sec_gfd/src/runner.py` | Selected-dataset candidate |
| GHRN | `methods/ghrn/src/runner.py` | Project-adapted candidate |
| CGADM | `methods/cgadm_hsmad/src/runner.py` | HSMAD-data adapter |
| DSGAD | `methods/dsgad/src/runner.py` | Project-adapted candidate |
| NRGL | Python API `run(...)` | No standalone CLI main in this snapshot |
| CurvGAD | partial components | Known architecture/shape blocker |
| SpaceGNN | `methods/spacegnn_hsmad/src/runner.py` | Weibo candidate completed; other combinations retain blocker evidence |
| AMNet | `methods/amnet_hsmad/run_smoke.py`, `run_full.py` | Isolated CUDA 11 ABI environment required |
| PMP | `methods/pmp_hsmad/src/runner.py` | Frozen HSMAD protocol candidate |

See [docs/BASELINE_SCRIPTS.md](docs/BASELINE_SCRIPTS.md) for the detailed index.

## 7. Environment and validation

Most DGL candidates require Python 3.10, PyTorch, a compatible DGL build, NumPy, SciPy, pandas, scikit-learn, and SymPy. AMNet uses a separate legacy PyTorch/PyG environment and should not overwrite the main DGL environment. See [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) for verified environment boundaries.

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

Run commands from the repository root. Some archived runners contain a fixed project `ROOT`; inspect them before running from a different path:

```bash
rg -n "ROOT\s*=|/root/autodl-tmp/HSMAD" methods
```

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

```bash
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

## 11. Upstream sources

- GADBench: <https://github.com/squareroot3/GADBench>
- BWGNN: <https://github.com/squareroot3/Rethinking-Anomaly-Detection>
- GAT: <https://github.com/PetarV-/GAT>
- GraphSAGE: <https://github.com/williamleif/GraphSAGE>
- AMNet: <https://github.com/Illyasville/AMNet>
- SparseGAD: <https://github.com/KellyGong/SparseGAD>
- SEC-GFD: <https://github.com/Sunxkissed/SEC-GFD>
- NRGL: <https://github.com/Shzuwu/NRGL>

Please cite the corresponding papers and official repositories when using any method or dataset.
