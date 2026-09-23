# HSMAD reproducibility workspace

This repository contains the HSMAD implementation and project-adapted baseline runners used to reproduce Table 1 under a shared data, split, and evaluation protocol.

> **Important:** most baseline folders are labelled `candidate_protocol_not_author_exact`. They are auditable project adaptations, not claims of byte-identical author code or exact four-decimal reproduction. Dataset binaries, frozen masks, checkpoints, logs, and experimental result files are intentionally not stored in Git.

## Project objective

The project compares HSMAD and its Table 1 baselines under the following controls:

- the same dataset version and persistent train/validation/test masks;
- training seeds `0..9`, without regenerating a split per seed;
- validation-only checkpoint and F1-threshold selection;
- test data used only after the model and threshold are fixed;
- test reporting with F1-Macro and AUROC;
- per-run config, log, checkpoint, metrics, wall time, peak memory, and SHA256 records;
- ten-seed summaries using the sample standard deviation (`ddof=1`).

## Repository layout

```text
.
├── main.py, model.py, dataset.py, manifold_update.py, utils.py  # HSMAD
├── methods/                                                     # baseline implementations
│   ├── gcn/
│   ├── gat/
│   ├── gat_v2_gadbench/
│   ├── graphsage/
│   ├── bwgnn/
│   ├── sparsegad/
│   ├── nrgl/
│   ├── sec_gfd/
│   ├── ghrn/
│   ├── svm/
│   └── ...
├── docs/DATASETS.md             # official/maintained download links
├── docs/BASELINE_SCRIPTS.md      # runner and config index
├── scripts/validate_repository.py
└── datasets/                     # local only; ignored by Git
```

## Dataset preparation

See [docs/DATASETS.md](docs/DATASETS.md) for the download links and provenance notes.

The main unified source is the official GADBench archive:

- GADBench: https://github.com/squareroot3/GADBench
- dataset archive: https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing

Extract the required files into:

```text
datasets/weibo
datasets/amazon
datasets/yelp
datasets/tolokers
datasets/tfinance
datasets/tsocial
```

These are DGL graph files without filename extensions in the current project convention. A graph is expected to contain `feature`, `label`, `train_mask`, `val_mask`, and `test_mask` in `graph.ndata`.

### Special split rules

- **Amazon:** the first 3305 uncovered nodes remain in the complete transductive graph and may participate in message passing, but they are excluded from all three masks, the loss, and reported metrics.
- **Tolokers:** the project uses the frozen HSMAD masks, not Tolokers' built-in 50/25/25 masks.
- **Graph methods:** project experiments generally use `to_bidirected -> remove_self_loop -> add_self_loop` where specified by the protocol config.
- **Feature-only methods:** must not read graph edges.

## What happened with Yelp?

Yelp has two separate issues that should not be conflated:

1. **Data provenance is verified.** The project compared the official GADBench Google Drive candidate with the local Yelp graph. Feature SHA256 and label SHA256 matched. The official candidate had exactly 45,954 additional edges—one self-loop per node—and the local graph is explainable as the official candidate after `remove_self_loop()`.
2. **Experiment coverage was deliberately frozen.** During later baseline work, Yelp was explicitly excluded from new runs while its protocol and provenance were being reviewed. Therefore, a method without a Yelp result is not evidence that the Yelp file is invalid or that the method cannot theoretically process Yelp.

In short: **the local Yelp data source is verified, but not every baseline has a completed/audited Yelp experiment.** Existing Yelp results must be interpreted according to the protocol and status stored with that method, rather than mixed with later candidates.

## Why do some baselines cover only one or two datasets?

The repository records actual completed adaptation work, not a fabricated full matrix. Missing combinations generally fall into one of these categories:

- **Official-source scope:** some author repositories directly support only particular datasets or formats (for example, AMNet officially provides PyG logic for Yelp/Elliptic rather than the complete six-dataset DGL protocol).
- **Adapter maturity:** a model may exist, while its dataset loader, formal runner, or independent checkpoint-recompute audit is not yet complete for every dataset.
- **Resource limits:** full-graph methods on T-Finance or T-Social can exceed a 24 GB GPU. The project does not silently sample, shrink the graph, or reduce the model merely to make a run finish.
- **Protocol freeze:** Yelp was intentionally held back; T-Social was reserved for suitable high-memory hardware.
- **Known implementation blockers:** CurvGAD reached an architecture/shape incompatibility; AMNet required an isolated legacy CUDA/PyG environment; PMP and some newer methods have only partial components in this snapshot.
- **Audit requirements:** smoke or a single diagnostic is not promoted to a ten-seed formal result until checkpoint recomputation, mask isolation, and test-leakage checks pass.

Thus, “only one or two datasets” usually means **the remaining cells are incomplete or blocked under the strict protocol**, not that low-scoring runs were removed.

## Baseline readiness

| Method | Code in repository | Standalone runner | Current caution |
|---|---:|---:|---|
| HSMAD | Yes | `main.py` | Requires local datasets and DGL/CUDA environment. |
| GCN | Yes | Yes | Project Kipf-topology candidate; not author-exact HSMAD baseline. |
| GAT (older candidate) | Yes | Smoke/diagnostic | Archived diagnostic topology, not the current primary GAT candidate. |
| GAT-v2 | Yes | Yes | GADBench-adapted candidate; T-Social uses the documented hidden-10 exception. |
| GraphSAGE | Yes | Yes | GADBench-style full-graph pool candidate; determinism audits matter on CUDA. |
| BWGNN | Yes | Yes | Candidate implementation; dataset coverage encoded in runner contracts. |
| SparseGAD | Yes | Yes | Candidate protocol, not author-exact. |
| SVM | Yes | Yes | Feature-only candidate. |
| SEC-GFD | Yes | Yes | Candidate adapter for selected datasets. |
| GHRN | Yes | Yes | Candidate adapter; official/protocol differences remain relevant. |
| CGADM | Yes | Yes | Candidate HSMAD-data adapter. |
| DSGAD | Yes | Yes | Candidate runner for the datasets declared in its contract. |
| NRGL | Yes | Python API (`run`) | No command-line `main` in the current snapshot. |
| CurvGAD | Partial | No formal runner | Precompute/adaptation code retained; known architecture/shape blocker. |
| AMNet | Partial | No complete runner | Environment and adaptation work retained; this snapshot is not directly trainable. |
| PMP | Partial | No complete runner | Model/protocol/audit components only in this snapshot. |

See [docs/BASELINE_SCRIPTS.md](docs/BASELINE_SCRIPTS.md) for the detailed entrypoint index.

## Environment

There is no single verified environment that runs every historical baseline. The DGL candidates were primarily developed with PyTorch/DGL CUDA environments; AMNet uses an isolated legacy PyTorch/PyG stack. Do not install AMNet's legacy dependencies over the main DGL environment.

At minimum, the principal DGL runners require compatible versions of:

- Python 3.10 (recommended for the project snapshots);
- PyTorch with CUDA support;
- DGL built for the same CUDA/PyTorch combination;
- NumPy, SciPy, pandas, scikit-learn and SymPy.

Before long training, verify imports and run a five-epoch smoke test for the exact method/dataset pair.

## Repository validation

This check does not require datasets or GPU libraries. It validates published JSON, Python syntax, required entrypoints, and accidental large files:

```bash
python scripts/validate_repository.py
```

Passing this check means the repository snapshot is structurally consistent. It **does not** prove that every training job can run without the correct datasets, CUDA runtime, DGL/PyG build, and GPU memory.

## Example commands

Run from the repository root.

### HSMAD

```bash
python main.py --dataset weibo --run 10 --epoch 1000 --patience 100 --hid_dim 64 --order 2 --q 0.5
```

### GCN candidate, one seed

```bash
python methods/gcn/src/run_kipf_v2.py \
  --config methods/gcn/configs/protocol_v2_weibo_formal.json \
  --seed 0
```

### GAT-v2 candidate, one seed

```bash
python methods/gat_v2_gadbench/src/run_formal.py \
  --config methods/gat_v2_gadbench/configs/weibo_protocol_v2_gadbench_hidden64_formal.json \
  --seed 0
```

### GraphSAGE candidate, one seed

```bash
python methods/graphsage/src/run_formal.py \
  --config methods/graphsage/configs/weibo_graphsage_gadbench_h64_formal.json \
  --seed 0
```

### BWGNN candidate, seeds 0–9

```bash
python methods/bwgnn/src/run_formal.py \
  --config methods/bwgnn/configs/weibo_bwg_h64_smoke.json \
  --seeds 0-9 \
  --run-type formal
```

The BWGNN filename contains `smoke` for historical reasons; inspect the config and runner-generated formal snapshot before using it. Prefer a dataset-specific formal config when one is available.

### SparseGAD candidate, one diagnostic seed

```bash
python methods/sparsegad/src/run_formal.py \
  --config methods/sparsegad/configs/weibo_sparsegad_h64_candidate.json \
  --run-type diagnostic \
  --seeds 0
```

### SVM feature-only candidate

```bash
python methods/svm/src/runner.py --dataset weibo --seed 0 --run-type diagnostic
```

## Safe execution sequence

For any method/dataset pair:

1. Run `python scripts/validate_repository.py`.
2. Verify dataset, feature, label, and mask SHA256 against the intended frozen version.
3. Run a five-epoch smoke test in a new output directory.
4. Independently reload the saved checkpoint and reproduce its metrics.
5. Run a separate seed-0 full diagnostic.
6. Only after the audit passes, run formal seeds `0..9` serially.
7. Calculate mean and sample standard deviation (`ddof=1`) from formal/OK records only.

Never reuse smoke/diagnostic checkpoints as formal results, delete a low-scoring seed, or choose a checkpoint/threshold using test metrics.

## Known limitations

- The repository currently publishes code and configs, not the large datasets or historical experiment artifacts.
- Some configs are protocol snapshots tied to a particular experiment; read them before changing dataset or output paths.
- Static validation cannot detect CUDA/DGL nondeterminism, insufficient memory, ABI mismatches, or semantic differences from an upstream author implementation.
- A completed candidate result is suitable for project comparison only when its accompanying audit passes; it must not automatically be called an author-exact reproduction.
- MLP formal artifacts existed in the experimental workspace, but a complete standalone MLP runner was not present in this local source snapshot and therefore was not invented for this upload.

## Upstream sources

Key upstream references include:

- GADBench: https://github.com/squareroot3/GADBench
- BWGNN: https://github.com/squareroot3/Rethinking-Anomaly-Detection
- Original GAT: https://github.com/PetarV-/GAT
- Original GraphSAGE: https://github.com/williamleif/GraphSAGE
- AMNet: https://github.com/Illyasville/AMNet
- SparseGAD: https://github.com/KellyGong/SparseGAD
- SEC-GFD: https://github.com/Sunxkissed/SEC-GFD
- NRGL: https://github.com/Shzuwu/NRGL

Please cite the original papers and repositories for any method or dataset used.
