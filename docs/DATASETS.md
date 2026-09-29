# Dataset download and provenance

This repository does **not** redistribute dataset binaries. Download the data from the official or benchmark-maintained sources below and place the extracted files under `datasets/` (ignored by Git).

## Unified GADBench package

GADBench provides a unified archive covering the benchmark datasets used here:

- Repository: https://github.com/squareroot3/GADBench
- Fixed reference commit used by this project: `f9aa021ce9b6c6580427fb633b596843be76ddc6`
- Official Google Drive archive (`datasets.zip`): https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing

After downloading, extract the archive into `datasets/`. Project-adapted baseline runners consume the graph's frozen train/validation/test masks and must not regenerate them silently. The HSMAD root loader is a documented exception: it deterministically rebuilds an approximately 40/20/40 split with `random_state=2` and overwrites graph masks. Verify counts and SHA256 before claiming that this reconstructed split is identical to a baseline's persisted split.

## Per-dataset sources

| Dataset | Official/maintained source | Notes |
|---|---|---|
| Weibo | PyGOD data repository: https://github.com/pygod-team/data (`weibo.pt.zip`) | GADBench also includes a processed Weibo version in the unified archive. Verify hashes before substituting formats. |
| Amazon | DGL `FraudAmazonDataset`: https://docs.dgl.ai/en/latest/generated/dgl.data.FraudAmazonDataset.html | Can be downloaded automatically by DGL; the GADBench processed version is also in the unified archive. In this project, the first 3305 uncovered nodes remain in the transductive graph but are excluded from train/validation/test masks. |
| YelpChi | DGL `FraudYelpDataset`: https://docs.dgl.ai/en/latest/generated/dgl.data.FraudYelpDataset.html | The project version was audited against the official GADBench Google Drive graph. Feature and label hashes matched; the project graph equals the official candidate after removing one self-loop per node. |
| Tolokers | DGL `TolokersDataset`: https://docs.dgl.ai/en/latest/generated/dgl.data.TolokersDataset.html | This project uses frozen HSMAD masks, not the dataset's built-in 50/25/25 masks. |
| T-Finance | BWGNN author folder: https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing | Also included in the GADBench unified archive. |
| T-Social | BWGNN author folder: https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing | Also included in the GADBench unified archive; full-graph training has high GPU-memory requirements. |

## Reproducibility rules

1. Do not commit raw datasets, extracted graphs, masks, or cached downloads.
2. Record SHA256 for feature, label, graph, and all three masks before training.
3. Keep the split fixed across seeds `0..9`; distinguish persisted graph masks from HSMAD's deterministic `random_state=2` reconstruction.
4. Graph methods use the documented graph preprocessing pipeline; feature-only methods must not read edges.
5. Dataset formats from different sources are not assumed interchangeable solely because node/edge counts match.

## Source citations

- GADBench official repository and dataset instructions: https://github.com/squareroot3/GADBench
- BWGNN official repository and T-Finance/T-Social source: https://github.com/squareroot3/Rethinking-Anomaly-Detection
- PyGOD maintained data repository: https://github.com/pygod-team/data
