# Runtime environments

The recovered methods span incompatible historical graph-library stacks. Do not install every dependency into one environment.

## DGL candidate environment

The server workspace verified Python 3.10 with PyTorch `2.1.2+cu121`, DGL `1.1.3+cu121`, NumPy `1.26.4`, SciPy `1.15.3`, pandas `2.3.3`, scikit-learn `1.7.2`, and SymPy `1.12`.

This environment covers the DGL-native runners such as BWGNN, GraphSAGE, CARE-GNN, GraphConsis, PC-GNN, SEC-GFD, DSGAD, SpaceGNN, PMP, and the feature-only MLP/SVM paths.

## PyG standard-baseline environment

ChebNet and GIN use PyG adapters. The verified isolated environment used PyTorch `2.1.0+cu121`, DGL `1.1.3`, PyTorch Geometric `2.4.0`, NumPy `1.26.4`, SciPy `1.15.3`, scikit-learn `1.7.2`, and SymPy `1.14.0`.

PyG binary extensions must match the exact PyTorch and CUDA ABI. Install them from the official PyG wheel index for the selected pair rather than copying `.so` files between environments.

## AMNet legacy environment

The fixed AMNet snapshot is commit `74f6a826ddb56dc73f82aa523bf350a1e28b8f03`. Its audited environment used Python 3.10, PyTorch `1.11.0+cu113`, torchvision `0.12.0+cu113`, PyG `2.0.4`, and matching compiled extensions. It requires CUDA 11 user-space libraries, including `libcusparse.so.11`; a newer host toolkit does not replace that ABI.

Use `methods/amnet_hsmad/run_in_official_env.sh` as the environment boundary. Do not replace the repository's main PyTorch/DGL installation to satisfy AMNet.

## Validation order

1. `python scripts/validate_repository.py`
2. `python -m unittest tests.test_project_paths tests.test_public_repository_contract -v`
3. Import the selected environment's graph libraries and print their real paths and versions.
4. Validate dataset fields, dimensions, frozen mask counts, and SHA256.
5. Run a five-epoch smoke before diagnostic or formal execution.

Environment validation proves dependency compatibility only. It does not turn a partial or blocked method/dataset combination into a completed result.
