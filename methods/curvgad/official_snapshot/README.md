# CurvGAD: Leveraging Curvature for Enhanced Graph Anomaly Detection

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2502.08605)
[![Conference](https://img.shields.io/badge/Conference-ICML%202025-blue)](https://icml.cc/Conferences/2025)

This is the official implementation of **CurvGAD: Leveraging Curvature for Enhanced Graph Anomaly Detection** accepted at ICML 2025.

## 📖 Abstract

Does the intrinsic curvature of complex networks hold the key to unveiling graph anomalies that conventional approaches overlook? Reconstruction-based graph anomaly detection (GAD) methods overlook such geometric outliers, focusing only on structural and attribute-level anomalies. To this end, we propose CurvGAD - a mixed-curvature graph autoencoder that introduces the notion of curvature-based geometric anomalies. 

CurvGAD introduces two parallel pipelines for enhanced anomaly interpretability:
1. **Curvature-equivariant geometry reconstruction**: Focuses exclusively on reconstructing the edge curvatures using a mixed-curvature, Riemannian encoder and Gaussian kernel-based decoder
2. **Curvature-invariant structure and attribute reconstruction**: Decouples structural and attribute anomalies from geometric irregularities by regularizing graph curvature under discrete Ollivier-Ricci flow

By leveraging curvature, CurvGAD refines the existing anomaly classifications and identifies new curvature-driven anomalies. Extensive experimentation over 10 real-world datasets (both homophilic and heterophilic) demonstrates an improvement of up to 6.5% over state-of-the-art GAD methods.

## 🏗️ Repository Structure

```
curvgad-codebase/
├── preprocessing/              # Curvature computation modules
│   ├── compute_curvature.py   # Main curvature computation script
│   ├── ollivier_ricci_flow.py # Ollivier-Ricci flow implementation
│   └── preprocess_external_datasets.py # Dataset preprocessing
├── models/                     # Model implementations
│   ├── curvgad_detector.py    # Main CurvGAD detector
│   ├── curvgad_gnn.py         # CurvGAD GNN architecture
│   ├── detector.py            # Base detector classes
│   ├── gnn.py                 # GNN implementations
│   ├── attention.py           # Attention mechanisms
│   ├── manifold_cheb_conv.py  # Manifold Chebyshev convolution
│   └── mobius_linear.py       # Möbius linear transformations
├── datasets/                   # Dataset storage
├── curvature_data/            # Precomputed curvature matrices
├── results/                   # Experimental results
├── plots/                     # Generated plots and visualizations
├── runs/                      # Training logs and outputs
├── benchmark.py               # Main benchmarking script
├── utils.py                   # Utility functions
├── job.sh                     # SLURM job submission script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup Environment

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/curvgad-codebase.git
cd curvgad-codebase
```

2. **Create a virtual environment:**
```bash
python -m venv curvgad_env
source curvgad_env/bin/activate  # On Windows: curvgad_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Key Dependencies
- **PyTorch**: Deep learning framework
- **DGL**: Deep Graph Library for graph neural networks
- **PyTorch Geometric**: Graph neural network library
- **Geoopt**: Riemannian optimization library
- **GraphRicciCurvature**: Ricci curvature computation
- **PyGOD**: Graph outlier detection library

## 📊 Datasets

The repository supports multiple graph datasets including:
- **Citation Networks**: Cora, CiteSeer, PubMed
- **Social Networks**: Actor, Chameleon, Squirrel
- **Web Networks**: Texas, Cornell, Wisconsin
- **Fraud Detection**: Amazon, Yelp, Reddit
- **Financial Networks**: TFinance, Elliptic

Datasets are automatically downloaded and preprocessed when first used.

## 🔧 Usage

### Step 1: Precompute Curvature

Before running the main model, you need to compute the Ollivier-Ricci curvature for your datasets:

```bash
cd preprocessing
python compute_curvature.py --datasets cora citeseer pubmed --use_accelerated --save_matrix
```

**Arguments:**
- `--datasets`: List of datasets to compute curvature for
- `--use_accelerated`: Use fast approximation method (recommended for large graphs)
- `--compare_both`: Compare exact vs. accelerated methods
- `--save_matrix`: Save curvature matrices for later use

### Step 2: Run CurvGAD

Execute the main benchmarking script:

```bash
python benchmark.py --models CurvGAD --datasets cora --trials 10 --use_autoencoder --manifold_config H32S32E32 --optimizer riemannian_adam --lr 0.01 --K 7
```

**Key Arguments:**
- `--models`: Model to use (CurvGAD or baseline methods)
- `--datasets`: Datasets to evaluate on
- `--trials`: Number of experimental trials
- `--use_autoencoder`: Enable autoencoder with curvature reconstruction
- `--manifold_config`: Manifold configuration (e.g., H32S32E32 for 32-dim hyperbolic, spherical, and Euclidean)
- `--optimizer`: Optimizer type (adam or riemannian_adam)
- `--lr`: Learning rate
- `--K`: Number of Chebyshev polynomial filters

### Step 3: Analyze Results

Results are automatically saved in the `results/` directory as Excel files. You can analyze the performance metrics including:
- AUC-ROC scores
- Average Precision (AP)
- Training time
- Memory usage

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{grover2025curvgadleveragingcurvatureenhanced,
      title={CurvGAD: Leveraging Curvature for Enhanced Graph Anomaly Detection}, 
      author={Karish Grover and Geoffrey J. Gordon and Christos Faloutsos},
      year={2025},
      eprint={2502.08605},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.08605}, 
}
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 📞 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [karish@cs.cmu.edu](mailto:karish@cs.cmu.edu)

---
