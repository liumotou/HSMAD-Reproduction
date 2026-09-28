# Preprocessing Module: Curvature Computation

This module contains the preprocessing pipeline for computing Ollivier-Ricci curvature, which is a fundamental component of CurvGAD.

## 📁 Files Overview

### `compute_curvature.py`
Main script for computing and storing Ollivier-Ricci curvature for graph datasets.

**Key Functions:**
- `compute_and_store_curvature()`: Main function for curvature computation
- `accelerated_ricci_approximation()`: Fast approximation method for large graphs
- `convert_multidigraph_to_graph()`: Converts multi-directed graphs to simple graphs

### `ollivier_ricci_flow.py`
Implementation of discrete Ollivier-Ricci flow for graph regularization.

**Key Functions:**
- `ollivier_ricci_flow()`: Applies Ricci flow to smooth graph geometry
- `compute_ricci_flow_step()`: Single step of the flow process

### `preprocess_external_datasets.py`
Dataset preprocessing utilities for external graph datasets.

**Key Functions:**
- Dataset loading and format conversion
- Train/validation/test split generation
- Feature normalization and preprocessing

## 🔧 Usage

### Basic Curvature Computation

```bash
python compute_curvature.py --datasets cora citeseer pubmed
```

### Fast Approximation (Recommended for Large Graphs)

```bash
python compute_curvature.py --datasets reddit weibo --use_accelerated --save_matrix
```

### Compare Methods

```bash
python compute_curvature.py --datasets texas --compare_both
```

## 🧮 Curvature Computation Methods

### 1. Exact Ollivier-Ricci Curvature

Uses the GraphRicciCurvature library to compute exact curvature values:

```python
from GraphRicciCurvature.OllivierRicci import OllivierRicci

orc = OllivierRicci(graph, alpha=0.5, verbose="INFO")
orc.compute_ricci_curvature()
```

**Advantages:**
- Mathematically exact
- Well-established implementation
- Reliable for small to medium graphs

**Disadvantages:**
- Computationally expensive O(n³)
- Memory intensive for large graphs
- Not scalable beyond ~10K nodes

### 2. Accelerated Approximation

Fast linear-time approximation using tighter lower bounds:

```python
def accelerated_ricci_approximation(graph, num_nodes):
    # Compute degree-based approximation
    curvature_approx = 2 - (d_u + d_v) / min(d_u, d_v)
    return curvature_matrix
```

**Advantages:**
- Linear time complexity O(m)
- Memory efficient
- Scalable to large graphs (>100K nodes)

**Disadvantages:**
- Approximation with bounded error
- Less precise than exact method
- May miss subtle curvature variations

## 📊 Output Format

Curvature data is stored in sparse matrix format (`.npz` files) in the `curvature_data/` directory:

```
curvature_data/
├── cora_curvature_matrix.npz
├── citeseer_curvature_matrix.npz
├── pubmed_curvature_matrix.npz
└── ...
```

Each file contains:
- **Sparse CSR matrix**: Edge curvature values
- **Row indices**: Source nodes
- **Column indices**: Target nodes
- **Data values**: Curvature values

## 🎯 Curvature Interpretation

### Positive Curvature (> 0)
- **Geometric meaning**: Locally spherical/curved inward
- **Graph interpretation**: Dense, well-connected regions
- **Anomaly relevance**: Normal community structures

### Negative Curvature (< 0)
- **Geometric meaning**: Locally hyperbolic/curved outward
- **Graph interpretation**: Sparse, tree-like structures
- **Anomaly relevance**: Potential structural anomalies

### Zero Curvature (≈ 0)
- **Geometric meaning**: Locally flat/Euclidean
- **Graph interpretation**: Regular grid-like patterns
- **Anomaly relevance**: Neutral geometric regions

## ⚡ Performance Optimization

### Memory Management
- Use sparse matrices for storage
- Process graphs in chunks for very large datasets
- Clear intermediate variables to free memory

### Computational Efficiency
- Leverage vectorized operations with NumPy
- Use multiprocessing for independent computations
- Cache results to avoid recomputation

### Recommended Settings

| Graph Size | Method | Parameters |
|------------|--------|------------|
| < 1K nodes | Exact | `alpha=0.5` |
| 1K-10K nodes | Exact | `alpha=0.5, verbose=False` |
| 10K-100K nodes | Accelerated | `use_accelerated=True` |
| > 100K nodes | Accelerated | `use_accelerated=True, save_matrix=True` |

## 🔬 Validation and Quality Control

### Curvature Range Validation
```python
# Check curvature bounds
assert -2 <= curvature_values.min() <= curvature_values.max() <= 2
```

### Symmetry Check
```python
# Verify matrix symmetry for undirected graphs
assert np.allclose(curvature_matrix, curvature_matrix.T)
```

### Sparsity Analysis
```python
# Monitor sparsity levels
sparsity = 1 - (curvature_matrix.nnz / (num_nodes * num_nodes))
print(f"Curvature matrix sparsity: {sparsity:.3f}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error**: Use `--use_accelerated` for large graphs
2. **Slow Computation**: Enable multiprocessing or use approximation
3. **NaN Values**: Check for disconnected components
4. **File Not Found**: Ensure dataset exists in `datasets/` directory

### Debug Mode
```bash
python compute_curvature.py --datasets debug_dataset --verbose --save_intermediate
```

For questions about curvature computation, please refer to the main project README or open an issue. 