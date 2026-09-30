# Models Module: CurvGAD Architecture

This module contains the core model implementations for CurvGAD, including the mixed-curvature graph neural network and anomaly detection framework.

## 📁 Files Overview

### `curvgad_detector.py`
Main detector class that implements the dual-pipeline CurvGAD architecture.

**Key Components:**
- `CurvGADDetector`: Main detector with dual reconstruction pipelines
- Curvature-equivariant geometry reconstruction
- Curvature-invariant structure/attribute reconstruction
- Trade-off parameter optimization

### `curvgad_gnn.py`
Core GNN implementation with mixed-curvature geometry.

**Key Components:**
- `CurvGAD`: Mixed-curvature graph neural network
- Manifold Chebyshev convolution layers
- Multi-manifold embedding spaces
- Attention-based filter aggregation

### `detector.py`
Base detector classes and baseline implementations.

**Key Components:**
- `BaseDetector`: Abstract base class for all detectors
- Baseline GNN detectors (GCN, GAT, GraphSAGE, etc.)
- Classical ML detectors (SVM, KNN, RF, XGBoost)
- Specialized detectors (BGNN, PCGNN, DCI)

### `gnn.py`
Standard GNN layer implementations.

**Key Components:**
- Various GNN architectures
- Layer normalization and dropout
- Activation functions and regularization

### `attention.py`
Attention mechanism implementations.

**Key Components:**
- Multi-head attention
- Graph attention networks
- Attention pooling mechanisms

### `manifold_cheb_conv.py`
Manifold-aware Chebyshev convolution layer.

**Key Components:**
- `ManifoldChebConv`: Chebyshev convolution on manifolds
- Polynomial basis functions
- Riemannian optimization compatibility

### `mobius_linear.py`
Möbius linear transformations for hyperbolic geometry.

**Key Components:**
- Hyperbolic linear layers
- Möbius transformations
- Gradient computation in hyperbolic space

## 🏗️ CurvGAD Architecture

### Dual-Pipeline Design

```
Input Graph → Feature Extraction → Dual Pipelines → Anomaly Scores
                                      ↓
                     ┌─────────────────────────────────────┐
                     │  Pipeline 1: Curvature Reconstruction  │
                     │  - Mixed-curvature encoder            │
                     │  - Gaussian kernel decoder            │
                     │  - Geometry-focused loss              │
                     └─────────────────────────────────────┘
                                      ↓
                     ┌─────────────────────────────────────┐
                     │  Pipeline 2: Structure/Attribute   │
                     │  - Euclidean encoder                │
                     │  - Standard decoder                 │
                     │  - Structure/attribute-focused loss │
                     └─────────────────────────────────────┘
```

### Mixed-Curvature Embedding

The model operates in a product manifold combining:

1. **Hyperbolic Space (H)**: Captures hierarchical structures
2. **Spherical Space (S)**: Models cyclic patterns
3. **Euclidean Space (E)**: Handles flat structures

Configuration example: `H32S32E32` = 32-dimensional hyperbolic + 32-dimensional spherical + 32-dimensional Euclidean

## 🔧 Usage Examples

### Basic CurvGAD Training

```python
from models.curvgad_detector import CurvGADDetector
from utils import Dataset

# Load dataset
data = Dataset('cora')

# Configure model
model_config = {
    'in_feats': data.graph.ndata['feature'].shape[1],
    'h_feats': 64,
    'num_classes': 2,
    'manifolds_config_str': 'H32S32E32',
    'K': 7,
    'dropout_rate': 0.1
}

train_config = {
    'device': 'cuda',
    'lr': 0.01,
    'optimizer': 'riemannian_adam',
    'use_autoencoder': True,
    'alpha': 1.0,
    'dataset_name': 'cora'
}

# Initialize detector
detector = CurvGADDetector(train_config, model_config, data)

# Train and evaluate
detector.fit(data, split_idx=0)
scores = detector.predict(data, split_idx=0)
```

### Manifold Configuration

```python
# Different manifold configurations
configs = {
    'hyperbolic_only': 'H96',          # Pure hyperbolic
    'spherical_only': 'S96',           # Pure spherical  
    'euclidean_only': 'E96',           # Pure Euclidean
    'mixed_equal': 'H32S32E32',        # Equal dimensions
    'hyperbolic_dominant': 'H64S16E16', # Hyperbolic focus
    'adaptive': 'H48S24E24'            # Balanced mix
}
```

### Custom Loss Functions

```python
def custom_curvature_loss(pred_curvature, true_curvature, edge_indices):
    """Custom loss for curvature reconstruction"""
    # Weighted MSE based on edge importance
    edge_weights = compute_edge_importance(edge_indices)
    loss = (edge_weights * (pred_curvature - true_curvature) ** 2).mean()
    return loss
```

## 🎯 Key Features

### 1. Mixed-Curvature Geometry

**Hyperbolic Components:**
- Captures tree-like and hierarchical structures
- Negative curvature regions
- Exponential volume growth

**Spherical Components:**
- Models cyclic and periodic patterns
- Positive curvature regions
- Bounded embedding space

**Euclidean Components:**
- Handles flat and regular structures
- Zero curvature regions
- Standard linear operations

### 2. Manifold Chebyshev Convolution

```python
class ManifoldChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K, manifold):
        # K-order Chebyshev polynomials on manifolds
        self.K = K
        self.manifold = manifold
        
    def forward(self, x, laplacian):
        # Compute Chebyshev basis on manifold
        # Apply polynomial filters
        return filtered_features
```

### 3. Attention-Based Filter Aggregation

```python
# Multi-scale filter bank
filter_outputs = []
for conv in self.conv_layers:
    h = conv(x, laplacian)
    filter_outputs.append(h)

# Attention-weighted aggregation
attn_weights = F.softmax(self.filter_attention, dim=0)
aggregated = (attn_weights * filter_outputs).sum(dim=0)
```

### 4. Dual Reconstruction Loss

```python
def compute_total_loss(self, data, predictions):
    # Curvature reconstruction loss
    curv_loss = self.curvature_reconstruction_loss(
        predictions['curvature'], data['true_curvature']
    )
    
    # Structure/attribute reconstruction loss
    struct_loss = self.structure_reconstruction_loss(
        predictions['adjacency'], data['adjacency']
    )
    
    attr_loss = self.attribute_reconstruction_loss(
        predictions['features'], data['features']
    )
    
    # Weighted combination
    total_loss = (
        self.alpha * curv_loss +
        self.beta * struct_loss +
        self.gamma * attr_loss
    )
    
    return total_loss
```

## 🔬 Model Variants

### 1. CurvGAD-Full
Complete model with both pipelines and all manifold types.

### 2. CurvGAD-Curv
Only curvature reconstruction pipeline.

### 3. CurvGAD-Struct
Only structure/attribute reconstruction pipeline.

### 4. CurvGAD-H/S/E
Single manifold variants (hyperbolic/spherical/Euclidean only).

## ⚡ Performance Optimization

### Memory Efficiency
- Sparse matrix operations for large graphs
- Gradient checkpointing for deep networks
- Mixed precision training

### Computational Efficiency
- Vectorized manifold operations
- Efficient Chebyshev polynomial computation
- Parallel processing for multiple manifolds

### Scalability Tips
- Use smaller embedding dimensions for large graphs
- Reduce number of Chebyshev filters (K)
- Enable gradient accumulation for large batches

## 🐛 Troubleshooting

### Common Issues

1. **Manifold Projection Errors**
   - Check manifold parameter initialization
   - Verify gradient flow through manifold layers
   - Use smaller learning rates for Riemannian optimization

2. **Curvature Loading Errors**
   - Ensure curvature matrices are precomputed
   - Check file paths in `curvature_data/`
   - Verify dataset name matching

3. **Memory Issues**
   - Reduce batch size or embedding dimensions
   - Use gradient checkpointing
   - Enable mixed precision training

4. **Convergence Problems**
   - Adjust learning rates for different manifolds
   - Use proper initialization schemes
   - Monitor gradient norms

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor manifold parameters
for name, param in model.named_parameters():
    if 'manifold' in name:
        print(f"{name}: {param.grad.norm()}")
```

## 📊 Evaluation Metrics

### Anomaly Detection Metrics
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under precision-recall curve
- **F1-Score**: Harmonic mean of precision and recall
- **Precision@K**: Precision at top-K predictions

### Reconstruction Metrics
- **Curvature MSE**: Mean squared error for curvature reconstruction
- **Structure Loss**: Cross-entropy for adjacency reconstruction
- **Feature Loss**: MSE for attribute reconstruction

### Geometric Metrics
- **Manifold Consistency**: Measure of embedding quality
- **Curvature Preservation**: How well curvature is preserved
- **Geometric Distortion**: Distortion in manifold embeddings

For detailed implementation questions, refer to the paper or open an issue on GitHub. 