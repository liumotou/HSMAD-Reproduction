# models/curvgad_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt import ManifoldParameter
from models.manifold_cheb_conv import ManifoldChebConv
import torch_geometric as tg
from torch_geometric.utils import get_laplacian, add_self_loops

class FeatAdjDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeatAdjDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x
    
class CurvGAD(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_filters=3,
                 K=3, manifolds_config_str="H8S8E8", dropout_rate=0.5,
                 activation='ReLU', gamma=1.0, **kwargs):
        super(CurvGAD, self).__init__()
        self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate)
        self.num_filters = num_filters
        self.K = K  # Chebyshev polynomial order
        self.num_classes = num_classes
        self.gamma = gamma
    

        # Parse manifolds configuration
        manifolds_config = self.parse_manifold_config(manifolds_config_str)
        manifolds = []
        for manifold_type, dim in manifolds_config:
            if manifold_type == 'hyperbolic':
                manifold = geoopt.PoincareBall(c=1.0, learnable=True)
            elif manifold_type == 'spherical':
                manifold = geoopt.SphereProjection(k=1.0, learnable=True)
            elif manifold_type == 'euclidean':
                manifold = geoopt.Stereographic(k=0.0, learnable=False)
            else:
                raise ValueError(f"Unknown manifold type: {manifold_type}")
            manifolds.append((manifold, dim))

        self.manifold = geoopt.StereographicProductManifold(*manifolds)
        self.total_dim = sum([dim for _, dim in manifolds_config])

        # Inside the CurvGAD class __init__ method
        # Define gamma as a hyperparameter (you can set its value or make it configurable)
        self.gamma = 1.0  # or any value you choose

        # Initial Euclidean linear transformation
        self.fc_in = nn.Linear(in_feats, self.total_dim)

        # Create filter bank of ManifoldChebConv layers with K filters
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_filters):
            conv = ManifoldChebConv(self.total_dim, self.total_dim, self.K, self.manifold)
            self.conv_layers.append(conv)

        # Attention parameters for filters, with K+1 entries (K filters + 1 unfiltered)
        self.filter_attention = nn.Parameter(torch.Tensor(self.num_filters + 1))
        nn.init.uniform_(self.filter_attention)

        # Linear layer for prediction
        self.fc_out = nn.Linear(self.total_dim, self.num_classes)

        # Initialize a single learnable sigma parameter for the entire product manifold
        self.sigma = nn.Parameter(torch.tensor(1.0))

        # Decoders for adjacency matrix and feature matrix
        self.adj_decoder = FeatAdjDecoder(self.total_dim, 32, self.total_dim)  # Decoder for adjacency matrix
    
        self.feature_decoder = FeatAdjDecoder(self.total_dim, 32, in_feats)  # Decoder for feature matrix

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layer and attention weights
        self.fc_out.reset_parameters()
        nn.init.uniform_(self.filter_attention)
        # Initialize sigma
        self.sigma.data.fill_(1.0)

    def forward(self, graph):
        # Extract node features from DGL graph
        x = graph.ndata['feature']

        # Convert DGL graph to PyG format to compute Laplacian
        edge_index = self.dgl_to_pyg_edge_index(graph)

        # Add self-loops to edge_index
        edge_index, _ = tg.utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalized Laplacian using PyG's get_laplacian function
        laplacian_edge_index, laplacian_edge_weight = get_laplacian(edge_index, normalization='sym')

        # Convert the Laplacian into a dense matrix (if sparse, use .to_dense())
        laplacian = torch.sparse_coo_tensor(laplacian_edge_index, laplacian_edge_weight).to_dense()

        # Initial Euclidean transformation
        x = self.fc_in(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Map x to manifold
        x = self.manifold.expmap0(x)
        x = self.manifold.projx(x)

        # Apply filter bank
        filter_outputs = []

        # Add unfiltered embedding (identity matrix)
        filter_outputs.append(x)  # No filtering, just the input embeddings

        # Apply K Chebyshev filters
        for conv in self.conv_layers:
            h = conv(x, laplacian)
            h = self.manifold.projx(h)
            filter_outputs.append(h)

        # Stack filter outputs and apply attention
        filter_outputs = torch.stack(filter_outputs, dim=0)  # Shape: [K+1, num_nodes, total_dim]
        attn_weights = F.softmax(self.filter_attention, dim=0).view(self.num_filters + 1, 1, 1)
        h = (attn_weights * filter_outputs).sum(dim=0)
        h = self.manifold.projx(h)
        self.h = h

        # Classification in manifold space using a linear layer
        logits = self.compute_logits(h)
        return logits
    
    def get_embeddings(self):
        return self.h  # Returns the stored embeddings

    def compute_logits(self, h):
        # h: Node embeddings in manifold space, shape [num_nodes, total_dim]
        # Use a linear layer to compute the logits directly
        logits = self.fc_out(h)  # Linear transformation to predict logits
        return logits

    def dgl_to_pyg_edge_index(self, graph):
        """
        Convert DGL graph's edge data to PyG's edge_index format.
        """
        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0)  # Shape: [2, num_edges]
        return edge_index  
    

    # models/curvgad_gnn.py

    def decode_curvature(self, h, edge_indices):
        """
        Reconstruct the curvature values for each edge and compute node curvatures using sparse matrices.

        Parameters:
        - h: Node embeddings in manifold space, shape [num_nodes, total_dim]
        - edge_indices: Edge indices, shape [2, num_edges]

        Returns:
        - curvature_pred_sparse: Sparse tensor of predicted edge curvatures
        - node_curvatures: Tensor of predicted node curvatures
        """
        src = edge_indices[0]
        dst = edge_indices[1]

        # Get embeddings for source and target nodes
        h_src = h[src]
        h_dst = h[dst]

        # Compute manifold distances between source and target embeddings
        dist = self.manifold.dist(h_src, h_dst)

        # Apply Gaussian kernel
        kernel_value = torch.exp(-self.gamma * dist ** 2 / (2 * self.sigma ** 2))

        # Predicted curvature using sigmoid transformation
        curvature_pred = 2 * torch.sigmoid(1 - kernel_value) - 1

        # Clamp curvature values to [-1, 1]
        curvature_pred = torch.clamp(curvature_pred, min=-1, max=1)

        num_nodes = h.size(0)

        # Create sparse tensor for predicted edge curvatures
        curvature_pred_sparse = torch.sparse_coo_tensor(
            edge_indices, curvature_pred, (num_nodes, num_nodes)
        ).coalesce()

        # Compute predicted node curvatures using sparse operations
        # Compute predicted node curvatures using sparse operations and edge indices
        node_curvatures = self.compute_node_curvatures(curvature_pred_sparse, edge_indices, num_nodes)

        return curvature_pred_sparse, node_curvatures

    def compute_node_curvatures(self, curvature_sparse, edge_indices, num_nodes):
        """
        Compute node curvatures by averaging the edge curvatures for each node using sparse operations.
        
        Parameters:
        - curvature_sparse: Sparse tensor of predicted or ground truth edge curvatures.
        - edge_indices: Tensor of shape [2, num_edges], containing source and target node indices for each edge.
        - num_nodes: Total number of nodes in the graph.
        
        Returns:
        - node_curvatures: Tensor of node-level curvature values.
        """
        # Sum of curvatures for each node (row-wise sum)
        node_curvature_sums = torch.sparse.sum(curvature_sparse, dim=1).to_dense()

        # Compute node degrees using edge_indices
        src = edge_indices[0]
        dst = edge_indices[1]
        all_nodes = torch.cat([src, dst], dim=0)

        # Each occurrence of a node in all_nodes corresponds to an edge connected to that node
        node_degrees = torch.bincount(all_nodes, minlength=num_nodes).float()

        # Avoid division by zero
        node_degrees = node_degrees.clamp(min=1)

        # Compute average curvature per node
        node_curvatures = node_curvature_sums / node_degrees

        return node_curvatures
    
    
    def compute_node_anomaly_scores(self, curvature_pred_sparse, edge_indices, num_nodes):
        """
        Compute node-level curvature values by averaging the edge curvatures connected to each node.

        Parameters:
        - curvature_pred_sparse: Sparse tensor of predicted curvature values for each edge.
        - edge_indices: Tensor of shape [2, num_edges], where each column represents an edge (source, target).
        - num_nodes: Total number of nodes in the graph.

        Returns:
        - node_curvature_values: Tensor of node-level curvature values.
        """
        # Convert sparse curvature tensor to dense for easier processing
        curvature_dense = curvature_pred_sparse.to_dense()

        # Compute the sum of all edge curvatures for each node (sum of all row curvatures)
        node_curvature_values = torch.sum(curvature_dense, dim=1)

        # Normalize by the number of neighbors (degree) to get the average
        node_degrees = torch.bincount(edge_indices.flatten(), minlength=num_nodes)
        node_curvature_values = node_curvature_values / node_degrees.float().clamp(min=1)

        return node_curvature_values

    def parse_manifold_config(self, manifold_config_str):
        """
        Parses the manifold configuration string and returns a list of tuples (manifold_type, dimension).
        Example input: "H8S8E8"
        """
        import re
        pattern = r'([HSE])(\d+)'
        matches = re.findall(pattern, manifold_config_str)
        manifolds_config = []
        for match in matches:
            manifold_type = {'H': 'hyperbolic', 'S': 'spherical', 'E': 'euclidean'}[match[0]]
            dim = int(match[1])
            manifolds_config.append((manifold_type, dim))
        return manifolds_config
    
    def decode_adj(self, h, edge_indices):
        """
        Reconstruct the adjacency matrix from latent embeddings using a link prediction method.

        Parameters:
        - h: Node embeddings in manifold space, shape [num_nodes, total_dim]
        - edge_indices: Edge indices, shape [2, num_edges]

        Returns:
        - adj_pred_sparse: Sparse tensor of predicted adjacency values (link probabilities)
        """
        src = edge_indices[0]
        dst = edge_indices[1]

        # Get embeddings for source and target nodes
        h_src = h[src]
        h_dst = h[dst]

        # Compute dot product between embeddings to predict link probabilities
        link_pred = torch.sigmoid(torch.sum(h_src * h_dst, dim=1))

        # Create sparse tensor for predicted adjacency
        num_nodes = h.size(0)
        adj_pred_sparse = torch.sparse_coo_tensor(edge_indices, link_pred, (num_nodes, num_nodes))
        return adj_pred_sparse.coalesce()  # Ensure the tensor is coalesced for efficient sparse operations


    def decode_features(self, h):
        """
        Decode the feature matrix from the latent embeddings.
        """
        features_pred = self.feature_decoder(h)  # Use a linear layer for reconstruction
        return features_pred

    def compute_logits(self, h):
        logits = self.fc_out(h)  # Linear transformation to predict logits
        return logits

    def dgl_to_pyg_edge_index(self, graph):
        """
        Convert DGL graph's edge data to PyG's edge_index format.
        """
        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0)  # Shape: [2, num_edges]
        return edge_index