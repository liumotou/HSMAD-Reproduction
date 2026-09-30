# models/curvgad_detector.py

from models.gnn import *
from models.curvgad_gnn import CurvGAD  # Import your custom GNN
from models.detector import BaseDetector
import geoopt
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.sparse import load_npz
import os
import dgl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class CurvGADDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        self.use_autoencoder = train_config.get('use_autoencoder', False)
        self.device = train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize trade-off parameters (as before)
        self.a_raw = nn.Parameter(torch.tensor(0.3, device=self.device))
        self.b_raw = nn.Parameter(torch.tensor(0.4, device=self.device))
        self.c_raw = nn.Parameter(torch.tensor(0.2, device=self.device))
        self.d_raw = nn.Parameter(torch.tensor(0.1, device=self.device))
        self.beta = nn.Parameter(torch.tensor(0.25, device=self.device))

        # Parse manifold configuration for the first model (e.g., 'H16E16S16')
        manifold_config_str = model_config.get('manifolds_config_str', 'H16E16S16')
        manifolds_config = self.parse_manifold_config(manifold_config_str)
        total_dim = sum([dim for _, dim in manifolds_config])

        # Create first model for curvature reconstruction
        self.model_curv = CurvGAD(**model_config).to(self.device)

        # Create second model for adjacency and feature reconstruction
        # Use 'E{total_dim}' as manifold configuration for Euclidean space
        manifold_config_str_euclidean = f'E{total_dim}'
        model_config_euclidean = model_config.copy()
        model_config_euclidean['manifolds_config_str'] = manifold_config_str_euclidean
        self.model_adj_feat = CurvGAD(**model_config_euclidean).to(self.device)

        # Combine parameters from both models
        if train_config['optimizer'] == 'riemannian_adam':
            self.optimizer = torch.optim.Adam(
                [{'params': self.model_curv.parameters()},
                {'params': self.model_adj_feat.parameters()},
                {'params': [self.a_raw, self.b_raw, self.c_raw, self.d_raw, self.beta]}],
                lr=model_config['lr']
            )
        else:
            self.optimizer = geoopt.optim.RiemannianAdam(
                [{'params': self.model_curv.parameters()},
                {'params': self.model_adj_feat.parameters()},
                {'params': [self.a_raw, self.b_raw, self.c_raw, self.d_raw, self.beta]}],
                lr=model_config['lr']
            )
        # Load curvature data and Ricci flow weights if autoencoder is used
        if self.use_autoencoder:
            self.load_curvature_data(train_config)
            self.load_ricci_flow_weights(train_config)

    def parse_manifold_config(self, manifold_config_str):
        """
        Parses the manifold configuration string and returns a list of tuples (manifold_type, dimension).
        Example input: "H16E16S16"
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

    def load_curvature_data(self, train_config):
        dataset_name = train_config['dataset_name']
        self.dataset_name = train_config['dataset_name']
        search = dataset_name.find("_")
        if search ==-1:
            curvature_file = os.path.join('curvature_data', f"{dataset_name}_curvature_matrix.npz")
        else:
            curvature_file = os.path.join('curvature_data', f"{dataset_name[:search]}_curvature_matrix.npz")
        if os.path.exists(curvature_file):
            curvature_matrix = load_npz(curvature_file).tocoo()

            # Convert to tensors
            self.edge_curvature_values = torch.tensor(curvature_matrix.data, dtype=torch.float32, device=self.device)
            self.edge_indices = torch.tensor(
                np.vstack((curvature_matrix.row, curvature_matrix.col)),
                dtype=torch.long, device=self.device
            )
            num_nodes = curvature_matrix.shape[0]
            self.num_nodes = num_nodes

            # Create sparse matrix for ground truth curvature
            self.curvature_gt_sparse = torch.sparse_coo_tensor(
                self.edge_indices, self.edge_curvature_values, (num_nodes, num_nodes)
            ).coalesce()

            # Compute true node curvatures using sparse operations
            self.true_node_curvatures = self.compute_node_curvatures(
                self.curvature_gt_sparse, self.edge_indices, num_nodes
            )
        else:
            raise FileNotFoundError(f"Curvature file {curvature_file} not found.")
        
    def load_ricci_flow_weights(self, train_config):
        dataset_name = train_config['dataset_name']
        ricci_flow_weights_file = os.path.join('ricci_flow_data', f"{dataset_name}_ricci_flow_adj_matrix.npz")
        if os.path.exists(ricci_flow_weights_file) and train_config['use_ricci_flow_weights']:
            print("Using Ricci flow matrix")
            ricci_flow_weights_matrix = load_npz(ricci_flow_weights_file).tocoo()
            # ricci_flow_weights_matrix.data = np.where(ricci_flow_weights_matrix.data > 0, 1, ricci_flow_weights_matrix.data)
            self.rf_edge_weights = torch.tensor(ricci_flow_weights_matrix.data, dtype=torch.float32, device=self.device)
            self.rf_edge_indices = torch.tensor(
                np.vstack((ricci_flow_weights_matrix.row, ricci_flow_weights_matrix.col)),
                dtype=torch.long, device=self.device
            )
            num_nodes = ricci_flow_weights_matrix.shape[0]
        else:
            num_nodes = self.data.graph.num_nodes()
            edge_indices = self.data.graph.edges()
            # Ensure that the edge indices are on the correct device and are long tensors
            self.rf_edge_indices = torch.stack(edge_indices).long().to(self.device)
            self.rf_edge_weights = torch.ones(self.rf_edge_indices[0].shape[0], dtype=torch.float32, device=self.device)
        
        self.rf_num_nodes = num_nodes
        # Create DGL graph with Ricci flow weights
        self.rf_graph = dgl.graph((self.rf_edge_indices[0], self.rf_edge_indices[1]), num_nodes=num_nodes)
        self.rf_graph = self.rf_graph.to(self.device)  # Ensure the graph is on the correct device
        self.rf_graph.edata['weight'] = self.rf_edge_weights
        # Copy node features and move them to the same device as the graph
        self.rf_graph.ndata['feature'] = self.data.graph.ndata['feature'].to(self.device)

        # **Define the ground truth adjacency matrix using the Ricci flow graph**
        # Create sparse adjacency matrix for ground truth adjacency
        self.adj_gt_sparse = torch.sparse_coo_tensor(
            self.rf_edge_indices, self.rf_edge_weights, (num_nodes, num_nodes)
        ).coalesce()

        # **Define the ground truth features**
        self.features_gt = self.rf_graph.ndata['feature']

        # Convert features to NumPy array
        features_np = self.features_gt.cpu().numpy()

        # Fit the scaler on the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_np)

        # Update the features in the graph
        self.rf_graph.ndata['feature'] = torch.tensor(features_scaled, dtype=torch.float32, device=self.features_gt.device)

        # Also update the ground truth features used for reconstruction
        self.features_gt = self.rf_graph.ndata['feature']

        # else:
        #     raise FileNotFoundError(f"Ricci flow weights file {ricci_flow_weights_file} not found.")
    def compute_node_anomaly_from_adj(self, adj_pred_sparse, adj_gt_sparse, edge_indices, num_nodes):
        """
        Compute node-level anomaly scores based on adjacency matrix reconstruction.

        Parameters:
        - adj_pred_sparse: Sparse tensor of predicted adjacency values.
        - adj_gt_sparse: Sparse tensor of ground truth adjacency values.
        - edge_indices: Tensor of shape [2, num_edges], containing source and target node indices for each edge.
        - num_nodes: Total number of nodes in the graph.

        Returns:
        - node_anomaly_scores: Tensor of node-level anomaly scores based on adjacency matrix reconstruction.
        """
        # Ensure both sparse matrices are on the correct device
        adj_pred_sparse = adj_pred_sparse.to(self.device)
        adj_gt_sparse = adj_gt_sparse.to(self.device)

        # Convert sparse matrices to dense for easier node-wise computation
        adj_pred_dense = adj_pred_sparse.to_dense()
        adj_gt_dense = adj_gt_sparse.to_dense()

        # Compute the squared difference between predicted and ground truth adjacency matrices
        adj_diff = (adj_pred_dense - adj_gt_dense) ** 2

        # Compute node anomaly scores by summing differences across rows (i.e., summing differences for all edges connected to each node)
        node_anomaly_scores = torch.sum(adj_diff, dim=1)

        # Normalize node anomaly scores by the degree (number of neighbors)
        node_degrees = torch.bincount(edge_indices.flatten(), minlength=num_nodes).float().clamp(min=1)

        # Divide by the degree to get average anomaly score per node
        node_anomaly_scores = node_anomaly_scores / node_degrees

        return node_anomaly_scores

    def normalize_anomaly_score(self, anomaly_scores):
        """
        Normalize the anomaly scores to a [0, 1] range.
        
        Parameters:
        - anomaly_scores: Tensor of anomaly scores.

        Returns:
        - normalized_scores: Tensor of normalized anomaly scores.
        """
        anomaly_min = anomaly_scores.min()
        anomaly_max = anomaly_scores.max()
        
        # Avoid division by zero in normalization
        if anomaly_max - anomaly_min == 0:
            return anomaly_scores
        
        normalized_scores = (anomaly_scores - anomaly_min) / (anomaly_max - anomaly_min)
        return normalized_scores
    
    def compute_node_curvatures(self, curvature_sparse, edge_indices, num_nodes):
        """
        Compute node curvatures by averaging the edge curvatures for each node using sparse operations.
        """
        node_curvature_sums = torch.sparse.sum(curvature_sparse, dim=1).to_dense()
        src = edge_indices[0]
        dst = edge_indices[1]
        all_nodes = torch.cat([src, dst], dim=0)
        node_degrees = torch.bincount(all_nodes, minlength=num_nodes).float()
        node_degrees = node_degrees.clamp(min=1)
        node_curvatures = node_curvature_sums / node_degrees
        return node_curvatures
    
    def compute_combined_anomaly_scores(self, predicted_node_curvatures, adj_pred_sparse, x_pred, num_nodes, return_parts = False):
        """
        Compute the combined anomaly scores using curvature, adjacency, and feature anomalies.

        Parameters:
        - predicted_node_curvatures: Tensor of predicted node curvature values.
        - adj_pred_sparse: Sparse tensor of predicted adjacency matrix.
        - x_pred: Tensor of predicted node features.
        - num_nodes: Total number of nodes in the graph.

        Returns:
        - anomaly_scores_train: Tensor of combined anomaly scores for the training data.
        """
        # Curvature-based anomaly scores (node-wise MSE)
        anomaly_scores_curv = torch.norm(predicted_node_curvatures - self.true_node_curvatures, p='fro')
        anomaly_scores_curv = self.normalize_anomaly_score(anomaly_scores_curv)

        # Adjacency-based anomaly scores
        anomaly_scores_adj = self.compute_node_anomaly_from_adj(
            adj_pred_sparse, self.adj_gt_sparse, self.edge_indices, num_nodes
        )
        anomaly_scores_adj = self.normalize_anomaly_score(anomaly_scores_adj)

        # Feature-based anomaly scores (node-wise MSE)
        anomaly_scores_feat = self.compute_node_anomaly_from_features(
            x_pred, self.features_gt, num_nodes
        )
        anomaly_scores_feat = self.normalize_anomaly_score(anomaly_scores_feat)

        # Combine the anomaly scores using the learnable weights a, b, and c
        anomaly_scores_train =(
            self.a * anomaly_scores_curv + 
            self.b * anomaly_scores_adj + 
            self.c * anomaly_scores_feat
        )
        if return_parts: 
            no_curvature = self.b * anomaly_scores_adj + self.c * anomaly_scores_feat
            return no_curvature, no_curvature + self.a * anomaly_scores_curv

        return anomaly_scores_train

    def compute_node_anomaly_from_features(self, x_pred, feature_gt, num_nodes):
        """
        Compute node-level anomaly scores based on the feature matrix reconstruction.
        
        Parameters:
        - x_pred: Tensor of predicted node features.
        - feature_gt: Tensor of ground truth node features.
        - num_nodes: Total number of nodes in the graph.

        Returns:
        - node_anomaly_scores: Tensor of node-level anomaly scores based on feature matrix reconstruction.
        """
        # Compute the squared difference between predicted and true features
        feature_diff = (x_pred - feature_gt) ** 2

        # Sum the squared differences across feature dimensions for each node
        node_anomaly_scores = torch.sum(feature_diff, dim=1)

        return node_anomaly_scores

    @property
    def a(self):
        """Apply sigmoid to raw a to restrict it to (0, 1)."""
        return torch.sigmoid(self.a_raw)

    @property
    def b(self):
        """Apply sigmoid to raw b to restrict it to (0, 1)."""
        return torch.sigmoid(self.b_raw)

    @property
    def c(self):
        """Apply sigmoid to raw c to restrict it to (0, 1)."""
        return torch.sigmoid(self.c_raw)

    @property
    def d(self):
        """Apply sigmoid to raw c to restrict it to (0, 1)."""
        return torch.sigmoid(self.d_raw)
    # Normalization function using standardization

    
    def frobenius_norm_sparse(self, sparse_tensor):
        """
        Compute the Frobenius norm for a sparse tensor.
        This is equivalent to sqrt(sum(square(values of the sparse tensor))).
        """
        # Ensure the sparse tensor is coalesced (i.e., remove duplicate entries and sum their values)
        sparse_tensor = sparse_tensor.coalesce()
        
        # Extract the values of the sparse tensor and compute the Frobenius norm
        return torch.sqrt(torch.sum(sparse_tensor._values() ** 2))
    

    def train(self):
        train_labels = self.labels[self.train_mask].to(self.device)
        val_labels = self.labels[self.val_mask].to(self.device)
        test_labels = self.labels[self.test_mask].to(self.device)

        best_val_score = float('-inf')
        patience_counter = 0
        test_score = {}

        for e in range(self.train_config['epochs']):
            self.model_curv.train()
            self.model_adj_feat.train()

            # Forward pass for curvature model
            logits_curv = self.model_curv(self.train_graph)
            h_curv = self.model_curv.get_embeddings()

            # Forward pass for adjacency and feature model using Ricci flow graph
            logits_adj_feat = self.model_adj_feat(self.rf_graph)
            h_adj_feat = self.model_adj_feat.get_embeddings()

            if self.use_autoencoder:
                # Curvature reconstruction
                curvature_pred_sparse, predicted_node_curvatures = self.model_curv.decode_curvature(h_curv, self.edge_indices)

                # Adjacency and feature reconstruction
                adj_pred_sparse = self.model_adj_feat.decode_adj(h_adj_feat, self.rf_edge_indices)
                features_pred = self.model_adj_feat.decode_features(h_adj_feat)

                # Compute curvature loss
                C_loss = self.frobenius_norm_sparse(curvature_pred_sparse - self.curvature_gt_sparse)

                # Compute adjacency loss
                A_loss = self.frobenius_norm_sparse(adj_pred_sparse - self.adj_gt_sparse)

                # Compute feature reconstruction loss
                # print(features_pred, self.features_gt)
                X_loss = torch.linalg.matrix_norm(features_pred -  self.features_gt)

                # Total reconstruction loss (weighted)
                total_rec_loss = self.a * C_loss + self.b * A_loss + self.c * X_loss

                # --- Anomaly score computation ---
                # Compute combined anomaly scores
                anomaly_scores_train = self.compute_combined_anomaly_scores(
                    predicted_node_curvatures, adj_pred_sparse, features_pred, self.num_nodes
                )

                # print(logits_curv.shape, anomaly_scores_train.shape)

                # Combine classification logits and anomaly scores using beta
                logits = self.beta * logits_curv + (1 - self.beta) * anomaly_scores_train.unsqueeze(-1)
                # print(train_labels)
                # Classification loss
                # print(logits)
                loss_cls = F.cross_entropy(logits[self.train_mask], train_labels)

                # Final loss
                loss = self.d * loss_cls + (1 - self.d) * total_rec_loss * 0.0001
                # loss = total_rec_loss
            else:
                # Classification loss (logits-only, no autoencoder)
                logits = self.model_curv(self.train_graph)
                
                loss_cls = F.cross_entropy(
                    logits[self.train_mask], train_labels,
                    weight=torch.tensor([1., self.weight], device=self.device)
                )
                loss = loss_cls

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model_curv.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.model_adj_feat.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Validation and early stopping
            self.model_curv.eval()
            self.model_adj_feat.eval()
            with torch.no_grad():
                # Validation pass
                logits_val_curv = self.model_curv(self.val_graph)
                h_val_curv = self.model_curv.get_embeddings()
                logits_val_adj_feat = self.model_adj_feat(self.rf_graph)
                h_val_adj_feat = self.model_adj_feat.get_embeddings()

                if self.use_autoencoder:
                    # Curvature reconstruction for validation
                    _, predicted_node_curvatures_val = self.model_curv.decode_curvature(h_val_curv, self.edge_indices)

                    # Adjacency and feature reconstruction for validation
                    adj_pred_sparse_val = self.model_adj_feat.decode_adj(h_val_adj_feat, self.rf_edge_indices)
                    features_pred_val = self.model_adj_feat.decode_features(h_val_adj_feat)

                    # Compute combined anomaly scores for validation
                    anomaly_scores_val = self.compute_combined_anomaly_scores(
                        predicted_node_curvatures_val, adj_pred_sparse_val, features_pred_val, self.num_nodes
                    )

                    # Combine logits and anomaly scores using beta
                    logits_val = self.beta * logits_val_curv + (1 - self.beta) * anomaly_scores_val.unsqueeze(-1)
                else:
                    logits_val = logits_val_curv

                # Apply softmax
                probs_val = logits_val.softmax(dim=1)[:, 1]

                # Evaluate
                val_score = self.eval(val_labels, probs_val[self.val_mask])
                current_val_metric = val_score[self.train_config['metric']]

                if current_val_metric > best_val_score:
                    best_val_score = current_val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1

                    if patience_counter > self.train_config['patience']:
                        print(f'Early stopping at epoch {e}')
            
                        break

                # Test evaluation
                logits_test_curv = self.model_curv(self.source_graph)
                h_test_curv = self.model_curv.get_embeddings()
                logits_test_adj_feat = self.model_adj_feat(self.rf_graph)
                h_test_adj_feat = self.model_adj_feat.get_embeddings()

                if self.use_autoencoder:
                    # Curvature reconstruction for testing
                    _, predicted_node_curvatures_test = self.model_curv.decode_curvature(h_test_curv, self.edge_indices)

                    # Adjacency and feature reconstruction for testing
                    adj_pred_sparse_test = self.model_adj_feat.decode_adj(h_test_adj_feat, self.rf_edge_indices)
                    features_pred_test = self.model_adj_feat.decode_features(h_test_adj_feat)

                    # Compute combined anomaly scores for testing
                    anomaly_scores_test = self.compute_combined_anomaly_scores(
                        predicted_node_curvatures_test, adj_pred_sparse_test, features_pred_test, self.num_nodes
                    )

                    # Combine the logits and anomaly scores using beta
                    logits_test = self.beta * logits_test_curv + (1 - self.beta) * anomaly_scores_test.unsqueeze(-1)
                else:
                    logits_test = logits_test_curv

                # Apply softmax to test logits
                test_probs = logits_test.softmax(dim=1)[:, 1]

                # Evaluate on test set
                test_score = self.eval(test_labels, test_probs[self.test_mask])

                print(f'Epoch {e}, Loss {loss.item():.4f}, Val {self.train_config["metric"]} {current_val_metric:.4f}, Test {self.train_config["metric"]} {test_score[self.train_config["metric"]]:.4f}')



                # Inside your training loop
                print(f"Epoch {e}, Total Loss {loss.item():.4f}, "
                    f"Loss_cls {loss_cls.item():.4f}, "
                    f"C_loss {C_loss.item():.4f}, "
                    f"A_loss {A_loss.item():.4f}, "
                    f"X_loss {X_loss.item():.4f}, "
                    f"a: {self.a.item():.2f}, b: {self.b.item():.2f}, c: {self.c.item():.2f}, d: {self.d.item():.4f}")

        logits_curv = self.model_curv(self.train_graph)
        h_curv = self.model_curv.get_embeddings()
        h_adj_feat = self.model_adj_feat.get_embeddings()
        curvature_pred_sparse, predicted_node_curvatures = self.model_curv.decode_curvature(h_curv, self.edge_indices)
        adj_pred_sparse = self.model_adj_feat.decode_adj(h_adj_feat, self.rf_edge_indices)
        features_pred = self.model_adj_feat.decode_features(h_adj_feat)
        anomaly_scores_no_curv, anomaly_scores_with_curv = self.compute_combined_anomaly_scores(
            predicted_node_curvatures, adj_pred_sparse, features_pred, self.num_nodes, return_parts=True
        )
        logits = self.beta * logits_curv + (1 - self.beta) * anomaly_scores_with_curv.unsqueeze(-1)

        self.generate_plots(anomaly_scores_no_curv.cpu().detach().numpy(), 
                            anomaly_scores_with_curv.cpu().detach().numpy(), 
                            self.true_node_curvatures.cpu().detach().numpy(), 
                            train_labels.cpu().detach().numpy(), 
                            self.train_mask.cpu().detach().numpy())

        return test_score


    def generate_plots(self, anomaly_scores_no_curv, anomaly_scores_with_curv, orc_values, labels, train_mask):
        # Normalize the anomaly scores for better comparison
        min_no_curv = min(anomaly_scores_no_curv)
        max_no_curv = max(anomaly_scores_no_curv)
        min_with_curv = min(anomaly_scores_with_curv)
        max_with_curv = max(anomaly_scores_with_curv)

        anomaly_scores_no_curv = (anomaly_scores_no_curv - min_no_curv) / (max_no_curv - min_no_curv)
        anomaly_scores_with_curv = (anomaly_scores_with_curv - min_with_curv) / (max_with_curv - min_with_curv)

        # Create a figure with two subplots side by side for the x-axis break
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 6), gridspec_kw={'width_ratios': [1, 5]})
        fig.subplots_adjust(wspace=None)  # adjust space between Axes

        # Plot anomaly scores without curvature in ax1
        l1 = ax1.scatter(
            np.zeros_like(anomaly_scores_no_curv[train_mask][labels == 0]),  # X-axis as 0
            anomaly_scores_no_curv[train_mask][labels == 0],
            c='blue', alpha=0.2, label='Normal (w/o Curv)', marker='^'
        )
        l2 = ax1.scatter(
            np.zeros_like(anomaly_scores_no_curv[train_mask][labels == 1]),  # X-axis as 0
            anomaly_scores_no_curv[train_mask][labels == 1],
            c='red', alpha=0.2, label='Anomalous (w/o Curv)', marker='^'
        )

        ax1.set_xlim(left=-0.1, right=0.1)  # x-axis limits for the "no curvature" part
        ax1.set_ylabel('Anomaly Score for Nodes')
        ax1.set_xticks([0])
        ax1.set_xticklabels(['W/o Curv'])
        ax1.grid(True)
        # ax1.set_title('Anomaly Scores Without Curvature')
        # ax1.legend()

        # Add a label for the break
        # ax1.text(0, -0.05, 'No Curv', va='center', ha='center', rotation=90, fontsize=10, color='black')

        # Plot anomaly scores with curvature in ax2
        l3 = ax2.scatter(
            orc_values[train_mask][labels == 0], anomaly_scores_with_curv[train_mask][labels == 0],
            c='blue', alpha=0.2, label='Normal (w/ Curv)', marker='o'
        )
        l4 = ax2.scatter(
            orc_values[train_mask][labels == 1], anomaly_scores_with_curv[train_mask][labels == 1],
            c='red', alpha=0.2, label='Anomalous (w/ Curv)', marker='o'
        )

        l5 = ax2.axvline(x=0, color='green', linestyle='--', linewidth=2, label='ORC = 0')
        ax2.set_xlim(left=min(orc_values)-0.1, right=max(orc_values)+0.1)  # x-axis for ORC values
        # ax2.set_xlabel('Ollivier-Ricci Curvature')
        # ax2.set_title(f'Anomaly Scores vs Ollivier-Ricci Curvature for {self.dataset_name}')
        # ax2.legend()

        # Add the "x-axis break" visual
        ax1.spines.right.set_visible(False)
        ax2.spines.left.set_visible(False)
        ax1.yaxis.tick_left()
        ax2.yaxis.tick_right()

        # Add diagonal lines to represent the break on the x-axis
        d = 0.02  # size of the diagonal lines in axes coordinates
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        # ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
        # ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the right axes
        ax2.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        # Set the y-axis limits based on the min and max of both anomaly scores
        ax1.set_ylim(
            bottom=min(min(anomaly_scores_with_curv), min(anomaly_scores_no_curv)) - 0.01,
            top=max(max(anomaly_scores_with_curv), max(anomaly_scores_no_curv)) + 0.01
        )

        fig.legend([l1, l2, l3, l4, l5], labels= ['Normal (w/o Curv)', 'Anomaly (w/o Curv)', 'Normal (w/ Curv)', 'Anomaly (w/ Curv)', 'ORC = 0']) 
        ax2.grid(True)
        # Save the plot
        fig.suptitle(f'{self.dataset_name}'.capitalize())
        fig.supxlabel('Ollivier-Ricci Curvature of Nodes')
        plt.tight_layout()
        plt.savefig(f'plots/{self.dataset_name}_anomaly_scores_vs_curvature.png', dpi=300)
        plt.show()