import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
import dgl.nn as dglnn
import sympy
import scipy
import numpy as np
from torch.nn import init
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import roc_auc_score
from manifold_update import WeightedMessagePassing
def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas

class PolyConv(nn.Module):
    def __init__(self, in_feats, out_feats, theta):
        super().__init__()
        self.theta = theta
        self.K = len(theta)
        self.in_feats = in_feats
        self.out_feats = out_feats

    def forward(self, graph, feat, edge_weight=None, edge_mask=None):
        with graph.local_scope():

            if edge_weight is None and edge_mask is None:
                w = None
            else:
                if edge_weight is None:
                    w = edge_mask.float()
                elif edge_mask is None:
                    w = edge_weight
                else:
                    w = edge_weight * edge_mask.float()

                graph.edata['w'] = w

                # 基于 w 计算度
                src, dst = graph.edges()
                deg = torch.zeros(graph.num_nodes(), device=feat.device)
                deg.index_add_(0, dst, w)
                deg = deg.clamp(min=1)
                D_invsqrt = deg.pow(-0.5).unsqueeze(-1)

            if w is None:
                deg = graph.in_degrees().float().clamp(min=1)
                D_invsqrt = deg.pow(-0.5).unsqueeze(-1)
                weighted_edges = None
            else:
                weighted_edges = 'w'

            h = self.theta[0] * feat
            Xk = feat

            for k in range(1, self.K):
                graph.ndata['h'] = Xk * D_invsqrt

                if weighted_edges is None:
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                else:
                    graph.update_all(fn.u_mul_e('h', weighted_edges, 'm'), fn.sum('m', 'h'))

                Xk = Xk - graph.ndata.pop('h') * D_invsqrt
                h = h + self.theta[k] * Xk

            return h

 
class EdgePartitioner(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EdgePartitioner, self).__init__()
        self.norm = nn.LayerNorm(input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_feat):
        edge_feat = self.norm(edge_feat)
        edge_pred = self.classifier(edge_feat)
        return edge_pred.squeeze()


class WeightedLapModule(nn.Module):
    def __init__(self, in_feats, h_feats, graph, quantile, d):
        super().__init__()
        self.quantile = quantile
        self.thetas = calculate_theta2(d=d)
        self.convs = nn.ModuleList([PolyConv(in_feats, h_feats, theta) for theta in self.thetas]).to(graph.device)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.norm1 = nn.LayerNorm(h_feats)
        self.norm2 = nn.LayerNorm(h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.convs), h_feats)
        self.norm3 = nn.LayerNorm(h_feats)

        self.norm_out = nn.LayerNorm(h_feats)
        self.log_alpha_dict = nn.ParameterDict({
            etype: nn.Parameter(torch.tensor(0.0)) for etype in graph.etypes
        })
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.rho = nn.Parameter(torch.tensor(0.5))

    def apply_branch(self, feat):
        h = self.linear1(feat)
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.linear2(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        return h

    def apply_convs(self, graph, h, convs, edge_weight=None, edge_mask=None):

        outputs = []

        for i, conv in enumerate(convs):
            h = conv(graph, h, edge_weight, edge_mask)
            outputs.append(h)

        return torch.cat(outputs, dim=-1)

    def forward(self, g, feat, edge_preds):
        alpha = torch.exp(self.log_alpha)
        h = self.apply_branch(feat)
        for etype, edge_pred in edge_preds.items():
            graph = g[etype]
            edge_hetero = torch.sigmoid(edge_pred)
            if len(edge_pred) > 4e7:
                sample_size = int(1e7)
                idx = torch.randperm(len(edge_hetero))[:sample_size]
                sampled_edge = edge_hetero[idx]
                threshold = torch.quantile(sampled_edge, self.quantile)
            else:
                threshold = torch.quantile(edge_hetero, self.quantile)

            low_mask = edge_hetero < threshold
            high_mask = edge_hetero >= threshold
            w  = torch.exp(-alpha * edge_hetero)

            h_low = self.apply_convs(graph, h, self.convs, edge_weight=w, edge_mask=low_mask)
            h_high = self.apply_convs(graph, h, self.convs, edge_weight=w, edge_mask=high_mask)

        rho = torch.sigmoid(self.rho)
        h_final = rho * h_low + (1 - rho) * h_high

        combined_feat = self.linear3(h_final)
        combined_feat = self.norm3(combined_feat)
        combined_feat = self.act(combined_feat)
        combined_feat = self.dropout(combined_feat)

        return combined_feat
    
class CombinedModel(nn.Module):
    VALID_ABLATIONS = {"full", "no_hwsf", "no_hrmu"}

    def __init__(self, graph, in_feats, hidden_feats, out_feats, d, quantile=0.3,
                 dropout=0.5, ablation="full"):
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation mode: {ablation}. "
                f"Expected one of {sorted(self.VALID_ABLATIONS)}."
            )
        self.graph = graph
        self.ablation = ablation
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.hidden_feats = hidden_feats
        self.edge_classifiers = nn.ModuleDict()  
        for etype in graph.etypes: 
            self.edge_classifiers[etype] = EdgePartitioner(in_feats, hidden_feats)

        # Keep the edge partitioner in every setting because both representation
        # branches use its heterophily scores and its loss remains supervised.
        self.curv = (
            WeightedMessagePassing(in_feats, hidden_feats, graph)
            if ablation != "no_hrmu" else None
        )
        self.graph_partition_module = (
            WeightedLapModule(in_feats, hidden_feats, graph, quantile, d)
            if ablation != "no_hwsf" else None
        )
        self.unique_u_dict = {}
        self.unique_v_dict = {}
        self.inverse_idx_dict = {}
        if ablation == "full":
            self.gate = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_parameter("gate", None)
        for etype in graph.etypes:
            src, dst = graph.edges(etype=etype)
            min_idx = torch.min(src, dst)
            max_idx = torch.max(src, dst)
            undirected_keys = torch.stack([min_idx, max_idx], dim=1)

            unique_keys, inverse_idx = torch.unique(undirected_keys, dim=0, return_inverse=True)
            u, v = unique_keys[:, 0], unique_keys[:, 1]

            self.unique_u_dict[etype] = u
            self.unique_v_dict[etype] = v
            self.inverse_idx_dict[etype] = inverse_idx
            
        
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_feats),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, features):
        edge_preds = {}

        for etype in self.graph.etypes:
            u = self.unique_u_dict[etype]
            v = self.unique_v_dict[etype]
            inverse_idx = self.inverse_idx_dict[etype]
            h_u = features[u]
            h_v = features[v]

            edge_feat_unique = torch.abs(h_u - h_v)
            edge_pred_unique = self.edge_classifiers[etype](edge_feat_unique)
            edge_pred_full = edge_pred_unique[inverse_idx]

            edge_preds[etype] = edge_pred_full

        if self.ablation == "full":
            h_curv = self.curv(self.graph, features, edge_preds)
            h_spec = self.graph_partition_module(self.graph, features, edge_preds)
            gate = torch.sigmoid(self.gate)
            combined = gate * h_curv + (1 - gate) * h_spec
        elif self.ablation == "no_hwsf":
            combined = self.curv(self.graph, features, edge_preds)
        else:  # no_hrmu
            combined = self.graph_partition_module(self.graph, features, edge_preds)

        output = self.fusion(combined)

        return output.log_softmax(dim=-1), edge_preds
    
    def compute_loss(self, output, labels, edge_labels, edge_preds, node_mask):
        device = output.device

        masked_output = output[node_mask]
        masked_labels = labels[node_mask]

        binary_mask = (masked_labels == 0) | (masked_labels == 1)

        masked_output = masked_output[binary_mask]
        masked_labels = masked_labels[binary_mask]

        node_loss = F.nll_loss(masked_output, masked_labels)

        edge_masks = {}
        for etype in self.graph.etypes:
            src, dst = self.graph.edges(etype=etype)
            src, dst = src.to(device), dst.to(device)

            label_mask_src = (labels[src] == 0) | (labels[src] == 1)
            label_mask_dst = (labels[dst] == 0) | (labels[dst] == 1)

            edge_mask = node_mask[src] & node_mask[dst] & label_mask_src & label_mask_dst
            edge_masks[etype] = edge_mask

        def calculate_edge_loss(edge_pred, edge_mask, edge_label):
            masked_edge_labels = edge_label[edge_mask].float()

            num_edges = masked_edge_labels.numel()
            pos_count = (masked_edge_labels == 1).sum().item()
            neg_count = (masked_edge_labels == 0).sum().item()

            if pos_count == 0 or neg_count == 0:
                pos_weight = neg_weight = 1.0
            else:
                total = pos_count + neg_count
                pos_weight = total / (2.0 * pos_count)
                neg_weight = total / (2.0 * neg_count)

            sample_weights = torch.where(
                masked_edge_labels == 1,
                torch.tensor(pos_weight, device=edge_pred.device),
                torch.tensor(neg_weight, device=edge_pred.device)
            )

            loss = F.binary_cross_entropy_with_logits(
                edge_pred[edge_mask],
                masked_edge_labels,
                weight=sample_weights,
                reduction='mean'
            )

            stats = {
                "num_edges": num_edges,
                "pos": pos_count,
                "neg": neg_count,
                "pos_ratio": pos_count / (num_edges + 1e-6)
            }

            return loss, stats

        edge_losses = []
        edge_stats = {}  

        if isinstance(edge_masks, torch.Tensor):
            for etype, edge_pred in edge_preds.items():
                edge_mask = edge_masks
                edge_label = edge_labels

                e_loss, stats = calculate_edge_loss(edge_pred, edge_mask, edge_label)
                edge_losses.append(e_loss)
                edge_stats[etype] = stats
        else:
            for etype in self.graph.etypes:
                edge_pred = edge_preds[etype]
                edge_mask = edge_masks[etype]
                edge_label = edge_labels[etype]

                e_loss, stats = calculate_edge_loss(edge_pred, edge_mask, edge_label)
                edge_losses.append(e_loss)
                edge_stats[etype] = stats

        edge_loss = torch.stack(edge_losses).mean()

        eps = 1e-8
        w_node = (edge_loss / (node_loss + edge_loss + eps)).detach()
        w_edge = (node_loss / (node_loss + edge_loss + eps)).detach()

        total_loss = w_node * node_loss + w_edge * edge_loss

        return total_loss
