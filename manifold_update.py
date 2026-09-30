import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.utils import expand_as_pair

EPS = 1e-6

def proj(x, K):
    if float(K) == 0:
        return x
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    maxnorm = (1 - 1e-5) / torch.sqrt(torch.abs(K))
    cond = norm > maxnorm
    return torch.where(cond, x / norm * maxnorm, x)


def expmap0(x, K):
    if float(K) == 0:
        return x
    sqrtK = torch.sqrt(torch.abs(K))
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)

    if K < 0:  # hyperbolic
        factor = torch.tanh(sqrtK * norm) / (sqrtK * norm)
    else:      # spherical
        norm = norm.clamp(max=(torch.pi / 2 - EPS) / sqrtK)
        factor = torch.tan(sqrtK * norm) / (sqrtK * norm)

    return proj(factor * x, K)


def logmap0(x, K):
    if float(K) == 0:
        return x
    sqrtK = torch.sqrt(torch.abs(K))
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)

    if K < 0:
        norm = norm.clamp(max=1 - EPS)
        factor = torch.atanh(sqrtK * norm) / (sqrtK * norm)
    else:
        norm = norm.clamp(max=(torch.pi / 2 - EPS) / sqrtK)
        factor = torch.atan(sqrtK * norm) / (sqrtK * norm)

    return factor * x


def kappa_add(x, y, K):
    if float(K) == 0:
        return x + y

    xy = (x * y).sum(dim=-1, keepdim=True)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)

    denom = 1 - 2 * K * xy + K * K * x2 * y2
    denom = denom.clamp_min(EPS)

    num = (1 - 2 * K * xy - K * y2) * x + (1 + K * x2) * y
    return proj(num / denom, K)

class CurvMessagePassing(nn.Module):
    def __init__(self, in_feats, out_feats, norm='both', bias=True):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.norm = norm

        self.W = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.W_self = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.W_self)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, graph, feat, K, edge_weight=None):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            h_src = feat_src @ self.W
            h_self = feat_dst @ self.W_self

            if self.norm in ['left', 'both']:
                if 'norm' in graph.ndata:
                    norm = graph.ndata['norm']
                else:
                    degs = graph.out_degrees().float().clamp(min=1)
                    norm = degs.pow(-0.5) if self.norm == 'both' else 1.0 / degs
                    graph.ndata['norm'] = norm
                h_src = h_src * norm.view(-1, 1)

            graph.srcdata['h'] = h_src
            if edge_weight is not None:
                graph.edata['w'] = edge_weight
                graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_agg'))
            else:
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_agg'))

            h_agg = graph.dstdata.pop('h_agg')

            h_agg = expmap0(h_agg, K)
            h_self = expmap0(h_self, K)

            h_out = kappa_add(h_agg, h_self, K)
            h_out = logmap0(h_out, K)

            h_out = h_out + self.bias
            
            return h_out

class WeightedHGCN2Layer(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats,
                 activation=F.relu, dropout=0.5):
        super().__init__()

        self.conv1 = CurvMessagePassing(in_feats, hidden_feats)
        self.conv2 = CurvMessagePassing(hidden_feats, out_feats)

        self.act = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, feat, K_list, edge_weight=None):
        h = self.conv1(graph, feat, K_list, edge_weight=edge_weight)
        h = self.act(h)
        h = self.dropout(h)

        h = self.conv2(graph, h, K_list, edge_weight=edge_weight)
        return h

class WeightedMessagePassing(nn.Module):
    def __init__(self, in_feats, hidden_feats, graph, K_max=1.0, dropout=0.5):
        super().__init__()

        self.gcnPos = WeightedHGCN2Layer(
            in_feats, hidden_feats, hidden_feats,
            dropout=dropout
        )
        self.gcnNeg = WeightedHGCN2Layer(
            in_feats, hidden_feats, hidden_feats,
            dropout=dropout
        )
        self.eps = 1e-6
        self.K_max = K_max
        self.raw_K_pos = nn.Parameter(torch.tensor(0.0))
        self.raw_K_neg = nn.Parameter(torch.tensor(0.0))

        self.fuse = nn.Linear(hidden_feats, hidden_feats)
        self.norm_fuse = nn.LayerNorm(hidden_feats)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.log_alpha_dict = nn.ParameterDict({
            etype: nn.Parameter(torch.tensor(0.0)) for etype in graph.etypes
        })
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, g, features, edge_preds):
        msg_pos_total = 0
        msg_neg_total = 0
        num_edge_types = len(edge_preds)
        for etype, edge_pred in edge_preds.items():
            graph = g[etype]

            edge_hetero = torch.sigmoid(edge_pred)
            alpha = torch.exp(self.log_alpha_dict[etype])

            w_pos = torch.exp(-alpha * edge_hetero)
            w_neg = 1 - torch.exp(-alpha * edge_hetero)

            K_pos = self.K_max * (torch.tanh(self.raw_K_pos) + 1.0) / 2.0    
            K_neg = -self.K_max * (torch.tanh(self.raw_K_neg) + 1.0) / 2.0   
            
            h_pos = self.gcnPos(graph, features, K_pos, edge_weight=w_pos)
            h_neg = self.gcnNeg(graph, features, K_neg, edge_weight=w_neg)


            msg_pos_total += h_pos
            msg_neg_total += h_neg

        if num_edge_types > 0:
            msg_pos_total /= num_edge_types
            msg_neg_total /= num_edge_types

        gamma = torch.sigmoid(self.gamma)
        z = gamma * msg_pos_total + (1 - gamma) * msg_neg_total

        z = self.norm_fuse(z)
        z = self.act(z)
        z = self.dropout(z)
        return z


