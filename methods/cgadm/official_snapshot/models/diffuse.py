from torch_geometric.nn import GINConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from torch.nn import Linear, Sequential, ReLU, Dropout, Identity
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import sympy
import scipy
from torch_sparse import SparseTensor



class PolyConv(MessagePassing):
    def __init__(self, theta, **kwargs):
        super(PolyConv, self).__init__(aggr='add', **kwargs)
        self._theta = theta
        self._k = len(self._theta)

    def forward(self, x, edge_index):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        D_invsqrt = torch.pow(degree(edge_index[0], dtype=x.dtype).clamp(min=1), -0.5)
        norm = D_invsqrt[edge_index[0]] * D_invsqrt[edge_index[1]]
        h = self._theta[0]*x
        for k in range(1, self._k):
            x = x - self.propagate(edge_index, x=x, norm=norm)
            h += self._theta[k]*x
        return h

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j




def calculate_theta(d):
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


class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, dropout_rate=0, activation='ReLU', **kwargs):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        if num_layers == 0:
            return
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for i in range(1, num_layers-1):
                self.layers.append(nn.Linear(h_feats, h_feats))
            self.layers.append(nn.Linear(h_feats, num_classes))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, h):
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers)-1:
                h = self.act(h)
        return h

class BWGNN(torch.nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=2, dropout_rate=0, activation='ReLU', **kwargs):
        super().__init__()
        self.thetas = calculate_theta(d=num_layers)
        self.convs = torch.nn.ModuleList([PolyConv(theta) for theta in self.thetas])
        self.linear = Linear(in_feats, h_feats)
        self.linear2 = Linear(h_feats, h_feats)
        self.mlp = MLP(h_feats*len(self.convs), h_feats, num_classes, mlp_layers, dropout_rate)
        self.act = getattr(torch.nn, activation)()
        self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else Identity()

    def forward(self, x, edge_index):
        x = self.act(self.linear(x))
        x = self.act(self.linear2(x))
        h_final = torch.cat([conv(x, edge_index) for conv in self.convs], dim=-1)
        h_final = self.dropout(h_final)
        h = self.mlp(h_final)
        return h
    


def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super(GaussianDiffusionTrainer, self).__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)


        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
    
    def forward(self, x, adj, node_topo, prior, ground_truth, mask):
        t_edge = torch.randint(self.T, size=(ground_truth.shape[0], ), device=ground_truth.device)
        noise = torch.randn_like(prior)
        x_t = (extract(self.sqrt_alphas_bar, t_edge, ground_truth.shape) * ground_truth + extract(self.sqrt_one_minus_alphas_bar, t_edge, ground_truth.shape) * noise + (1- extract(self.sqrt_alphas_bar, t_edge, ground_truth.shape)) * prior)
        noise_recon = self.model(x, adj, node_topo, x_t, prior, t_edge, mask)
        loss = F.mse_loss(noise_recon.squeeze(), noise, reduction='none')

        return loss
    

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]


        self.register_buffer("coeff0", self.betas * torch.sqrt(alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer("coeff1", (1. - alphas_bar_prev) * torch.sqrt(alphas) / (1. - alphas_bar))
        self.register_buffer("coeff2", 1 + (torch.sqrt(alphas_bar)-1)*(torch.sqrt(alphas) + torch.sqrt(alphas_bar_prev)) / (1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps, prior):
        assert x_t.shape == eps.shape
        return (
            (x_t - (1 -extract(self.sqrt_alphas_bar, t, x_t.shape)) * prior - extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape) * eps) / extract(self.sqrt_alphas_bar, t, x_t.shape)
        )

    def p_mean_variance(self, graph_emb, node_topo, x_t, prior, t_edge):
        t = t_edge
        var = self.posterior_var
        var = extract(var, t, x_t.shape)

        eps = self.model.predict(graph_emb, node_topo, x_t, prior, t_edge).squeeze()
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps, prior)

        return xt_prev_mean, var

    def forward(self, x, adj, node_topo, prior_sample, prior):
        x_t = prior_sample
        graph_emb = self.model.compute_graph_emb(x, adj)
        for time_step in reversed(range(self.T)):
            t_edge = x_t.new_ones([prior.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(graph_emb, node_topo, x_t, prior, t_edge)
            if time_step > 0:
                noise = torch.randn_like(x_t)
                x_t = extract(self.coeff0, t_edge, x_t.shape) * mean + extract(self.coeff1, t_edge, x_t.shape) * x_t + extract(self.coeff2, t_edge, x_t.shape) * prior + torch.sqrt(var) * noise
            else:
                x_t = mean
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        
        return x_0
    

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb
    

class GIN_noparam(torch.nn.Module):
    def __init__(self, num_layers=2, **kwargs):
        super(GIN_noparam, self).__init__()

        self.conv = GINConv(nn.Identity(), train_eps=True)
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        h_final = x.detach().clone()
        for i in range(self.num_layers):
            x = self.conv(x, edge_index)
            h_final = torch.cat([h_final, x], -1)
        return h_final



class Denoiser(nn.Module):
    def __init__(self, T, topo_dim, gin_h, in_channels, hidden_channels, out_channels, dropout, ln, tailact) -> None:
        super(Denoiser, self).__init__()

        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()
        
        self.topo_feature = nn.Sequential(
            nn.Linear(topo_dim, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())


        self.time_embedding = TimeEmbedding(T, hidden_channels, hidden_channels)
        self.prior_feature = nn.Sequential(nn.Linear(1, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.current_feature = nn.Sequential(nn.Linear(1, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))


        self.encoder = BWGNN(in_channels, hidden_channels, hidden_channels)
        

        self.lin = nn.Sequential(nn.Linear(hidden_channels * 4, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, out_channels))
        

    def forward(self, x, adj, node_topo, x_t, prior, t_node, mask=None):
        prior = prior.unsqueeze(-1)
        x_t = x_t.unsqueeze(-1)

        prior_emb = self.prior_feature(prior)
        x_t_emb = self.current_feature(x_t)

        t_node_emb = self.time_embedding(t_node)

        adjmask = torch.ones_like(adj[0], dtype=torch.bool)
        # randomly sample 2% of index from range(len(adj[0])):
        idx_batch = torch.randperm(len(adj[0]))[:int(0.02 * len(adj[0]))]
        adjmask[idx_batch] = 0
        adj = adj[:, adjmask]
        node_graph_emb = self.encoder(x, adj)
        if mask is not None:
            node_graph_emb = node_graph_emb[mask]


        node_h = node_graph_emb + self.topo_feature(node_topo)


        predict_input = torch.cat([node_h, prior_emb, x_t_emb, t_node_emb], dim=-1)
        return self.lin(predict_input)
    
    @torch.no_grad()
    def predict(self, node_graph_emb, node_topo, x_t, prior, t_node, mask=None):
        prior = prior.unsqueeze(-1)
        x_t = x_t.unsqueeze(-1)

        prior_emb = self.prior_feature(prior)
        x_t_emb = self.current_feature(x_t)

        t_node_emb = self.time_embedding(t_node)

        node_h = node_graph_emb + self.topo_feature(node_topo)


        predict_input = torch.cat([node_h, prior_emb, x_t_emb, t_node_emb], dim=-1)
        return self.lin(predict_input)
    
    @torch.no_grad()
    def compute_graph_emb(self, x, adj):
        return self.encoder(x, adj)
    
    @torch.no_grad()
    def compute_prior_feature(self, x, adj):
        node_graph_emb = self.encoder(x, adj)
        return node_graph_emb

    

class Model(nn.Module):
    def __init__(self, opt, Data, device) -> None:
        super(Model, self).__init__()

        self.name = "CGADM"
        self.full_graph = Data.graph
        self.x = self.full_graph.feature.to(device)
        self.adj = self.full_graph.edge_index.to(device)
        self.train_mask = self.full_graph.train_masks
        self.val_mask = self.full_graph.val_masks
        self.test_mask = self.full_graph.test_masks
        self.labels = self.full_graph.label

        with torch.no_grad():
            self.gin = GIN_noparam(num_layers=opt.num_layers).to(device)
            self.node_h = self.gin(self.x, self.adj)

        self.node_topo = torch.cat([Data.node_centrality.unsqueeze(1)], dim=1).to(device)

        self.denoiser = Denoiser(opt.num_steps, self.node_topo.shape[1], self.node_h, self.x.shape[1], opt.hiddim, 1, opt.gnndp, True, True)

        self.diffusion_trainer = GaussianDiffusionTrainer(self.denoiser, opt.beta_1, opt.beta_T, opt.num_steps)
        self.diffusion_sampler = GaussianDiffusionSampler(self.denoiser, opt.beta_1, opt.beta_T, opt.num_steps)

        self.prior_model = xgb.XGBClassifier(tree_method='gpu_hist', eval_metric=roc_auc_score, verbose=2, n_estimators=100)
        self.train_X = self.node_h[self.train_mask].cpu().numpy()
        self.train_y = self.labels[self.train_mask].cpu().numpy()
        self.val_X = self.node_h[self.val_mask].cpu().numpy()
        self.val_y = self.labels[self.val_mask].cpu().numpy()
        self.prior_model.fit(self.train_X, self.train_y, eval_set=[(self.val_X, self.val_y)])
        self.prior = torch.tensor(self.prior_model.predict_proba(self.node_h.cpu().numpy())[:, 1])

        self.device = device
        self.T = opt.num_steps


    def forward(self):
        train_labels = self.labels[self.train_mask].to(self.device)
        train_prior = self.prior[self.train_mask].to(self.device)
        node_topo = self.node_topo[self.train_mask].to(self.device)

        loss = self.diffusion_trainer(self.x, self.adj, node_topo, train_prior, train_labels, self.train_mask).mean()
        prior_feature = torch.cat([self.denoiser.compute_graph_emb(self.x, self.adj), self.node_h], dim=1)
        new_train_x = prior_feature[self.train_mask].cpu().numpy()
        new_val_x = prior_feature[self.val_mask].cpu().numpy()
        self.prior_model.fit(new_train_x, self.train_y, eval_set=[(new_val_x, self.val_y)])
        self.prior = torch.tensor(self.prior_model.predict_proba(prior_feature.cpu().numpy())[:, 1])

        return loss


    def predict(self):
        prior = self.prior.to(self.device)
        sample_noise = torch.randn_like(prior)
        confidence = (prior - 0.5).abs() / 0.5
        prior_sample = prior + sample_noise
        result = confidence * prior + (1 - confidence) * self.diffusion_sampler(self.x, self.adj, self.node_topo, prior_sample, prior)

        return result


def get_optimizer(model, opt):
    return torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

def get_scheduler(optimizer, opt):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)