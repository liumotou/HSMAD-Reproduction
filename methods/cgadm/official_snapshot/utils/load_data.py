import os
from torch_geometric.transforms import Compose
import torch
import torch_geometric as pyg
from torch_geometric.utils import to_dense_adj, negative_sampling, to_networkx, degree
from ogb.linkproppred import PygLinkPropPredDataset
from torch.utils.data.dataset import Dataset
from torch_sparse import SparseTensor
import pandas
from dgl.data import FraudAmazonDataset, FraudYelpDataset
from sklearn.model_selection import train_test_split
import dgl
import numpy as np
import random
import time
import networkx as nx


def data_load(opt):
    dataset = Dataset(name=opt.dataset_name)
    dataset.split()
    graph = from_dgl(dataset.graph)


    return Data(graph)



class Data(object):
    def __init__(self, graph):
        self.graph = graph
        self.graph.edge_index = self.graph.edge_index.to(torch.int64)
        self.edge_set = graph.edge_index.t()
        self.train_dataset = Train_dataset(self.edge_set)

        self.num_features = graph.feature.shape[1]

        self.val_labels = graph.label[graph.val_masks]
        self.test_labels = graph.label[graph.test_masks]


        graph_x = to_networkx(graph, to_undirected=True)
        

        node_centrality = nx.degree_centrality(graph_x)
        node_degree = degree(graph.edge_index[0])
        self.node_degree = node_degree / node_degree.max()

        self.node_centrality = torch.tensor([node_centrality[i] for i in range(graph.num_nodes)])



class Dataset:
    def __init__(self, name='tfinance', homo=True, add_self_loop=True, to_bidirectional=False, to_simple=True):
        if name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label'])

        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            graph.ndata['mark'] = graph.ndata['train_mask']+graph.ndata['val_mask']+graph.ndata['test_mask']
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'mark'])

        else:
            graph = dgl.load_graphs('datasets/'+name)[0][0]
        graph.ndata['feature'] = graph.ndata['feature'].float()
        graph.ndata['label'] = graph.ndata['label'].long()
        self.name = name
        self.graph = graph
        if add_self_loop:
            self.graph = dgl.add_self_loop(self.graph)
        if to_bidirectional:
            self.graph = dgl.to_bidirected(self.graph, copy_ndata=True)
        if to_simple:
            self.graph = dgl.to_simple(self.graph)

    def split(self, samples=20):
        labels = self.graph.ndata['label']
        n = self.graph.num_nodes()
        if 'mark' in self.graph.ndata:
            index = self.graph.ndata['mark'].nonzero()[:,0].numpy().tolist()
        else:
            index = list(range(n))
        train_masks = torch.zeros([n,20]).bool()
        val_masks = torch.zeros([n,20]).bool()
        test_masks = torch.zeros([n,20]).bool()

        train_ratio, val_ratio = 0.2, 0.1

        i = 0
        seed = int(time.time())
        set_seed(seed)
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index], train_size=train_ratio, random_state=seed, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, train_size=int(len(index)*val_ratio), random_state=seed, shuffle=True)
        train_masks[idx_train,i] = 1
        val_masks[idx_valid,i] = 1
        test_masks[idx_test,i] = 1


        self.graph.ndata['train_masks'] = train_masks[:, 0]
        self.graph.ndata['val_masks'] = val_masks[:, 0]
        self.graph.ndata['test_masks'] = test_masks[:, 0]



def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def from_dgl(g):
    import dgl

    from torch_geometric.data import Data, HeteroData

    if not isinstance(g, dgl.DGLGraph):
        raise ValueError(f"Invalid data type (got '{type(g)}')")

    if g.is_homogeneous:
        data = Data()
        data.edge_index = torch.stack(g.edges(), dim=0)

        for attr, value in g.ndata.items():
            data[attr] = value
        for attr, value in g.edata.items():
            data[attr] = value

        return data

    data = HeteroData()

    for node_type in g.ntypes:
        for attr, value in g.nodes[node_type].data.items():
            data[node_type][attr] = value

    for edge_type in g.canonical_etypes:
        row, col = g.edges(form="uv", etype=edge_type)
        data[edge_type].edge_index = torch.stack([row, col], dim=0)
        for attr, value in g.edge_attr_schemes(edge_type).items():
            data[edge_type][attr] = value

    return data



class Train_dataset(Dataset):
    def __init__(self, train_dataset):
        super(Train_dataset, self).__init__()
        self.train_dataset = train_dataset

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        pos_pair = self.train_dataset[idx]

        return pos_pair, idx


dataset = Dataset(name="tfinance")
dataset.split()
graph = from_dgl(dataset.graph)

print(graph)