import dgl
from dgl.data.utils import load_graphs
import torch
import numpy as np
from sklearn.model_selection import train_test_split
class Dataset:
    def __init__(self, 
                 name='tfinance', 
                 prefix='datasets/',
                 seed=42):
        graphs, _ = load_graphs(prefix + name)
        graph = graphs[0]
        labels = graph.ndata['label'].long().squeeze(-1).cpu().numpy()
        features = graph.ndata['feature']
        if prefix == 'datasets/':
            graph = dgl.to_bidirected(graph)
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)   
        
        self.name = name
        self.graph = graph

        
        index = np.arange(len(labels))
        if name == 'amazon':
            index = list(range(3305, len(labels)))
        np.random.seed(seed)

        # === 掩码构造与输出统计统一函数 ===
        def assign_masks(idx_train, idx_valid, idx_test):
            train_mask = torch.zeros(len(labels), dtype=torch.bool)
            val_mask   = torch.zeros(len(labels), dtype=torch.bool)
            test_mask  = torch.zeros(len(labels), dtype=torch.bool)

            train_mask[idx_train] = 1
            val_mask[idx_valid]   = 1
            test_mask[idx_test]   = 1

            graph.ndata['train_mask'] = train_mask
            graph.ndata['val_mask']   = val_mask
            graph.ndata['test_mask']  = test_mask

            # === 输出信息 ===
            def ratio_info(mask):
                subset = labels[mask.numpy()]
                num_anom = np.sum(subset == 1)
                num_norm = np.sum(subset == 0)
                ratio = num_anom / max(1, (num_norm + num_anom))
                return num_norm, num_anom, ratio

            n_train, a_train, r_train = ratio_info(train_mask)
            n_val,   a_val,   r_val   = ratio_info(val_mask)
            n_test,  a_test,  r_test  = ratio_info(test_mask)

            print(f"Train: {train_mask.sum().item()} nodes (normal={n_train}, anomaly={a_train}, ratio={r_train:.3f})")
            print(f"Val:   {val_mask.sum().item()} nodes (normal={n_val}, anomaly={a_val}, ratio={r_val:.3f})")
            print(f"Test:  {test_mask.sum().item()} nodes (normal={n_test}, anomaly={a_test}, ratio={r_test:.3f})")

        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index],  
                                                                stratify=labels[index],  
                                                                train_size=0.4,
                                                                random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest,
                                                                stratify=y_rest,
                                                                test_size=0.67,
                                                                random_state=2, shuffle=True)
        assign_masks(idx_train, idx_valid, idx_test)

        graph.ndata['label'] = torch.tensor(labels, dtype=torch.long)
        graph.ndata['feature'] = features.float()
