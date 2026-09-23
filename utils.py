import os
import argparse
import yaml
import numpy as np
import random
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, recall_score
import dgl
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

def compute_edge_auc(g, edge_score_dict, node_mask, device=None):
    """
    计算异质图中每种边类型的 AUC，仅考虑 node_mask 为 True 的节点。
    支持 GPU / CPU 自动适配。
    
    参数：
        g: DGL 图对象
        edge_score_dict: dict, {etype: edge_pred_tensor} 预测分数
        node_mask: tensor(bool), [num_nodes], True 表示节点参与计算
        device: None 或 'cpu'/'cuda'，默认为 node_mask 所在设备
    返回：
        auc_dict: 每种边类型的 AUC
        mean_auc: 平均 AUC
    """
    if device is None:
        device = node_mask.device

    node_labels = g.ndata['label'].to(device)
    node_mask = node_mask.to(device)

    auc_dict = {}
    auc_list = []

    print("=== Edge AUC Info (masked) ===")
    for etype in g.etypes:
        src, dst = g.edges(etype=etype)
        src, dst = src.to(device), dst.to(device)

        # ---- 仅保留 node_mask 内的边 ----
        edge_mask = node_mask[src] & node_mask[dst]

        if edge_mask.sum() == 0:
            print(f"Edge type: {etype} -> No edges in mask, skipping.")
            auc_dict[etype] = float('nan')
            continue

        # ---- 真实标签 & 预测分数 ----
        edge_label = (node_labels[src] != node_labels[dst]).long()[edge_mask]
        edge_score = edge_score_dict[etype][edge_mask]

        # ---- 转 CPU numpy 计算 AUC ----
        edge_label = edge_label.cpu().numpy()
        edge_score = edge_score.detach().cpu().numpy()

        num_edges = len(edge_label)
        num_pos = np.sum(edge_label == 1)
        num_neg = np.sum(edge_label == 0)

        try:
            auc = roc_auc_score(edge_label, edge_score)
        except ValueError:
            auc = float('nan')  # 标签单一时无法计算

        auc_dict[etype] = auc
        auc_list.append(auc)

        print(f"Edge type: {etype}")
        print(f"  Total edges in mask: {num_edges}, Positive: {num_pos}, Negative: {num_neg}")
        print(f"  AUC: {auc:.4f}")

    mean_auc = np.nanmean(auc_list)
    print(f"Mean AUC over all edge types: {mean_auc:.4f}")

    return auc_dict, mean_auc


def preprocess_and_save(graph, features, dataset_name, save_dir="./preprocessed"):
    """
    对节点特征进行曲率和异配性预处理，并保存。
    不使用节点度信息，保持特征维度与原始特征一致。
    如果已存在文件，则直接读取并返回。

    Args:
        graph (DGLGraph): 输入图
        features (torch.Tensor): 节点特征 [num_nodes, in_feats]
        dataset_name (str): 数据集名称，用于命名保存文件
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    curv_file = os.path.join(save_dir, f"{dataset_name}_curv_features.pt")
    hetero_file = os.path.join(save_dir, f"{dataset_name}_hetero_features.pt")

    # # 如果文件已存在，直接读取
    # if os.path.exists(curv_file) and os.path.exists(hetero_file):
    #     curv_features = torch.load(curv_file)
    #     hetero_features = torch.load(hetero_file)
    #     print(f"[INFO] 文件已存在，直接加载:\n  {curv_file}\n  {hetero_file}")
    #     return curv_features, hetero_features

    device = features.device
    N, F = features.shape

    # ---------------------
    # 邻居均值
    graph.ndata['h'] = features
    graph.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'neighbor_mean'))
    neighbor_mean = graph.ndata.pop('neighbor_mean')

    # ---------------------
    # 邻居方差
    graph.ndata['h'] = features
    graph.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'neighbor_mean_var'))
    neigh_mean = graph.ndata.pop('neighbor_mean_var')
    graph.ndata['h'] = features
    graph.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'neighbor_sq_sum'))
    neigh_sq_sum = graph.ndata.pop('neighbor_sq_sum')
    neighbor_var = neigh_sq_sum / graph.in_degrees().float().clamp(min=1).unsqueeze(1).to(device) - neigh_mean ** 2

    # ---------------------
    # 构建特征（不限制维度，可增加信息量）
    curv_features = torch.cat([features, features - neighbor_mean], dim=1)
    hetero_features = torch.cat([features, neighbor_var], dim=1)


    # ---------------------
    # 保存
    torch.save(curv_features, curv_file)
    torch.save(hetero_features, hetero_file)
    print(f"[INFO] 保存完成:\n  {curv_file}\n  {hetero_file}")

    return curv_features, hetero_features


def compute_gmean(labels_np, preds_np):
    from sklearn.metrics import confusion_matrix
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(labels_np, preds_np).ravel()
    # 计算 Sensitivity (召回率)
    if (tp + fn == 0):
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)
    # 计算 Specificity (特异度)
    if (tn + fp == 0):
        specificity = 0
    else:
        specificity = tn / (tn + fp)
    # 计算 G-Mean
    gmean = np.sqrt(sensitivity * specificity)
    return gmean
    
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp')
#     parser.add_argument('--gamma', type=float, default=1)
#     parser.add_argument('--C', type=int, default=1)
#     parser.add_argument('--K', type=int, default=1)
    args_input = parser.parse_args()
    config_path = '../config/'+args_input.dataset+'.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    args = argparse.Namespace(**config)
#     args.gamma = args_input.gamma
#     args.C = args_input.C
#     args.K = args_input.K
    print('----------------------------------')
    print('              args')
    print('----------------------------------')
    print(f'dataset:\t{args.dataset}')
    print(f'seed:\t{args.seed}')
    print(f'epoch:\t{args.epoch}')
    print(f'early_stop:\t{args.early_stop}')
    print(f'lr:\t{args.lr}')
    print(f'weigth_decay:{args.weight_decay}')
    print(f'gamma:\t{args.gamma}')
    print(f'C:\t{args.C}')
    print(f'K:\t{args.K}')
    print(f'intra_dim:\t{args.intra_dim}')
    print(f'dropout:\t{args.dropout}')
    print(f'cuda:\t{args.cuda}')
    print('----------------------------------')
    return args

class EarlyStop():
    def __init__(self, early_stop, if_more=True) -> None:
        self.best_eval = 0
        self.best_epoch = 0
        self.if_more = if_more
        self.early_stop = early_stop
        self.stop_steps = 0
    
    def step(self, current_eval, current_epoch):
        do_stop = False
        do_store = False
        if self.if_more:
            if current_eval > self.best_eval:
                self.best_eval = current_eval
                self.best_epoch = current_epoch
                self.stop_steps = 1
                do_store = True
            else:
                self.stop_steps += 1
                if self.stop_steps >= self.early_stop:
                    do_stop = True
        else:
            if current_eval < self.best_eval:
                self.best_eval = current_eval
                self.best_epoch = current_epoch
                self.stop_steps = 1
                do_store = True
            else:
                self.stop_steps += 1
                if self.stop_steps >= self.early_stop:
                    do_stop = True
        return do_store, do_stop

def conf_gmean(conf):
	tn, fp, fn, tp = conf.ravel()
	return (tp*tn/((tp+fn)*(tn+fp)))**0.5
def prob2pred(prob, threshhold=0.5):
    pred = np.zeros_like(prob, dtype=np.int32)
    pred[prob >= threshhold] = 1
    pred[prob < threshhold] = 0
    return pred
def evaluate(labels, logits, result_path = ''):
    probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()
    if len(result_path)>0:
        np.save(result_path+'_result_preds', preds)
        np.save(result_path+'_result_probs', probs)
    conf = confusion_matrix(labels, preds)
    recall = round(recall_score(labels, preds), 4)
    f1_macro = round(f1_score(labels, preds, average='macro'), 4)
    auc = round(roc_auc_score(labels, probs), 4)
    gmean = round(conf_gmean(conf), 4)
    return f1_macro, auc, gmean, recall

def hinge_loss(labels, scores):
    margin = 1
    ls = labels*scores
    
    loss = F.relu(margin-ls)
    loss = loss.mean()
    return loss

def normalize(mx):
	"""
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx

