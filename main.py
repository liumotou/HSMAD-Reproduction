# -*- coding: utf-8 -*-
import argparse
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, average_precision_score
import numpy as np
import torch
from model import *
from sklearn.model_selection import train_test_split
from dataset import Dataset  # 假设Dataset类用于加载数据和构建图
from utils import *
import warnings
warnings.filterwarnings('ignore')
import time

def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def train(graph, args, device):
    """
    统一训练函数：包含模型初始化、训练、验证、早停与最终评估。
    """
    # === 数据准备 ===
    features = graph.ndata['feature'].to(device)
    labels = graph.ndata['label'].to(device)

    train_mask = graph.ndata['train_mask'].to(device).bool()
    val_mask = graph.ndata['val_mask'].to(device).bool()
    test_mask = graph.ndata['test_mask'].to(device).bool()
    edge_labels = {}  # 保存每种关系类型的边标签

    for etype in graph.etypes:
        src, dst = graph.edges(etype=etype)

        src_ntype, _, dst_ntype = graph.to_canonical_etype(etype)
        src_labels = graph.nodes[src_ntype].data['label']
        dst_labels = graph.nodes[dst_ntype].data['label']
        edge_label = (src_labels[src] != dst_labels[dst]).long()
        edge_labels[etype] = edge_label

    graph = graph.to(device)

    in_feats = graph.ndata['feature'].shape[1]
    h_feats = args.hid_dim
    out_feats = 2

    print(f"Initializing model on device: {device}")
    print(f"Ablation mode: {args.ablation}")

    # === 初始化模型 ===
    model = CombinedModel(
        graph, in_feats, h_feats, out_feats, d=args.order,
        quantile=args.q, ablation=args.ablation
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=50)

    # === 指标记录 ===
    total, best_f1, best_auc, final_trec, final_tpre, final_tmf1, final_tauc, final_gmean = 0., 0., 0., 0., 0., 0., 0., 0.
    train_losses, val_losses = [], []
    epochs_without_improvement = 0
    patience = args.patience
    best_model_state = None

    # === 时间 & 显存记录 ===
    train_times, inference_times, train_memories, inference_memories = [], [], [], []

    for epoch in range(args.epoch):
        # =============== 训练阶段 ===============
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()

        optimizer.zero_grad()
        output, edge_preds = model(features)
        loss = model.compute_loss(output, labels, edge_labels, edge_preds, train_mask) 
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # 记录训练时间和显存
        train_times.append(time.time() - t0)
        train_memories.append(torch.cuda.max_memory_allocated(device) / 1024**2)  # MB

        # =============== 验证阶段 ===============
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
        t1 = time.time()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                output, edge_preds = model(features)
                val_loss = model.compute_loss(output, labels, edge_labels, edge_preds, val_mask)
                val_losses.append(val_loss.item())

                probs = output.exp()[:, 1]
                mf1, thres = get_best_f1(labels[val_mask].cpu(), probs[val_mask].cpu().numpy())

                # 记录推理时间和显存
                inference_times.append(time.time() - t1)
                inference_memories.append(torch.cuda.max_memory_allocated(device) / 1024**2)  # MB

                # =============== Early Stopping ===============
                if total < mf1:
                    total = mf1
                    best_f1 = mf1 
                    pred = torch.zeros_like(labels, device=device)
                    pred[probs > thres] = 1

                    labels_np = labels[test_mask].cpu().numpy()
                    preds_np = pred[test_mask].cpu().numpy()
                    probs_np = probs[test_mask].cpu().numpy()

                    binary_mask = (labels_np == 0) | (labels_np == 1)

                    labels_np = labels_np[binary_mask]
                    preds_np = preds_np[binary_mask]
                    probs_np = probs_np[binary_mask]

                    trec = recall_score(labels_np, preds_np)
                    tpre = precision_score(labels_np, preds_np)
                    tmf1 = f1_score(labels_np, preds_np, average='macro')
                    tauc = roc_auc_score(labels_np, np.nan_to_num(probs_np))
                    tauprc = average_precision_score(labels_np, np.nan_to_num(probs_np))
                    gmean = compute_gmean(labels_np, preds_np)
                    final_trec, final_tpre, final_tmf1, final_tauc, final_tauprc, final_gmean = trec, tpre, tmf1, tauc, tauprc, gmean
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 10

                if epochs_without_improvement >= patience:
                    print(f"⏸️ Early stopping at epoch {epoch} (no F1 improvement for {patience} epochs).")
                    break
            
                print(f"[Epoch {epoch+1}] "
                    f"TrainLoss: {loss.item():.4f}| ValLoss: {val_loss.item():.4f}"
                    f"| Val F1: {mf1:.4f}| Best F1: {best_f1:.4f}| Test F1: {tmf1:.4f}")


    # === 输出最终结果 ===
    print('✅ Test Results: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} AUPRC {:.2f} G-Mean {:.2f}'.format(
        final_trec * 100, final_tpre * 100, final_tmf1 * 100, final_tauc * 100, final_tauprc * 100, final_gmean * 100))

    print(f"Average Train Time: {np.mean(train_times):.4f}s | Inference Time: {np.mean(inference_times):.4f}s")
    print(f"Peak Train Memory: {np.max(train_memories):.2f} MB | Inference Memory: {np.max(inference_memories):.2f} MB")

    return (final_trec, final_tpre, final_tmf1, final_tauc, final_tauprc, final_gmean,
            train_times, inference_times, train_memories, inference_memories)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--homo', type=int, default=1)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--train_ratio', type=float, default=0.4)
    parser.add_argument('--q', type=float, default=0.5)
    parser.add_argument(
        '--ablation', type=str, default='full',
        choices=['full', 'no_hwsf', 'no_hrmu'],
        help='Ablation setting: full model, w/o HWSF, or w/o HRMU.'
    )
    parser.add_argument('--gpu', type=int, default=0,
                        help='Choose a specific CUDA device number, like 0, 1, etc.',
                        # 根据 torch.cuda.device_count() 生成合法的整数范围选项
                        choices=list(range(torch.cuda.device_count())))

    # 解析命令行参数
    args = parser.parse_args()

    # 根据传入的参数设置设备，将整数转换为对应的 cuda 设备标识格式
    device = torch.device(f'cuda:{args.gpu}')
    dataset_name = args.dataset
    homo = args.homo
    if homo == 0:
        prefix='datasets/hetero/'
    else:
        prefix='datasets/'
    dataset = Dataset(name=dataset_name, prefix=prefix)
    graph = dataset.graph
    print(graph)
    # 生成训练、验证和测试掩码
    num_nodes = graph.num_nodes()
    node_labels = graph.ndata['label']

    # 获取节点的掩码
    train_mask = graph.ndata['train_mask'].bool()
    val_mask = graph.ndata['val_mask'].bool()
    test_mask = graph.ndata['test_mask'].bool()

    graph = graph.to(device)

    final_recs, final_pres, final_mf1s, final_aucs, final_auprcs, final_gmeans = [], [], [], [], [], []
    all_train_times, all_inference_times, all_train_memories, all_inference_memories = [], [], [], []
    # ===== 总开始时间 =====
    t0_total = time.time()
    for tt in range(args.run):
        print(f"\n🚀 Running {tt+1}/{args.run} ...")
        t0_train = time.time()
        rec, pre, mf1, auc, auprc, gmean, train_times, inference_times, train_memories, inference_memories = \
            train(graph, args, device)
        t1_train = time.time()
        train_time = t1_train - t0_train
        print(f"Time cost: {train_time:.2f}s")
        # === 累积单次结果 ===
        final_recs.append(rec)
        final_pres.append(pre)
        final_mf1s.append(mf1)
        final_aucs.append(auc)
        final_auprcs.append(auprc)
        final_gmeans.append(gmean)

        # === 累积资源信息 ===
        all_train_times.extend(train_times)
        all_inference_times.extend(inference_times)
        all_train_memories.extend(train_memories)
        all_inference_memories.extend(inference_memories)

    # === 转为numpy数组 ===
    final_recs   = np.array(final_recs)
    final_pres   = np.array(final_pres)
    final_mf1s   = np.array(final_mf1s)
    final_aucs   = np.array(final_aucs)
    final_auprcs = np.array(final_auprcs)
    final_gmes   = np.array(final_gmeans)

    # === 输出统计结果 ===
    print('\n' + '-' * 60)
    print('📊 Test Results Summary'.center(60))
    print('-' * 60)
    print('Rec-mean: {:.2f}, Rec-std: {:.2f}'.format(100 * np.mean(final_recs),   100 * np.std(final_recs)))
    print('Pre-mean: {:.2f}, Pre-std: {:.2f}'.format(100 * np.mean(final_pres),   100 * np.std(final_pres)))
    print('MF1-mean: {:.2f}, MF1-std: {:.2f}'.format(100 * np.mean(final_mf1s),   100 * np.std(final_mf1s)))
    print('AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_aucs),   100 * np.std(final_aucs)))
    print('AUPRC-mean: {:.2f}, AUPRC-std: {:.2f}'.format(100 * np.mean(final_auprcs), 100 * np.std(final_auprcs)))
    print('GMe-mean: {:.2f}, GMe-std: {:.2f}'.format(100 * np.mean(final_gmes),   100 * np.std(final_gmes)))
    print('-' * 60)

    # ===== 总结束时间 =====
    t1_total = time.time()
    total_time_min = (t1_total - t0_total) / 60
    print(f"\n⏱ Total running time for {args.run} runs: {total_time_min:.2f} minutes\n")
    # === 输出资源统计 ===
    print('\n🕒 Training Time  (mean ± std): {:.4f} ± {:.4f} s/epoch'.format(
        np.mean(all_train_times), np.std(all_train_times)))
    print('⚙️  Inference Time (mean ± std): {:.4f} ± {:.4f} s/epoch'.format(
        np.mean(all_inference_times), np.std(all_inference_times)))
    print('💾 Train Memory Peak: {:.2f} ± {:.2f} MB'.format(
        np.mean(all_train_memories), np.std(all_train_memories)))
    print('💾 Inference Memory Peak: {:.2f} ± {:.2f} MB'.format(
        np.mean(all_inference_memories), np.std(all_inference_memories)))
    print('-' * 60)

    
    
