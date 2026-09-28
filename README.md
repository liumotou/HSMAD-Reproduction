# HSMAD 统一复现工作区

[中文](README.md) | [English](README_EN.md)

本仓库包含 HSMAD 主方法代码，以及在统一数据、固定划分和统一评价口径下整理的 Table 1 baseline 候选实现与运行脚本。

> **复现定位**：除非单独注明，`methods/` 下的 baseline 应理解为 `candidate_protocol_not_author_exact`，即“经过项目适配、可审计的候选协议”，不宣称与作者代码逐字节一致，也不承诺复现到论文四位小数。
>
> 数据集、冻结 mask、checkpoint、训练日志和正式实验结果体积较大，不上传到 GitHub；仓库提供下载来源、代码、配置和使用说明。

## 1. 研究目标

在相同数据版本、相同 train/validation/test 划分和相同最终评价口径下，对 HSMAD 及其 baseline 进行可复算、可审计的实验：

- 六个数据集使用持久化的冻结 mask，不随训练 seed 改变；
- 训练 seed 固定为 `0..9`；
- 训练集只参与损失计算与参数更新；
- 验证集负责 early stopping、checkpoint 和 F1 阈值选择；
- 测试集只在模型与阈值固定后计算最终 F1-Macro 和 AUROC；
- 每次运行保存 config、日志、history、checkpoint、metrics、耗时、峰值显存和 SHA256；
- 十次结果采用样本标准差，即 `ddof=1`。

## 2. 仓库结构

```text
.
├── main.py                       # HSMAD 入口
├── model.py                      # HSMAD 模型
├── dataset.py                    # 数据加载
├── manifold_update.py
├── utils.py
├── methods/                      # baseline 代码、配置和契约测试
├── docs/DATASETS.md              # 数据集下载地址与来源说明
├── docs/BASELINE_SCRIPTS.md      # baseline 入口索引
├── scripts/validate_repository.py
└── datasets/                     # 本地数据目录，不提交到 Git
```

## 3. 数据集下载与放置

完整下载说明见 [docs/DATASETS.md](docs/DATASETS.md)。主要统一来源为：

- GADBench：<https://github.com/squareroot3/GADBench>
- GADBench 数据包：<https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing>

当前代码约定数据文件放置为：

```text
datasets/weibo
datasets/amazon
datasets/yelp
datasets/tolokers
datasets/tfinance
datasets/tsocial
```

这些文件是 DGL graph 文件，通常不带扩展名。每个图应至少包含：

```python
graph.ndata['feature']
graph.ndata['label']
graph.ndata['train_mask']
graph.ndata['val_mask']
graph.ndata['test_mask']
```

### 下载地址存在不等于一定能直接训练

需要区分四个层次：

1. **来源确认**：链接来自论文、官方仓库或 benchmark；
2. **下载完成**：文件完整下载，大小与 SHA256 可核验；
3. **格式兼容**：文件能被当前 DGL/PyG 版本读取，字段和 shape 正确；
4. **训练通过**：依赖、CUDA、显存、代码路径和配置均匹配，并完成 smoke/diagnostic/formal。

因此，新环境下载后应先核对文件和 mask，再运行 smoke，不能仅凭链接可打开就认定完整训练一定成功。

### 特殊数据规则

- **Amazon**：前 3305 个未覆盖节点保留在完整 transductive 图中参与消息传递，但不得进入 train/validation/test mask、loss 或指标。
- **Tolokers**：使用项目冻结的 HSMAD mask，不使用图内原始 50/25/25 masks。
- **图方法**：按对应配置执行图预处理；多数项目候选使用 `to_bidirected -> remove_self_loop -> add_self_loop`。
- **纯特征方法**：MLP、SVM 等不得读取图边。

## 4. Yelp 数据说明

Yelp 的“数据来源”和“实验覆盖”是两个不同问题：

1. **来源已经核验**：项目曾将当前 Yelp DGL 图与 GADBench 官方 Google Drive 候选文件逐项对照。feature SHA256 与 label SHA256 一致；官方候选比当前图多 45,954 条边，恰好是每个节点一个 self-loop。当前图可解释为官方候选执行 `remove_self_loop()` 后的版本。
2. **不是所有 baseline 都完成 Yelp**：后续 baseline 阶段曾冻结 Yelp，不再自动启动新训练。因此缺少某方法的 Yelp 结果，不代表 Yelp 文件错误，也不代表该算法理论上无法处理 Yelp。

结论：**当前 Yelp 数据来源可标记为已验证的官方 GADBench 预处理版本，但每个 baseline 的 Yelp 完成度仍需单独判断。**

## 5. 为什么有些 baseline 只有一两个数据集？

本仓库保留真实完成状态，不用空结果拼成完整矩阵。常见原因包括：

- 作者仓库本身只支持少量数据集或特定数据格式；
- 模型代码已适配，但新数据集 loader、runner 或独立 checkpoint 复算尚未完成；
- 大图全图训练有显存要求，项目不通过擅自采样、删边或减小模型来规避；
- 某数据集被协议冻结，暂不启动新实验；
- 方法存在依赖、ABI、预计算、张量 shape 或确定性问题；
- 只有 smoke/diagnostic，还没有通过正式十 seed 审计。

因此，“只完成一两个数据集”通常表示其他组合仍是 `INCOMPLETE` 或 `BLOCKED`，并不表示删除了低分结果。

## 6. Baseline 代码状态

| 方法 | 代码 | 入口 | 当前说明 |
|---|---:|---|---|
| HSMAD | 完整 | `main.py` | 项目主方法 |
| MLP | 有 | `methods/mlp/src/formal_runner.py` | feature-only；服务器实验源码已恢复 |
| GCN | 有 | `methods/gcn/src/run_kipf_v2.py` | Kipf 两层候选协议 |
| ChebNet | 有 | `methods/chebnet/src/run.py` | PyG 候选协议 |
| GIN | 有 | `methods/gin/src/run.py` | PyG 候选协议 |
| GWNN | 有 | `methods/gwnn/src/run.py` | 论文公式候选协议 |
| GAT v1 | 有 | smoke/diagnostic runner | 旧诊断版本，不作为当前主候选 |
| GAT-v2 | 有 | `methods/gat_v2_gadbench/src/run_*.py` | GADBench 结构适配候选 |
| GraphSAGE | 有 | `methods/graphsage/src/run_smoke.py` / `run_formal.py` | GADBench pool 候选 |
| GraphConsis | 有 | `methods/graphconsis/src/run.py` | 单关系适配候选 |
| CARE-GNN | 有 | `methods/caregnn/src/run.py` | 单关系适配候选 |
| PC-GNN | 有 | `methods/pcgnn/src/run.py` | 单关系适配候选 |
| BWGNN | 有 | `methods/bwgnn/src/run_smoke.py` / `run_formal.py` | 项目适配候选 |
| SparseGAD | 有 | `methods/sparsegad/src/run_smoke.py` / `run_formal.py` | 项目适配候选 |
| SVM | 有 | `methods/svm/src/runner.py` | feature-only 候选 |
| SEC-GFD | 有 | `methods/sec_gfd/src/runner.py` | 部分数据集候选 |
| GHRN | 有 | `methods/ghrn/src/runner.py` | 项目适配候选 |
| CGADM | 有 | `methods/cgadm_hsmad/src/runner.py` | HSMAD 数据适配候选 |
| DSGAD | 有 | `methods/dsgad/src/runner.py` | 项目适配候选 |
| NRGL | 有 | Python API `run(...)` | 当前快照没有独立 CLI main |
| CurvGAD | 部分 | 无正式 runner | 保留预计算适配；存在结构/shape 阻塞 |
| SpaceGNN | 有 | `methods/spacegnn_hsmad/src/runner.py` | Weibo 已形成候选；其他组合保留阻塞证据 |
| AMNet | 有但环境隔离 | `methods/amnet_hsmad/run_smoke.py` / `run_full.py` | 固定官方源码与适配 runner 已恢复；需独立 CUDA 11 ABI 环境 |
| PMP | 有 | `methods/pmp_hsmad/src/runner.py` | HSMAD 冻结协议适配候选 |

详细入口见 [docs/BASELINE_SCRIPTS.md](docs/BASELINE_SCRIPTS.md)。

## 7. 环境准备

不同历史方法并不一定共用同一套依赖。DGL 系列候选通常需要：

- Python 3.10；
- PyTorch；
- 与 PyTorch/CUDA 匹配的 DGL；
- NumPy、SciPy、pandas、scikit-learn、SymPy。

AMNet 使用独立的旧版 PyTorch/PyG 环境，不应直接覆盖主要 DGL 环境。

服务器上已验证过的环境边界及依赖说明见 [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md)。不要把 DGL 主环境和 AMNet 的旧版 PyG 二进制扩展混装。

先确认基础依赖：

```bash
python - <<'PY'
import torch, dgl
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('dgl:', dgl.__version__)
PY
```

## 8. 代码和数据预检

### 8.1 仓库静态检查

```bash
python scripts/validate_repository.py
```

该脚本检查已发布 Python 语法、JSON 配置、关键入口和异常大文件。通过静态检查只说明仓库结构正常，不等于 GPU 训练已经通过。

### 8.2 检查单个 DGL 数据文件

```bash
python - <<'PY'
import dgl

g = dgl.load_graphs('datasets/weibo')[0][0]
print('nodes:', g.num_nodes())
print('edges:', g.num_edges())
print('feature:', tuple(g.ndata['feature'].shape))
print('label:', tuple(g.ndata['label'].shape))
for key in ('train_mask', 'val_mask', 'test_mask'):
    print(key, int(g.ndata[key].bool().sum()))
PY
```

正式实验前还应核对 feature、label、mask 和数据文件 SHA256 是否与目标冻结版本一致。

## 9. 如何运行

所有命令默认从仓库根目录以模块方式执行。主 runner 默认把当前 checkout 作为项目根目录；数据位于外部已核验工作区时，可局部设置：

```bash
export HSMAD_ROOT=/absolute/path/to/verified/workspace
```

`HSMAD_ROOT` 目录应同时包含 `datasets/`，并将结果写入该工作区的 `results/`。不要在不了解输出隔离逻辑的情况下直接批量运行。

### 9.1 HSMAD 十 seed

```bash
python main.py \
  --dataset weibo \
  --run 10 \
  --epoch 1000 \
  --patience 100 \
  --hid_dim 64 \
  --order 2 \
  --q 0.5
```

### 9.2 GCN 单 seed

```bash
python -m methods.gcn.src.run_kipf_v2 \
  --config methods/gcn/configs/protocol_v2_weibo_formal.json \
  --seed 0
```

### 9.3 GAT-v2 单 seed

```bash
python -m methods.gat_v2_gadbench.src.run_formal \
  --config methods/gat_v2_gadbench/configs/weibo_protocol_v2_gadbench_hidden64_formal.json \
  --seed 0
```

### 9.4 GraphSAGE 单 seed

```bash
python -m methods.graphsage.src.run_formal \
  --config methods/graphsage/configs/weibo_graphsage_gadbench_h64_formal.json \
  --seed 0
```

### 9.5 BWGNN 十 seed

BWGNN runner 的 `--seeds` 使用逗号分隔：

```bash
python -m methods.bwgnn.src.run_formal \
  --config methods/bwgnn/configs/weibo_bwg_h64_smoke.json \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --run-type formal
```

配置文件名保留了历史 smoke 命名；formal runner 会用正式 `max_epoch=200`、`patience=50` 覆盖 smoke 控制字段，但仍建议运行前检查生成的 config snapshot。

### 9.6 SparseGAD 单 seed diagnostic

```bash
python -m methods.sparsegad.src.run_formal \
  --config methods/sparsegad/configs/weibo_sparsegad_h64_candidate.json \
  --run-type diagnostic \
  --seeds 0
```

### 9.7 SVM feature-only diagnostic

```bash
python -m methods.svm.src.runner \
  --dataset weibo \
  --seed 0 \
  --run-type diagnostic
```

### 9.8 SEC-GFD 示例

```bash
python -m methods.sec_gfd.src.runner \
  --dataset weibo \
  --run-type smoke \
  --seeds 0
```

### 9.9 GHRN 示例

```bash
python -m methods.ghrn.src.runner \
  --dataset weibo \
  --run-type smoke \
  --seeds 0
```

## 10. 推荐运行顺序

每个“方法 × 数据集”组合建议按以下顺序执行：

1. `python scripts/validate_repository.py`；
2. 检查数据字段、shape、mask 计数和 SHA256；
3. 运行独立 5 epoch smoke；
4. 重新加载 checkpoint，独立复算指标并检查 test leakage；
5. 从头运行 seed=0 diagnostic；
6. diagnostic 审计通过后，再从头运行 formal seed `0..9`；
7. 只汇总 `formal/OK` 记录，样本标准差使用 `ddof=1`。

不得把 smoke 或 diagnostic 混入 formal，不得删除低分 seed，也不得使用 test 指标选 checkpoint、阈值或超参数。

## 11. 已知限制

- GitHub 中不含数据二进制、冻结 mask、checkpoint、日志和历史结果文件；
- 一些配置是特定实验的冻结快照，切换数据集前必须核对；
- 静态检查无法发现 CUDA/DGL 非确定性、显存不足、ABI 不匹配或上游实现语义差异；
- MLP、CAREGNN、ChebNet、GIN、GWNN、GraphConsis、PC-GNN、SpaceGNN、AMNet 和 PMP 的实际服务器源码已恢复；“代码存在”仍不等于每个数据集都已通过 smoke/diagnostic/十 seed 审计；
- 只有通过 checkpoint 复算、mask 隔离和无泄漏审计的候选结果，才适合进入项目比较表。

## 12. 上游来源

- GADBench: <https://github.com/squareroot3/GADBench>
- BWGNN: <https://github.com/squareroot3/Rethinking-Anomaly-Detection>
- GAT: <https://github.com/PetarV-/GAT>
- GraphSAGE: <https://github.com/williamleif/GraphSAGE>
- AMNet: <https://github.com/Illyasville/AMNet>
- SparseGAD: <https://github.com/KellyGong/SparseGAD>
- SEC-GFD: <https://github.com/Sunxkissed/SEC-GFD>
- NRGL: <https://github.com/Shzuwu/NRGL>

使用数据或方法时，请同时引用相应论文与官方仓库。
