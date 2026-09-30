# HSMAD 统一复现工作区

[中文](README.md) | [English](README_EN.md)

本仓库公开 HSMAD 主方法、论文 Table 1 baseline 的项目适配代码，以及统一数据划分、评价和审计所需的运行脚本。

## 仓库范围与复现口径

- `HSMAD` 是项目主方法；论文 Table 1 另有 **17 个 baseline**，二者合计 18 个表内方法。
- 除非单独注明，`methods/` 中的实现采用 `candidate_protocol_not_author_exact` 口径：代码经过项目适配并可审计，但不宣称与作者仓库逐字节一致，也不承诺复现论文中的每一位小数。
- ChebNet、GIN、GWNN、SVM、CARE-GNN 和 GraphConsis 是仓库额外保留的补充方法，不计入论文 Table 1 的 17 个 baseline。
- 数据集、冻结 mask、checkpoint、训练日志和正式实验结果因体积较大不上传 GitHub；仓库提供下载来源、代码、配置和使用说明。

## 1. 研究目标

在相同数据版本、相同 train/validation/test 划分和相同最终评价口径下，对 HSMAD 及其 baseline 进行可复算、可审计的实验：

- 所有对比实验使用同一固定数据划分，训练 seed 不得改变 split；
- 训练 seed 固定为 `0..9`；
- 训练集只参与损失计算与参数更新；
- 验证集负责 early stopping、checkpoint 和 F1 阈值选择；
- 测试集只在模型与阈值固定后计算最终 F1-Macro 和 AUROC；
- 每次运行保存 config、日志、history、checkpoint、metrics、耗时、峰值显存和 SHA256；
- 十次结果采用样本标准差，即 `ddof=1`。

### 固定划分的实现边界

- HSMAD 主入口当前在 `dataset.py` 中以 `random_state=2` 确定性地重建约 40/20/40 的 split，并覆盖图内 mask；这属于“固定可重复划分”，不是直接读取持久化 mask。
- 项目适配的 baseline runner 通常读取数据文件中的 `train_mask`、`val_mask`、`test_mask`，并在有冻结哈希的配置中核对 mask SHA256。
- 正式比较前必须确认 HSMAD 重建的 split 与目标 baseline 冻结 split 一致；不得仅凭 seed 相同推定 mask 相同。

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

上述 mask 是 baseline runner 的默认输入契约。HSMAD 的 `dataset.py` 会按固定 `random_state=2` 重建并覆盖 mask；如果目标是与 baseline 使用逐节点完全相同的 split，必须在训练前比较计数与 SHA256。

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

## 6. 方法代码覆盖范围

HSMAD 主方法入口为 `main.py`，不计入下面的 17 个 baseline。论文 Table 1 共列出 17 个 baseline；本节严格按论文名单统计，避免把仓库中的补充实验方法重复算入。

### 6.1 Table 1 baseline（17 个）

| 方法 | 代码状态 | 主要入口 | 当前说明 |
|---|---|---|---|
| MLP | 入口已提供 | `methods/mlp/src/formal_runner.py` | feature-only 项目候选 |
| GCN | 入口已提供 | `methods/gcn/src/run_kipf_v2.py` | Kipf 两层项目候选 |
| GAT | 入口已提供 | `methods/gat_v2_gadbench/src/run_formal.py` | 当前主候选；`methods/gat/` 保留较早的诊断实现，二者只计一个 baseline |
| GraphSAGE | 入口已提供 | `methods/graphsage/src/run_formal.py` | GADBench pool 项目候选 |
| AMNet | 入口已提供 | `methods/amnet_hsmad/run_full.py` | 需独立的旧版 CUDA 11/PyG 环境 |
| BWGNN | 入口已提供 | `methods/bwgnn/src/run_formal.py` | 项目适配候选 |
| GHRN | 入口已提供 | `methods/ghrn/src/runner.py` | 项目适配候选 |
| SparseGAD | 入口已提供 | `methods/sparsegad/src/run_formal.py` | 项目适配候选 |
| SEC-GFD | 入口已提供 | `methods/sec_gfd/src/runner.py` | 部分数据集候选 |
| NRGL | 代码/API 已提供 | `methods/nrgl/src/runner.py` 中的 `run(...)` | 当前快照没有独立 CLI main |
| PC-GNN | 入口已提供 | `methods/pcgnn/src/run.py` | 单关系适配候选 |
| ConsisGAD | 尚未收录 | — | 仓库当前没有 ConsisGAD 实现；不能用 GraphConsis 替代 |
| PMP | 入口已提供 | `methods/pmp_hsmad/src/runner.py` | HSMAD 冻结协议适配候选 |
| DSGAD | 入口已提供 | `methods/dsgad/src/runner.py` | 项目适配候选 |
| CurvGAD | 部分代码 | `methods/curvgad_hsmad/src/` | 已保留预计算/适配代码，尚无可执行的正式 runner |
| SpaceGNN | 入口已提供 | `methods/spacegnn_hsmad/src/runner.py` | 项目适配候选 |
| CGADM | 入口已提供 | `methods/cgadm_hsmad/src/runner.py` | HSMAD 数据适配候选 |

### 6.2 补充方法（不计入 Table 1 baseline）

| 方法 | 主要入口 | 说明 |
|---|---|---|
| ChebNet | `methods/chebnet/src/run.py` | PyG 补充候选 |
| GIN | `methods/gin/src/run.py` | PyG 补充候选 |
| GWNN | `methods/gwnn/src/run.py` | 论文公式补充候选 |
| SVM | `methods/svm/src/runner.py` | feature-only 补充候选 |
| CARE-GNN | `methods/caregnn/src/run.py` | 单关系适配补充候选 |
| GraphConsis | `methods/graphconsis/src/run.py` | 单关系适配补充候选；与 Table 1 的 ConsisGAD 不是同一方法 |

详细入口见 [docs/BASELINE_SCRIPTS.md](docs/BASELINE_SCRIPTS.md)。

## 7. 环境准备

不同历史方法并不一定共用同一套依赖，因此仓库有意不提供一个会把所有方法混装到一起的根目录 `requirements.txt`。DGL 系列候选通常需要：

- Python 3.10；
- PyTorch；
- 与 PyTorch/CUDA 匹配的 DGL；
- NumPy、SciPy、pandas、scikit-learn、SymPy。

AMNet 使用独立的旧版 PyTorch/PyG 环境，不应直接覆盖主要 DGL 环境。

服务器上已验证过的版本、隔离环境创建模板和安装核对步骤见 [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md)。不要把 DGL 主环境、标准 PyG 环境和 AMNet 的旧版 PyG 二进制扩展混装。

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

所有命令默认从仓库根目录执行。推荐入口均使用仓库相对路径，不包含开发机器的绝对路径。

部分较新的适配 runner 通过 `methods.project_paths.project_root()` 解析根目录；这些 runner 的数据与结果位于外部已核验工作区时，可对单条命令局部设置：

```bash
export HSMAD_ROOT=/absolute/path/to/verified/workspace
```

`HSMAD_ROOT` 目录应同时包含 `datasets/`，并将结果写入该工作区的 `results/`。该变量并非所有历史 runner 的通用功能：HSMAD 主入口及仍以 `Path(__file__)` 定位根目录的 runner 应把数据放在当前 checkout 的 `datasets/`。运行前可在入口源码中检查是否导入 `methods.project_paths`。

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

GraphSAGE formal 强制要求确定性环境变量：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
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

## 12. 上游来源与本地 provenance

- GADBench: <https://github.com/squareroot3/GADBench>
- BWGNN: <https://github.com/squareroot3/Rethinking-Anomaly-Detection>
- GAT: <https://github.com/PetarV-/GAT>
- GraphSAGE: <https://github.com/williamleif/GraphSAGE>
- AMNet: <https://github.com/Illyasville/AMNet>
- SparseGAD: <https://github.com/KellyGong/SparseGAD>
- SEC-GFD: <https://github.com/Sunxkissed/SEC-GFD>
- NRGL: <https://github.com/Shzuwu/NRGL>
- CGADM: <https://github.com/weicy15/CGADM>

并非每个项目适配都能对应一个已独立核验的公开官方仓库。其余方法的固定 commit、配置来源或官方快照证据保存在对应的 `methods/<method>/configs/`、`methods/<method>/audit/` 或 `methods/<method>/official_snapshot/` 中；缺少已核验 URL 时不在此猜测链接。ConsisGAD 当前没有代码，也没有以 GraphConsis 替代。

## 13. 引用说明

本仓库主要用于学术研究与实验复现。使用本项目时，建议引用 HSMAD 论文；使用具体数据集或 baseline 时，请同时引用相应论文或官方仓库。第三方代码的许可说明以其原始仓库为准。
