# JPEG Q100 ByteFormer 微调 SpatialSense 实验报告

## 1. 实验概览

本实验使用 Apple/CoreNet 的 ByteFormer-Tiny 模型以及 ImageNet JPEG Q100 预训练权重，
在 SpatialSense 数据集上预测指定 subject–object 有序对象对之间的空间关系。

实验将原本的 ImageNet 1,000 类分类头替换为 9 维多标签分类头。输入不是解码后的
RGB 像素张量，而是重新编码后的 JPEG 文件字节序列。为了让模型知道需要判断哪一对
对象，在图像中用红框标记 subject，用蓝框标记 object。

最终结果如下：

| 数据划分 | checkpoint | Macro Balanced Accuracy | Macro F1 | Macro AUROC |
| --- | --- | ---: | ---: | ---: |
| Validation | Epoch 12 `best.pt` | 0.5498 | 0.5507 | 0.5626 |
| Test | Epoch 12 `best.pt` | 0.5331 | 0.5403 | 0.5369 |

测试结果略高于随机基线，但整体泛化能力有限。训练 loss 从 0.6856 降至 0.0096，
验证 loss 却从 0.7515 上升至 1.8464，表明模型发生了明显过拟合。

## 2. 实验环境与文件位置

| 项目 | 位置或取值 |
| --- | --- |
| 主代码仓库 | `/home/Zhang_jinsong/spacebyte` |
| CoreNet 源码 | `/home/Zhang_jinsong/spacebyte/corenet` |
| Conda 环境 | `/data/students/Zhang_jinsong/spacebyte/envs/byteformer` |
| 原始数据 | `/data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense` |
| 预处理数据 | `/data/students/Zhang_jinsong/spacebyte/processed/spatialsense_byteformer_q100` |
| 预训练权重 | `/data/students/Zhang_jinsong/spacebyte/models/byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt` |
| 训练结果 | `/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_q100_baseline` |
| 测试结果 | `/data/students/Zhang_jinsong/spacebyte/result/byteformer/test/metrics.json` |
| 训练设备 | 单张 NVIDIA GeForce RTX 4090，`cuda:0` |
| PyTorch | 2.3.0+cu121 |

正式训练于 2026-10-07 22:56 左右开始，23:36 左右结束，总耗时约 40 分钟。

## 3. 数据集与任务定义

### 3.1 数据规模

SpatialSense 原始标注包含 11,569 张图像和 17,498 条已观察关系标签。将同一图像中
边界框完全相同的有序 subject–object 对进行聚合后，得到 17,092 个模型样本。

| Split | 图像数 | 对象对样本数 | 已观察关系标签数 |
| --- | ---: | ---: | ---: |
| Train | 7,645 | 10,989 | 11,238 |
| Validation | 1,553 | 2,564 | 2,638 |
| Test | 2,371 | 3,539 | 3,622 |
| 合计 | 11,569 | 17,092 | 17,498 |

每种关系的全数据正、负样本数量相等，但不同关系之间的标注数量相差较大：

| 关系 | 负例 | 正例 | 合计 |
| --- | ---: | ---: | ---: |
| above | 631 | 631 | 1,262 |
| behind | 1,479 | 1,479 | 2,958 |
| in | 679 | 679 | 1,358 |
| in front of | 1,087 | 1,087 | 2,174 |
| next to | 781 | 781 | 1,562 |
| on | 2,433 | 2,433 | 4,866 |
| to the left of | 513 | 513 | 1,026 |
| to the right of | 401 | 401 | 802 |
| under | 745 | 745 | 1,490 |

### 3.2 为什么采用多标签而不是 9 类 Softmax

这 9 种空间关系不是互斥类别。例如同一个对象对可以同时满足 `above`、`behind`
和 `next to`。数据中 16,713 个对象对只标注了一种关系，354 个标注两种，24 个标注
三种，1 个标注五种，没有对象对完整标注全部 9 种关系。

因此，“没有出现某关系的标注”表示未知，而不是负例。若直接采用 9 类 Softmax，
会错误地强制关系互斥；若将所有未标注位置当作 0，也会引入大量伪负标签。本实验为
每个样本构造两个 9 维向量：

- `target[j]`：第 `j` 种关系的人工真假标签；
- `mask[j]`：第 `j` 种关系是否具有人工标注。

模型输出 9 个相互独立的 logit，只有 `mask=1` 的位置参与损失和指标计算。

固定输出顺序为：`above`、`behind`、`in`、`in front of`、`next to`、`on`、
`to the left of`、`to the right of`、`under`。

## 4. 数据预处理与输入表示

每个有序对象对执行以下处理：

1. 保留 SpatialSense 官方 train/valid/test 划分；
2. 将原图转换为 RGB；
3. 等比例缩放，并以灰色像素补边到 224×224，不改变原始长宽比；
4. 同步变换 subject 和 object 边界框；
5. 用 3 像素宽红框标记 subject，用蓝框标记 object；
6. 以 JPEG quality 100 重新编码；
7. 读取 JPEG 文件的原始二进制字节，转换为 `torch.int32` 序列；
8. 一个 batch 内按最长字节序列补齐，padding 值为 `-1`。

这里的模型输入确实是 JPEG 文件字节，而不是 Pillow/torchvision 解码后的图像像素。
预处理结果为固定文件，本次正式训练没有启用随机在线图像增强。

完整数据校验结果：

- 17,092 个对象对和 17,498 个已观察标签全部通过；
- JPEG 字节长度最小 17,682，最大 74,618，平均 39,547；
- kernel size 8 的 token reduction 后，最长序列为 18,653 tokens；
- 官方配置允许的最大长度为 50,000 tokens；
- 未发现缺失文件、越界长度或无标注样本。

## 5. 模型结构与预训练权重

基础模型配置：

| 参数 | 取值 |
| --- | --- |
| 模型 | ByteFormer-Tiny |
| 预训练数据 | ImageNet |
| 输入编码 | JPEG Q100 bytes |
| Conv1D kernel size | 8 |
| Transformer window size | 128 |
| 特征维度 | 192 |
| 微调后参数量 | 15,729,417 |

模型构建顺序很重要：

1. 按官方配置构建 1,000 类 ByteFormer；
2. 使用 `strict=True` 完整加载 ImageNet 预训练参数；
3. 将 `Linear(192, 1000)` 替换为 `Linear(192, 9)`；
4. 新分类头权重使用截断正态分布初始化，标准差 0.02，bias 初始化为 0；
5. 保留预训练的字节嵌入、token reduction、位置编码和 Transformer 主干。

严格加载成功说明预训练权重与所选 ByteFormer 配置匹配。只有新的 9 维分类头是从头训练。

## 6. 损失函数

### 6.1 Sigmoid 与二元交叉熵

模型对每个关系输出一个实数 logit `z`。对应的正类概率为：

```text
p = sigmoid(z) = 1 / (1 + exp(-z))
```

单个二元标签 `y ∈ {0,1}` 的 binary cross-entropy 为：

```text
BCE(z, y) = -[y log(sigmoid(z)) + (1-y) log(1-sigmoid(z))]
```

代码使用 PyTorch 的 `binary_cross_entropy_with_logits`，直接接收 logits。它将 sigmoid
和 BCE 合并计算，比先显式计算 sigmoid 再取对数具有更好的数值稳定性。

### 6.2 Masked Binary Cross-Entropy

设 batch 中第 `i` 个样本、第 `j` 个关系的标注掩码为 `m_ij`，本实验损失为：

```text
L = Σ(i,j) m_ij · BCE(z_ij, y_ij) / Σ(i,j) m_ij
```

它具有以下含义：

- `mask=1`：该位置有人工正/负标签，参与损失；
- `mask=0`：该关系未知，损失乘以 0，不产生梯度；
- 分母是 batch 中实际可用的标注数量，而不是 `batch_size × 9`。

该设计避免了把未标注关系错误当成负例，是本数据集上最关键的训练约束之一。

验证/测试 loss 也只在已观察标签上计算。后期验证 loss 显著变大，意味着模型在部分
错误样本上给出了越来越自信的概率；交叉熵会对“高置信度错误”施加很大惩罚。

## 7. 采样与训练配置

### 7.1 加权采样

`on` 有 4,866 条标注，而 `to the right of` 只有 802 条。为了降低高频关系主导训练的
程度，本实验采用 `WeightedRandomSampler`，对含稀有关系的样本赋予更大采样权重。

若关系 `j` 在训练集中出现 `c_j` 次，则其权重与 `1/c_j` 成正比；一个样本的权重是
其所有已标注关系权重的平均值。采样为有放回采样，因此每个 epoch 仍包含 10,989 次
抽样，但某些样本可能重复，另一些可能未被抽到。

该采样只平衡“关系出现频率”，没有单独重加权正例和负例。本数据每种关系的正负数量
本身相等，因此这一点不会造成明显正负偏置。

### 7.2 正式训练超参数

| 参数 | 取值 |
| --- | ---: |
| Epochs | 25 |
| Batch size | 16 |
| DataLoader workers | 4 |
| 随机种子 | 20261007 |
| 优化器 | AdamW |
| Backbone learning rate | 1e-5 |
| 分类头预热 learning rate | 1e-3 |
| 解冻后分类头 learning rate | 1e-4 |
| Weight decay | 0.05 |
| LR scheduler | CosineAnnealingLR |
| Gradient clipping | global norm 1.0 |
| 混合精度 | BF16 |
| 设备 | cuda:0，单卡 |

训练分为两个阶段：

- Epoch 1–3：冻结 ByteFormer 主干，只训练新分类头；
- Epoch 4–25：解冻全部参数，以较小 backbone learning rate 进行全量微调。

每轮结束后在 validation split 上评估。`last.pt` 每轮覆盖保存，`best.pt` 按 validation
macro balanced accuracy 选择。本次最优 checkpoint 来自 Epoch 12。

本次实际运行没有使用多卡 DDP、梯度累积、按字节长度分桶、随机颜色增强或水平翻转。

## 8. 指标定义与解释

所有指标只在 `mask=1` 的人工已标注位置上计算。预测概率阈值固定为 0.5；概率不低于
0.5 判为正类，否则判为负类。

### 8.1 混淆矩阵术语

- TP（True Positive）：真实为正，预测也为正；
- TN（True Negative）：真实为负，预测也为负；
- FP（False Positive）：真实为负，但预测为正；
- FN（False Negative）：真实为正，但预测为负。

### 8.2 Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Accuracy 表示预测正确的比例，直观但对类别不平衡敏感。如果负例远多于正例，全部预测
为负也可能得到很高 Accuracy。本数据中每种关系的正负数量相等，因此 Accuracy 与
Balanced Accuracy 数值相同；代码仍分别计算两者，以保持评估定义清晰。

### 8.3 Balanced Accuracy

```text
Recall / TPR = TP / (TP + FN)
Specificity / TNR = TN / (TN + FP)
Balanced Accuracy = (TPR + TNR) / 2
```

Balanced Accuracy 同等重视正类召回率和负类召回率。二分类随机猜测的期望值约为 0.5。
本实验使用 validation macro balanced accuracy 选择 `best.pt`。

### 8.4 Precision、Recall 与 F1

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

Precision 衡量“预测为正的结果中有多少是真的”，Recall 衡量“真实正例中有多少被找到”。
F1 是二者的调和平均，只有 Precision 和 Recall 都较高时才会较高。本次代码直接报告 F1，
未在 JSON 中单独保存 Precision 和 Recall。F1 使用固定 0.5 阈值，因此会受阈值选择影响。

### 8.5 AUROC

ROC 曲线在所有可能的分类阈值下绘制 TPR 与 FPR 的关系，AUROC 是该曲线下面积。
它也可以理解为：随机抽取一个正例和一个负例时，模型给正例更高分数的概率。

- AUROC = 1.0：排序完全正确；
- AUROC = 0.5：与随机排序相当；
- AUROC < 0.5：排序方向比随机更差。

AUROC 不依赖单一的 0.5 阈值，适合观察模型是否学到了可排序信号。但它不直接告诉我们
部署时应选什么阈值，也不等价于准确率。

### 8.6 Micro 与 Macro 汇总

- Micro：将所有关系的已观察标签展平成一个大二分类集合后计算。标注多的关系贡献更大；
- Macro：先对 9 种关系分别计算，再对 9 个结果做等权平均。每种关系贡献相同。

由于各关系标注数量差异明显，本报告优先使用 Macro 指标判断整体空间关系能力，Micro
指标用于描述所有标注位置上的总体预测效果。

## 9. 训练前验证

正式训练前完成了以下检查：

- 5 个单元测试全部通过；
- 官方 1,000 类权重严格载入成功，并成功替换为 9 维分类头；
- 16/8 个 train/valid 样本的 GPU 冒烟训练完成，logits 形状与数值正常；
- 16 个固定样本的分类头过拟合测试中，训练 loss 最低约 0.023，证明数据、标签、
  loss、反向传播与 checkpoint 链路可学习；
- batch size 16 全模型反向传播 loss 和梯度均为有限值；
- 探测时峰值 allocated 显存约 6.00 GiB，reserved 显存约 7.17 GiB。

## 10. 训练过程与验证结果

### 10.1 关键轮次

| Epoch | 阶段 | Train loss | Validation loss | Macro Bal. Acc. | Macro F1 | Macro AUROC |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 仅分类头 | 0.6856 | 0.7515 | 0.5209 | 0.5228 | 0.5350 |
| 3 | 仅分类头结束 | 0.6281 | 0.7822 | 0.5368 | 0.5279 | 0.5538 |
| 4 | 全模型解冻 | 0.5498 | 0.8123 | 0.5289 | 0.5175 | 0.5614 |
| 6 | 全量微调 | 0.2555 | 0.9709 | 0.5470 | 0.5451 | 0.5545 |
| **12** | **最佳 checkpoint** | **0.0345** | **1.4545** | **0.5498** | **0.5507** | **0.5626** |
| 25 | 最后一轮 | 0.0096 | 1.8464 | 0.5427 | 0.5395 | 0.5584 |

最佳验证轮 Epoch 12 的完整汇总：

- observed labels：2,638；
- micro accuracy：0.5561；
- micro F1：0.5665；
- macro accuracy / balanced accuracy：0.5498；
- macro F1：0.5507；
- macro AUROC：0.5626。

### 10.2 曲线解读

Epoch 4 解冻 backbone 后，训练 loss 持续快速下降，而验证 loss 几乎单调上升。Epoch 1
到 Epoch 25：

- 训练 loss 从 0.6856 降至 0.0096，下降约 98.6%；
- 验证 loss 从 0.7515 上升至 1.8464；
- 验证 Macro Balanced Accuracy 只从 0.5209 提升到 0.5427；
- 最佳 Macro Balanced Accuracy 出现在 Epoch 12，之后没有继续稳定改善。

这说明模型能够记忆训练集，但学到的可泛化空间关系信号较弱。验证 loss 与分类指标并不
矛盾：少量高置信度错误可以显著抬高 BCE loss，同时 0.5 阈值下的正确/错误数量只发生
较小变化。

训练曲线文件：

`/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_q100_baseline/training_curves.png`

## 11. 最终测试集结果

最终测试严格使用 validation macro balanced accuracy 选出的 Epoch 12 `best.pt`，没有用
test split 选择 checkpoint。测试包含 3,539 个对象对和 3,622 个已观察标签。

### 11.1 汇总结果

| 指标 | 数值 |
| --- | ---: |
| Test masked BCE loss | 1.5183 |
| Micro Accuracy | 0.5431 |
| Micro F1 | 0.5600 |
| Macro Accuracy | 0.5331 |
| Macro Balanced Accuracy | 0.5331 |
| Macro F1 | 0.5403 |
| Macro AUROC | 0.5369 |

### 11.2 各关系结果

| 关系 | 已观察标签数 | Accuracy | Balanced Accuracy | F1 | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| above | 288 | 0.5417 | 0.5417 | 0.5600 | 0.5326 |
| behind | 508 | 0.5551 | 0.5551 | 0.5891 | 0.5653 |
| in | 338 | 0.4822 | 0.4822 | 0.5070 | 0.4803 |
| in front of | 450 | 0.5222 | 0.5222 | 0.5336 | 0.5373 |
| next to | 324 | 0.5216 | 0.5216 | 0.5289 | 0.5059 |
| on | 1,100 | 0.5809 | 0.5809 | 0.6070 | 0.6084 |
| to the left of | 178 | 0.5225 | 0.5225 | 0.4970 | 0.5343 |
| to the right of | 146 | 0.5616 | 0.5616 | 0.5362 | 0.5632 |
| under | 290 | 0.5103 | 0.5103 | 0.5035 | 0.5049 |

`on` 是表现最好的关系，AUROC 为 0.6084；`behind` 和 `to the right of` 也有弱但可见的
排序能力。`in` 的 AUROC 为 0.4803，低于随机水平；`next to` 和 `under` 接近随机。

与最佳验证结果相比，测试 Macro Balanced Accuracy 下降约 1.67 个百分点，Macro AUROC
下降约 2.57 个百分点，说明验证集上的有限收益没有完全迁移到测试集。

## 12. 结果结论与局限

### 12.1 可以确认的结论

1. JPEG Q100 ByteFormer 可以在不解码图片的情况下完成端到端空间关系微调；
2. 红/蓝边界框和 JPEG 字节中包含了一定可学习信号；
3. 模型在 `on`、`behind` 等关系上优于随机；
4. 整体测试 Macro AUROC 0.5369、Macro Balanced Accuracy 0.5331，只略高于随机；
5. 训练集与验证集曲线严重分离，全量微调方案存在明显过拟合。

### 12.2 可能原因（实验结果支持的推测）

- SpatialSense 标注稀疏，绝大多数对象对只有一个受监督关系；
- 模型容量相对 10,989 个训练对象对较大，全量解冻后容易记忆训练样本；
- ImageNet 预训练目标是整图物体分类，并不直接优化对象对空间推理；
- subject/object 信息仅通过较细的红蓝框编码，经过 JPEG 编码后可能不够突出；
- 有放回加权采样会重复稀有样本，可能进一步加重记忆；
- 本次没有在线增强、早停、阈值校准或更强正则化；
- 固定 0.5 阈值未必是每种关系的最佳决策阈值。

这些是合理解释，但需要消融实验才能确定各因素的实际贡献。

## 13. 后续改进建议

按优先级建议：

1. 使用早停，并将总训练控制在 6–12 epochs 附近；
2. 降低 backbone learning rate，例如从 1e-5 降至 1e-6，或长期冻结底层模块；
3. 提高正则化：增加 dropout、stochastic depth 或适当提高 weight decay；
4. 加入不会破坏空间标签的颜色/亮度/JPEG 质量增强；
5. 尝试更醒目的框、半透明 mask 或 subject/object 区域编码方式；
6. 在 validation split 上为每种关系单独选择阈值，再锁定阈值评估 test；
7. 比较不加 weighted sampler、仅训练分类头、部分解冻和全量解冻；
8. 重复多个随机种子并报告均值与标准差，避免单次实验偶然性；
9. 增加 RGB CNN/ViT 基线，以判断瓶颈来自 ByteFormer 还是任务构造；
10. 如继续使用字节模型，比较 JPEG Q60/Q100、边框粗细和输入分辨率。

测试集只应在方案固定后用于最终报告，不应根据当前测试结果反复调参。后续实验应继续用
validation split 选择结构、超参数和阈值。

## 14. 复现实验命令

激活环境：

```bash
cd /home/Zhang_jinsong/spacebyte
conda activate /data/students/Zhang_jinsong/spacebyte/envs/byteformer
```

预处理与验证：

```bash
python scripts/prepare_spatialsense.py \
  --config configs/spatialsense_byteformer_q100.yaml

python scripts/validate_spatialsense.py \
  --config configs/spatialsense_byteformer_q100.yaml \
  --output /data/students/Zhang_jinsong/spacebyte/result/byteformer/data_validation/report.json
```

正式训练：

```bash
python scripts/train.py \
  --config configs/spatialsense_byteformer_q100.yaml \
  --run-name spatialsense_q100_baseline
```

生成训练曲线：

```bash
python scripts/plot_metrics.py \
  --metrics /data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_q100_baseline/metrics.jsonl \
  --output /data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_q100_baseline/training_curves.png
```

最终测试：

```bash
python scripts/evaluate.py \
  --config configs/spatialsense_byteformer_q100.yaml \
  --checkpoint /data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_q100_baseline/best.pt \
  --split test \
  --device cuda:0 \
  --output /data/students/Zhang_jinsong/spacebyte/result/byteformer/test/metrics.json
```

## 15. 主要实验产物

| 产物 | 路径 |
| --- | --- |
| 训练配置 | `configs/spatialsense_byteformer_q100.yaml` |
| 完整数据校验 | `/data/students/Zhang_jinsong/spacebyte/result/byteformer/data_validation/report.json` |
| 每轮训练日志 | `/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_q100_baseline/metrics.jsonl` |
| 最佳模型 | `/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_q100_baseline/best.pt` |
| 最后一轮模型 | `/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_q100_baseline/last.pt` |
| 训练曲线 | `/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_q100_baseline/training_curves.png` |
| 最终测试报告 | `/data/students/Zhang_jinsong/spacebyte/result/byteformer/test/metrics.json` |

