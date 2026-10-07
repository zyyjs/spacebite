# SpatialSense 微调 JPEG Q100 ByteFormer 训练计划

## 1. 目标

使用 SpatialSense 数据集微调 Apple ByteFormer，将指定主体—客体对的图像表示映射为 9 种空间关系的预测概率。

基础模型与数据位置：

- CoreNet 源码：`corenet/`
- 预训练模型：`/data/students/Zhang_jinsong/spacebyte/models/byteformer_jpeg_q100_k8_w128/`
- SpatialSense：`/data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense/`
- Conda 环境：`/data/students/Zhang_jinsong/spacebyte/envs/byteformer/`

基础模型采用：

- ByteFormer-Tiny
- ImageNet JPEG Q100 预训练权重
- Conv1D kernel size：8
- Transformer window size：128
- 特征维度：192

## 2. 数据集现状

SpatialSense 的完整统计如下：

| 划分 | 图像数 | 关系标注数 |
| --- | ---: | ---: |
| train | 7,645 | 11,238 |
| valid | 1,553 | 2,638 |
| test | 2,371 | 3,622 |
| 合计 | 11,569 | 17,498 |

关系正负样本各 8,749 条。9 种关系为：

1. `above`
2. `behind`
3. `in`
4. `in front of`
5. `next to`
6. `on`
7. `to the left of`
8. `to the right of`
9. `under`

按照图像、主体名称和边界框、客体名称和边界框聚合后，共有 17,092 个有序主体—客体对：

- 16,713 对只标注了 1 种关系
- 354 对标注了 2 种关系
- 24 对标注了 3 种关系
- 1 对标注了 5 种关系
- 没有任何对象对完整标注全部 9 种关系

因此，未标注的关系不能视为负样本，也不适合使用互斥的 9 类 softmax。

## 3. 任务定义

采用 9 维多标签输出，每个维度对应一种空间关系。模型输入不包含待查询的 predicate，因此能够一次输出全部关系的概率。

固定类别顺序：

```text
0 above
1 behind
2 in
3 in front of
4 next to
5 on
6 to the left of
7 to the right of
8 under
```

每个主体—客体对生成两个 9 维向量：

- `target`：已知关系的真假标签
- `mask`：该关系是否具有人工标注

例如只有 `on=True` 被标注：

```text
target = [0, 0, 0, 0, 0, 1, 0, 0, 0]
mask   = [0, 0, 0, 0, 0, 1, 0, 0, 0]
```

使用带掩码的二元交叉熵：

```text
loss = sum(mask * BCEWithLogits(logits, target)) / sum(mask)
```

只有人工标注过的位置参与反向传播。

## 4. 数据预处理

不修改原始 SpatialSense。处理结果写入独立目录：

```text
/data/students/Zhang_jinsong/spacebyte/processed/
└── spatialsense_byteformer_q100/
    ├── images/
    ├── manifests/
    │   ├── train.jsonl
    │   ├── valid.jsonl
    │   └── test.jsonl
    └── metadata.json
```

处理步骤：

1. 读取并校验 `annotations.json`。
2. 排除 macOS 生成的 `._*` 元数据文件。
3. 根据 URL 或本地路径匹配图像。
4. 按有序主体—客体对聚合已有关系标注。
5. 保留官方 `train/valid/test` 划分，不重新随机切分。
6. 将图像等比例缩放并填充至 `224×224`。
7. 同步变换主体和客体边界框。
8. 使用红色标记 subject，蓝色标记 object。
9. 重新编码为 JPEG Quality 100。
10. 写入 `target`、`mask`、原图信息、边界框和样本标识。

预处理后必须核对：

- 图像和标注均无缺失
- 划分数量与原始数据一致
- 边界框在图像范围内
- 每个样本至少有一个 `mask=1`
- 同一 predicate 不存在冲突标签
- 生成结果可通过固定随机种子复现

## 5. 数据增强

第一版采用保守增强策略：

- 允许轻微亮度和对比度变化
- 不使用可能裁掉主体或客体的 `RandomResizedCrop`
- 不使用 MixUp 或 CutMix
- 不随意改变长宽比

如果使用水平翻转，必须同时：

- 变换主体和客体边界框
- 交换 `to the left of` 与 `to the right of` 的 target
- 交换 `to the left of` 与 `to the right of` 的 mask

验证集和测试集不使用随机增强。

## 6. 模型改造

加载 JPEG Q100 官方配置与预训练权重：

```text
/data/students/Zhang_jinsong/spacebyte/models/
└── byteformer_jpeg_q100_k8_w128/
    ├── imagenet_jpeg_q100_k8_w128.pt
    └── config.yaml
```

执行顺序：

1. 按官方配置构建 1,000 类 ByteFormer。
2. 严格加载完整 ImageNet 预训练权重。
3. 将原分类头 `Linear(192, 1000)` 替换为 `Linear(192, 9)`。
4. 初始化新的 9 维分类头。
5. 保留字节嵌入、位置编码和 Transformer 主干参数。

输出为 9 个独立 logits，推理时分别应用 sigmoid。

项目代码放在主仓库中，不直接修改 CoreNet 子模块。计划结构：

```text
configs/
└── spatialsense_byteformer_q100.yaml
scripts/
├── prepare_spatialsense.py
├── train.py
└── evaluate.py
src/spacebyte/
├── data/spatialsense.py
├── losses/masked_bce.py
└── models/spatialsense_byteformer.py
tests/
```

## 7. 采样与批处理

不同关系的样本数差异较大，例如 `on` 明显多于 `to the right of`。训练时使用按关系频率加权的采样策略，避免高频关系主导优化。

JPEG 文件字节长度可变，同一批次会补齐到最长序列。为减少无效 padding：

- 预先记录每个样本的 JPEG 字节长度
- 按字节长度分桶
- 在桶内构建批次
- 保持训练阶段桶顺序随机化

## 8. 训练阶段

### 8.1 数据管线检查

- 可视化不少于 50 个随机样本
- 覆盖全部 9 种关系
- 覆盖 JPG 和 PNG 来源图像
- 检查重叠框、边界框靠近图像边缘等情况
- 测试左右翻转标签变换

### 8.2 最小冒烟测试

- 使用 64–256 个训练样本
- 单张 GPU
- 训练 1 个 epoch
- 验证前向、反向、保存和恢复 checkpoint
- 确认无 NaN/Inf

### 8.3 小样本过拟合测试

- 使用约 32 个固定样本
- 关闭随机增强
- 训练到损失显著下降
- 验证模型、标签和损失实现可以正常学习

### 8.4 分类头预热

- 冻结 ByteFormer 主干
- 只训练新分类头
- 训练 3–5 个 epoch
- 分类头初始学习率约 `1e-3`

### 8.5 全量微调

建议初始设置：

| 参数 | 建议值 |
| --- | --- |
| epochs | 20–30 |
| backbone learning rate | `1e-5` |
| head learning rate | `1e-4` |
| optimizer | AdamW |
| weight decay | `0.05` |
| scheduler | Cosine |
| warmup | 总步数的 5% |
| precision | BF16 mixed precision |
| gradient clipping | `1.0` |
| random seed | 固定并记录 |

先在单卡上从 `batch_size=16` 开始探测显存。稳定后再根据实际字节长度和显存占用调整至每卡 16–32，并使用 4 卡 DDP。若显存不足，优先使用梯度累积，不改变输入编码。

## 9. 评估方案

只在 `mask=1` 的已标注位置计算指标。

每种关系分别报告：

- Accuracy
- Balanced Accuracy
- AUROC
- Precision
- Recall
- F1

汇总指标：

- 9 种关系的 Macro Average
- 所有已标注位置的 Micro Average
- 验证损失

每种关系的决策阈值只在 `valid` 集上选择，最终固定阈值并在 `test` 集评估一次。

由于数据没有完整的 9 维标签，不使用以下指标：

- 9 维完全匹配准确率
- 将未标注关系作为负类得到的 mAP
- 以缺失标签补零后计算的普通多标签准确率

## 10. 实验顺序

1. 红/蓝边界框 + JPEG Q100 + 9 维 masked BCE（主实验）
2. 不标记边界框（定位信息消融实验）
3. 输入 predicate 的二分类模型（与原始任务一致的基线）
4. 加入主体和客体名称嵌入
5. 在资源允许时对比 TIFF 预训练权重

查询 predicate 的二分类基线不可省略。9 维直接预测的监督非常稀疏，该基线可以帮助判断性能差异来自任务定义还是模型实现。

## 11. 输出与 checkpoint

建议输出目录：

```text
/data/students/Zhang_jinsong/spacebyte/runs/
└── byteformer_q100_spatialsense/
    ├── checkpoints/
    ├── logs/
    ├── metrics/
    ├── predictions/
    └── resolved_config.yaml
```

至少保存：

- 最佳验证集 Macro AUROC checkpoint
- 最佳验证集 Macro Balanced Accuracy checkpoint
- 最后一轮 checkpoint
- 优化器和 scheduler 状态
- 每种关系的验证阈值
- 完整环境和配置快照

## 12. 验收条件

正式训练开始前必须满足：

- 数据预处理结果通过数量和完整性检查
- 可视化样本中的主体/客体框正确
- 左右翻转的标签交换测试通过
- 预训练权重除分类头外全部正确加载
- CUDA 冒烟测试通过
- 32 样本过拟合测试通过
- checkpoint 保存与恢复测试通过
- 训练和验证指标仅使用已标注位置
- Git 工作区中的代码、配置和说明已提交
- 数据、模型权重、环境和训练输出未提交到 Git

满足以上条件后，再启动完整单卡实验和多卡正式训练。
