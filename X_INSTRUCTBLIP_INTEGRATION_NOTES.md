# X-InstructBLIP 源码研究与 ByteFormer 接入笔记

## 1. 代码来源

- 论文：X-InstructBLIP: A Framework for Aligning Image, 3D, Audio, Video to
  LLMs and its Emergent Cross-modal Reasoning
- 官方仓库：`https://github.com/salesforce/LAVIS.git`
- 本地子模块：`/home/Zhang_jinsong/spacebyte/lavis`
- 固定提交：`05cb670f47296522e24e7727e366b4a997d28582`
- 许可证：BSD-3-Clause
- 状态：上游仓库已于 2026-09-18 归档为只读

X-InstructBLIP 不是单独的 GitHub 仓库，而是 LAVIS 中的一个项目。论文项目说明位于：

```text
lavis/projects/xinstructblip/README.md
```

核心模型实现位于：

```text
lavis/lavis/models/blip2_models/blip2_vicuna_xinstruct.py
```

训练和评估配置位于：

```text
lavis/lavis/projects/xinstruct_blip/train/
lavis/lavis/projects/xinstruct_blip/eval/
lavis/lavis/configs/models/blip2/blip2_xinstruct_vicuna7b.yaml
lavis/lavis/configs/models/blip2/blip2_xinstruct_vicuna13b.yaml
```

## 2. X-InstructBLIP 的核心结构

每一种模态都有独立的编码器和对齐模块，但共享同一个冻结 LLM：

```text
模态输入
   ↓
冻结的模态编码器
   ↓
可训练 LayerNorm
   ↓
32 个可训练 Query Tokens + instruction-aware Q-Former
   ↓
可训练 LLM Projection
   ↓
模态提示词，例如 "image:"
   ↓
冻结的 Vicuna LLM
   ↓
文本回答
```

论文实现支持 image、video、audio 和 3D point cloud：

| 模态 | 默认编码器 |
| --- | --- |
| image | EVA-CLIP ViT-G/14 |
| video | 逐帧 EVA-CLIP ViT-G/14 |
| audio | BEATs |
| 3D point cloud | ULIP PointBERT |

LLM 使用 Vicuna v1.1 7B 或 13B。默认情况下，模态编码器和 LLM 都冻结，主要训练：

- 模态 LayerNorm；
- Query Tokens；
- 模态专属 Q-Former；
- Q-Former 到 LLM embedding space 的线性投影。

Q-Former同时接收自然语言 instruction 和模态 token，所以提取的是与问题相关的表示，
而不是固定的全局特征。

## 3. 训练目标

X-InstructBLIP 不使用我们此前的 9 维 Masked BCE。它把任务转换成文本生成：

```text
输入：模态 token + instruction
目标：文本答案
损失：只作用在答案 token 上的 causal language modeling cross-entropy
```

LLM 的输入 instruction token 和模态 query token 对应位置使用 `target=-100`，不计入语言
建模损失；只让模型学习生成目标回答。

论文的关键做法是分别训练每种模态的对齐模块，不要求在同一批次联合训练多种模态。
各模态投射到同一个冻结 LLM embedding space 后，可以在推理时组合，从而产生跨模态推理
能力。

## 4. ByteFormer 不能接分类 logits

当前 SpatialSense ByteFormer 的输出路径是：

```text
JPEG bytes
  → byte embedding
  → Conv1D token reduction
  → 12 层 windowed Transformer
  → token mean pooling
  → Linear(192, 9)
```

`Linear(192, 9)` 的 9 维 logits 只表示 SpatialSense 分类结果，信息量不足，也失去了 token
级结构。正确的接入点是 `post_transformer_norm` 之后、mean pooling 和 classifier 之前：

```text
ByteFormer token features: [batch, token_count, 192]
ByteFormer attention mask:  [batch, token_count]
```

这些 token 再输入 Q-Former：

```text
JPEG bytes
  → 冻结 ByteFormer backbone
  → token features + padding mask
  → instruction-aware Q-Former
  → Linear(768, Vicuna hidden size)
  → 冻结 Vicuna
  → Yes/No 或自然语言答案
```

## 5. Token 数量适配

当前 JPEG Q100 数据平均约 39,547 bytes。ByteFormer 使用 kernel size 8、stride 4 的
初始 token reduction，并在主干中继续进行多次 token merging。初始卷积后平均约为
9,885 tokens，经过主干下采样后预计只剩约 150–300 个最终 token。使用 4 个真实测试
样本进行探测时，batch 最大初始序列经卷积后为 10,797 tokens，经完整主干后为 169 个
192 维 token。

因此应使用最后一层 ByteFormer token，而不是把初始近万 token 直接交给 Q-Former。
最终 token 数量对 32-query Q-Former 是可管理的。

## 6. 必须处理的接口差异

### 6.1 输入数据类型

原 X-InstructBLIP image encoder 接收：

```text
[batch, 3, height, width] float tensor
```

ByteFormer 接收：

```text
[batch, byte_length] int32 tensor，使用 -1 padding
```

需要实现一个 ByteFormer encoder adapter，并让数据集直接读取 JPEG 文件字节。

### 6.2 Padding mask

X-InstructBLIP 原实现假设普通 image encoder 的全部输出 token 有效，因此直接创建全 1
attention mask。ByteFormer 的 JPEG 长度可变，必须将主干产生的真实 padding mask 传入
Q-Former，否则 Q-Former 会关注补齐 token。

推荐把 encoder 接口扩展为返回：

```python
features, attention_mask = byteformer_encoder(byte_tensor)
```

并让 X-InstructBLIP 在 encoder 返回 tuple 时使用真实 mask。

另外，实测发现当前 CoreNet `ByteFormer.get_backbone_inputs()` 中：

```python
mask[x == -1].fill_(float("-inf"))
```

在当前 PyTorch 2.3 环境下没有原地修改原始 `mask`。4 个字节长度分别为 42,742、
43,193、40,077、37,369 的样本，在同一 batch 中都被错误报告为 169 个有效最终 token。
接入前必须改为显式赋值，例如：

```python
mask = torch.zeros_like(x, dtype=torch.float)
mask.masked_fill_(x.eq(-1), float("-inf"))
```

并增加不同 JPEG 长度 batch 的 mask 回归测试。该问题也可能影响此前 ByteFormer 分类微调，
后续应在修复后重新跑一个对照实验；目前没有直接修改 CoreNet 子模块。

### 6.3 特征维度

EVA-CLIP 默认输出 1,408 维特征，ByteFormer-Tiny 输出 192 维。Q-Former 的
`encoder_width` 支持设置为 192，其 cross-attention 会完成 192 维 encoder features 到
768 维 BERT/Q-Former hidden states 的映射，因此不需要伪装成 1,408 维。

### 6.4 模态注册方式

原类把 `image/pc/video/audio` 参数显式写在构造函数中。接入 ByteFormer 有两条路线：

1. 把 ByteFormer 当成新的 `byte` 模态，增加 `init_byte_encoder` 和对应配置字段；
2. 把 ByteFormer 当成一种 image encoder，实现 `image_model: byteformer` 分支。

推荐第一条路线。它能保留原始 RGB image 模态，未来可以同时输入 RGB image token 与
JPEG-byte token，真正利用 X-InstructBLIP 的多模态组合能力。

## 7. SpatialSense 指令格式建议

原来一个对象对一次输出 9 个 logits。接入 VLM 后，建议按已观察 predicate 展开为一条
True/False 文本指令。17,498 个已观察标签可形成 17,498 个指令样本。

示例：

```text
byte: <ByteFormer query tokens>
USER: In the image, the subject in the red box is the cup and the object in the
blue box is the table. Is the cup on the table? Answer yes or no.
ASSISTANT: yes
```

这种格式有三个优点：

- 与 SpatialSense 原始 binary relation annotation 一致；
- 不需要把未标注关系当作负例；
- 可以直接使用 causal language modeling loss。

第一版应限制回答为 `yes` 或 `no`，先建立可量化基线，再尝试解释性自然语言回答。

## 8. 推荐的训练阶段

### 阶段 A：接口验证

- 加载 JPEG Q100 ByteFormer；
- 删除/跳过 9 维分类头；
- 验证 token feature 和 mask 的 shape；
- 使用极小 LLM 或 mock LLM 检查梯度链路，避免立即下载大权重。

### 阶段 B：只训练 ByteFormer-to-LLM 对齐模块

- 冻结 ByteFormer；
- 冻结 Vicuna；
- 训练 Byte LayerNorm、Byte Q-Former、Query Tokens 和 LLM projection；
- 使用 SpatialSense yes/no instruction 数据；
- 按 validation Macro Accuracy 或 yes/no Accuracy 早停。

### 阶段 C：有限解冻

- 保持 Vicuna 冻结；
- 仅解冻 ByteFormer 最后 1–2 层；
- backbone learning rate 显著低于 Q-Former；
- 与完全冻结 ByteFormer 的结果做消融比较。

### 阶段 D：多模态组合

- 同时启用 RGB image encoder 和 ByteFormer byte encoder；
- 比较 `image only`、`byte only`、`image + byte`；
- 检查组合是否改善空间关系推理，而不是只增加参数。

## 9. 权重与环境要求

当前只下载了代码，没有下载 X-InstructBLIP、EVA-CLIP、BEATs 或 Vicuna 权重。

正式运行还需要：

- Vicuna v1.1 7B/13B 本地权重；
- X-InstructBLIP 对应 Q-Former checkpoint，或从 BLIP-2 Q-Former 初始化；
- Hugging Face/BERT tokenizer 文件；
- 若启用其他模态，还需要 EVA-CLIP、BEATs 或 ULIP 权重。

官方推荐 40 GB GPU。服务器的 48 GB GPU 理论上可运行 Vicuna-7B 方案，但应先进行显存
探测，不直接开始全量训练。

LAVIS 依赖较旧且固定了多个版本：

- `transformers==4.33.2`
- `timm==0.4.12`
- `fairscale==0.4.4`
- `open3d==0.13.0`

当前 ByteFormer 环境尚未安装这些依赖。为避免破坏已验证的 CoreNet 环境，后续应新建
独立 Conda 环境，而不是直接修改现有 `byteformer` 环境。

## 10. 当前结论

X-InstructBLIP 可以作为 ByteFormer 接入 VLM 的基础框架，但不是更换一个配置项就能直接
运行。核心改造包括：

1. 为 ByteFormer 暴露分类头前的 token features；
2. 修复、保留并传递可变长度字节序列的 padding mask；
3. 注册新的 `byte` 模态和 192 维 Q-Former cross-attention 输入；
4. 将 SpatialSense 标注转成 yes/no instruction 数据；
5. 使用冻结 ByteFormer + 冻结 LLM、仅训练对齐模块作为第一版基线；
6. 为归档的旧版 LAVIS 创建隔离环境并固定依赖。

这一方案保留了 ByteFormer 直接处理文件字节的特点，同时使用 X-InstructBLIP 的
instruction-aware Q-Former 把字节特征转成 LLM 可以理解的 token 表示。
