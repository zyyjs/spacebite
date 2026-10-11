# ByteFormer Token Adapter 实现与验证

## 目标与边界

`ByteFormerTokenAdapter` 使用完全相同的官方 ByteFormer-Tiny 结构和预训练权重，在
`post_transformer_norm` 之后停止，不执行 mean pooling 和 `Linear(192, 1000)`。

输出接口：

```python
features, attention_mask = adapter(byte_tensor)
```

```text
byte_tensor:     [B, L] int32/int64，真实字节为 0–255，右侧 padding=-1
features:        [B, T, 192]
attention_mask:  [B, T] int64，1=有效 token，0=padding token
```

本实现没有训练模型、替换分类头、修改 checkpoint 或修改 CoreNet 子模块。

## CoreNet 实际前向流程

源码：`corenet/corenet/modeling/models/classification/byteformer.py`。

1. 输入是 `[B,L]` 整数 JPEG 字节；
2. byte embedding 输出 `[B,L,192]`；
3. Conv1D 前临时转成 `[B,192,L]`，kernel size 8、stride 4，卷积后再转回
   `[B,T0,192]`；
4. 12 个 WindowedTransformerEncoder 均保持 `[B,T,192]`；
5. 在 transformer 0、1、3、5、7、9 后执行 window=2 的 TokenMerging，token 数取
   `ceil(T/2)`，通道保持 192；
6. `post_transformer_norm` 前后 shape 均为 `[B,T,192]`；
7. 官方 `forward()` 随后把 padding token 置零，并对 token 维进行 masked mean，得到
   `[B,192]`；
8. 官方 `Linear(192,1000)` 输出 `[B,1000]` ImageNet logits。

mask 在 `get_backbone_inputs()` 中以 `[B,L]` 创建。初始 Conv1D 后，CoreNet 对相同卷积
窗口的 mask 做归约；每个 TokenMerging 则把两个相邻 mask 做归约，所以 mask 会和 feature
使用相同的实际下采样拓扑，而不是从原始字节长度直接插值或裁剪。

## 四个真实 JPEG 的 shape

图像首先按官方验证预处理变为 224×224，再编码成 JPEG Q100。表中每一列均为实际运行
得到的 token 数；所有 feature 的最后一维均为 192。

| JPEG bytes | embedding T | Conv1D T | merge@0 | merge@1 | merge@3 | merge@5 | merge@7 | merge@9/final T |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42,448 | 42,448 | 10,611 | 5,306 | 2,653 | 1,327 | 664 | 332 | 166 |
| 45,260 | 45,260 | 11,314 | 5,657 | 2,829 | 1,415 | 708 | 354 | 177 |
| 51,483 | 51,483 | 12,869 | 6,435 | 3,218 | 1,609 | 805 | 403 | 202 |
| 58,921 | 58,921 | 14,729 | 7,365 | 3,683 | 1,842 | 921 | 461 | 231 |

统一 batch 输出：

```text
features.shape       = [4, 231, 192]
attention_mask.shape = [4, 231]
valid token counts   = [166, 177, 202, 231]
padding token counts = [65, 54, 29, 0]
```

## Padding 问题复现与原因

### 1. 字节 mask 没有被修改

上游代码：

```python
mask = torch.zeros_like(x, dtype=torch.float)
mask[x == -1].fill_(float("-inf"))
```

布尔高级索引产生临时 tensor，`fill_()` 修改的是临时结果而非原始 `mask`。实测 4 个不同
长度样本的字节 mask 没有出现任何 `-inf`，经过完整 backbone 后每个样本的 masked token
数量仍全部为 0。

adapter 中使用可追踪的项目代码：

```python
internal_mask = torch.zeros_like(byte_tensor, dtype=torch.float32)
internal_mask.masked_fill_(byte_tensor.eq(-1), float("-inf"))
```

### 2. 混合长度 batch 的窗口边界依赖

CoreNet 的 `WindowedTransformerEncoder.forward()` 计算了包含 padding 和 shifted-window
边界的 `total_mask`，但 pinned commit 实际调用父类 attention 时传入的是原始
`attn_mask`。逐层对照显示，单样本与混合 batch 在 embedding 和卷积后完全一致，误差在
带 `window_shift=64` 的 transformer 3 首次明显出现。

若直接改变 CoreNet 让它使用完整 `total_mask`，无 padding 输入的输出也会变化，因而不能
同时满足“保持官方 checkpoint 的原始分类路径”这一约束。最终采用以下可复现策略：

1. 检查每行 `-1` 只能构成右侧连续后缀；
2. 按真实 JPEG 长度取出每条 byte stream；
3. 每条样本独立通过同一官方 embedding、Conv1D、12 层 Transformer、6 次
   TokenMerging 和 `post_transformer_norm`；
4. 只在最终 token feature 层补齐到 batch 最大 T；
5. 根据每条实际 backbone 输出长度生成二值 attention mask。

所以有效 token 数来自实际卷积和 TokenMerging，并非根据原始字节 mask 做近似缩放。
这种 correctness-first 实现会逐样本执行 backbone，速度低于完全向量化 batch，但能同时
保持官方单样本结果和 padding 不变性。

## 验证结果

| 检查 | 最大绝对误差 | 平均绝对误差 |
| --- | ---: | ---: |
| 官方 logits vs adapter feature 经官方 pooling/classifier 重建 | 0.0 | 0.0 |
| 同一 JPEG：单样本 vs 四样本混合 batch 的有效 token | 0.0 | 0.0 |
| 同一 JPEG：原长度 vs 额外添加 4,096 个右侧 padding | 0.0 | 0.0 |

其他结果：

- features 和重建 logits 中没有 NaN/Inf；
- checkpoint 使用 `strict=True`，`missing_keys=[]`、`unexpected_keys=[]`；
- checkpoint SHA-256 仍为
  `13646ae929d20f3fb48f0a25e0a945dd59ad070746dbf651c6b2a64856867d82`；
- 推理前后模型参数摘要一致；
- 官方分类头仍为 `Linear(192,1000)`；
- `cuda:0` 验证峰值显存为 658,080,768 bytes，约 627.6 MiB。

机器可读验证报告：

```text
/data/students/Zhang_jinsong/spacebyte/result/byteformer/token_adapter/verification.json
```

自动化测试：

```bash
python -m pytest -q tests/test_byteformer_token_adapter.py
```

覆盖输出 shape、分类路径等价、混合长度 batch、单样本/batch 一致性、padding 长度不变性
和官方 checkpoint 完整性。
