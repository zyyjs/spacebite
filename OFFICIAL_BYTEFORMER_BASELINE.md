# 官方预训练 ByteFormer 基线验证

## 基线定义

本分支使用 Apple CoreNet 官方的 ByteFormer-Tiny：

- 数据/任务：ImageNet 1000 类图像分类；
- 文件编码：JPEG，quality 100；
- 初始 token reduction：Conv1D kernel size 8、stride 4；
- Transformer window size：128；
- embedding dimension：192；
- 最大 token 数：50,000；
- 分类头：`Linear(192, 1000)`；
- 官方公布的 ImageNet Top-1：67.64%。

没有替换分类头，没有训练模型，也没有加载 SpatialSense 微调 checkpoint。

## 源码、配置和权重

| 项目 | 值 |
| --- | --- |
| CoreNet commit | `b205775948b27f71c55630984c42b98fff4297d9` |
| 官方配置 | `/home/Zhang_jinsong/spacebyte/corenet/projects/byteformer/imagenet_jpeg_q100/conv_kernel_size=8.yaml` |
| 配置 SHA-256 | `d7fe827c1bea557c719e201e44a58dfd2966bcd64535c49994cee8a885ef24f8` |
| checkpoint | `/data/students/Zhang_jinsong/spacebyte/models/byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt` |
| checkpoint SHA-256 | `13646ae929d20f3fb48f0a25e0a945dd59ad070746dbf651c6b2a64856867d82` |
| checkpoint tensors | 169 |
| 参数总量 | 15,920,680 |

权重来源：

```text
https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/multimodal_classification/imagenet_jpeg_q100_k8_w128.pt
```

## Checkpoint 加载结果

使用 `model.load_state_dict(state_dict, strict=True)`：

```text
missing keys: []
unexpected keys: []
classifier input dimension: 192
classifier output dimension: 1000
loaded state_dict exactly matches checkpoint: true
```

因此模型的全部参数均由官方 checkpoint 覆盖，不存在仍为随机初始化的 backbone 参数，
也没有用 `strict=False` 跳过任何加载错误。

## Forward smoke test

测试输入采用一张真实 JPEG 图像，并通过 CoreNet 官方配置对应的验证流水线处理：

```text
RGB image
→ bicubic resize（短边 256，保持长宽比）
→ center crop 224 × 224
→ float tensor [3, 224, 224]
→ PIL JPEG quality=100 编码
→ int32 JPEG bytes [1, 35725]
→ ByteFormer
→ ImageNet logits [1, 1000]
```

运行结果：

| 项目 | 结果 |
| --- | --- |
| 设备 | `cuda:0` |
| 模型输入 shape | `[1, 35725]` |
| 模型输入 dtype | `torch.int32` |
| 模型输出 shape | `[1, 1000]` |
| 模型输出 dtype | `torch.float32` |
| NaN/Inf | 无 |
| SpatialSense checkpoint | 未加载 |

机器可读报告保存在：

```text
/data/students/Zhang_jinsong/spacebyte/result/byteformer/official_baseline/smoke_test.json
```

本轮没有修复 padding mask，没有绕过官方分类头，也没有接入 Q-Former、LAVIS 或 Vicuna。
