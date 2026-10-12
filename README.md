# Spacebyte

本分支提供 Apple 官方预训练 ByteFormer 基线的严格加载与前向验证。当前模型是
ImageNet JPEG Q100、卷积核 `k=8`、窗口大小 `w=128` 的 ByteFormer-Tiny，保留官方
`Linear(192, 1000)` ImageNet 分类头；不加载 SpatialSense 九分类微调权重。

SpatialSense 九分类实现完整保留在分支
`archive/byteformer-spatialsense-9cls`。本分支仍保留既有实验报告作为历史记录，但这些
报告和外部训练产物不会被官方基线入口加载。

## 源码与固定版本

- CoreNet 官方源码：`corenet/` 子模块，固定提交
  `b205775948b27f71c55630984c42b98fff4297d9`；
- 官方模型配置：
  `corenet/projects/byteformer/imagenet_jpeg_q100/conv_kernel_size=8.yaml`；
- LAVIS/X-InstructBLIP 源码：`lavis/` 子模块；
- X-InstructBLIP 接入研究：`X_INSTRUCTBLIP_INTEGRATION_NOTES.md`。

首次克隆时初始化全部子模块：

```bash
git clone --recurse-submodules https://github.com/zyyjs/spacebite.git
```

已克隆仓库可运行：

```bash
git submodule update --init --recursive
```

## 官方权重

当前服务器上的官方 checkpoint：

```text
/data/students/Zhang_jinsong/spacebyte/models/byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt
```

来源：

```text
https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/multimodal_classification/imagenet_jpeg_q100_k8_w128.pt
```

SHA-256：

```text
13646ae929d20f3fb48f0a25e0a945dd59ad070746dbf651c6b2a64856867d82
```

加载器会先校验 SHA-256，然后使用 `strict=True` 加载全部 169 个 checkpoint tensor；
不替换分类头，也不接受缺失或多余参数。

## 环境

服务器现有环境：

```bash
conda activate /data/students/Zhang_jinsong/spacebyte/envs/byteformer
```

也可从仓库配置重建：

```bash
conda env create --file environment.yml
```

## Forward smoke test

选择任意可读取的图像。验证入口会严格按照官方 YAML 执行验证预处理：保持长宽比将短边
缩放到 256、中心裁剪为 224×224、转换为 tensor，再使用 PIL 重新编码为 JPEG Q100
字节，最后送入 ByteFormer。

```bash
python scripts/verify_official_byteformer.py \
  --image /path/to/example.jpg \
  --device cuda:0 \
  --output /data/students/Zhang_jinsong/spacebyte/result/byteformer/official_baseline/smoke_test.json
```

该命令只做 checkpoint 校验、严格加载和一次推理，不训练、不替换分类头、不修复 padding
mask，也不加载任何 SpatialSense checkpoint。

## Token features 与 attention mask

本分支新增的 `ByteFormerTokenAdapter` 绕过官方 mean pooling 和
`Linear(192, 1000)`，输出：

```text
features:       [B, T, 192]
attention_mask: [B, T]，1=有效 token，0=padding token
```

官方分类头仍原样保留。最小验证命令：

```bash
python scripts/verify_byteformer_token_adapter.py \
  --config corenet/projects/byteformer/imagenet_jpeg_q100/conv_kernel_size=8.yaml \
  --checkpoint /data/students/Zhang_jinsong/spacebyte/models/byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt \
  --image-dir /path/to/jpeg-directory \
  --device cuda:0
```

实现、源码流程、padding 问题复现和测试结果详见
`BYTEFORMER_TOKEN_ADAPTER.md`。

## Random Byte Q-Former bridge

`ByteQFormerBridge` 将冻结 ByteFormer 的 `[B,T,192]` token features 与文本 instruction
输入随机初始化的 12 层 LAVIS Q-Former，通过 32 个可学习 query tokens 得到
`[B,32,768]`，再投影到 Vicuna-7B embedding 宽度 `[B,32,4096]`。本阶段仅验证结构、
mask 和梯度传播；没有下载 BLIP-2 或 Vicuna 权重，也没有训练模型。

独立环境、验证命令、参数量、显存和兼容性说明见
`BYTE_QFORMER_BRIDGE.md`。

## 历史资料

- `TRAINING_PLAN.md`：此前 SpatialSense 九分类训练计划；
- `IMPLEMENTATION_STATUS.md`：此前实现状态；
- `BYTEFORMER_SPATIALSENSE_EXPERIMENT_REPORT.md`：此前实验与测试结果；
- `X_INSTRUCTBLIP_INTEGRATION_NOTES.md`：未来接入 X-InstructBLIP 的研究笔记。

这些文档仅作记录。九分类源代码和配置请从归档分支读取。

## Git 安全约定

数据集、模型权重、训练产物、环境变量和访问令牌不得提交到仓库。需要共享的小型示例
数据应先脱敏，并存放在单独的样例目录中。
