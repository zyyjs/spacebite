# BLIP-2 Stage-1 initialization for Byte Q-Former

本阶段仅把官方 BLIP-2 Stage-1 checkpoint 中语义对应且 shape 完全一致的 tensor 迁移到
`ByteQFormerBridge`。没有接入或下载 Vicuna、EVA-CLIP，没有训练，也没有修改
ByteFormer、CoreNet 或 LAVIS 子模块。

## 官方 checkpoint

```text
来源  https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth
路径  /data/students/Zhang_jinsong/spacebyte/models/blip2_stage1/blip2_pretrained.pth
大小  746,998,955 bytes
SHA-256  f31f96e4a97ce9ac7a140ea1961b8a871822ae1cd0a23f0f1122fbefd70db9f0
```

checkpoint 顶层为普通 `dict`，唯一键为 `model`；`model` 是包含 346 个 tensor 的
`OrderedDict`，总计 210,178,169 个 tensor elements。扣除
`Qformer.bert.embeddings.position_ids` 这个 512-element buffer 后，state 中共有
210,177,657 个参数 elements。这里按 state dict 逐 tensor 统计，因此 tied embedding
在 checkpoint 中出现两次时也按两个 state entry 计数。

顶层模块分布：

```text
Qformer.*       336 tensors
query_tokens      1 tensor
ln_vision.*       2 tensors
vision_proj.*     2 tensors
text_proj.*       2 tensors
itm_head.*        2 tensors
temp              1 tensor
```

checkpoint 不包含 `visual_encoder.*` 权重，但包含视觉 LayerNorm、视觉/文本投影、ITM
head 和 MLM head；不包含名为 classifier 的分类头。

## 显式键名映射

迁移器只使用以下规则：

```text
query_tokens       -> query_tokens
Qformer.<suffix>   -> qformer.<suffix>
其他 checkpoint 键 -> 不映射
```

映射后还必须满足 shape 完全相同才会复制。每个已加载 tensor 在复制后使用
`torch.equal` 与源 tensor 再次核对；所有未加载的目标 tensor 也与加载前快照逐一核对。
实现没有调用 `load_state_dict(strict=False)`，也不进行 reshape、截断、补零或维度伪装。
成功迁移后 bridge 的 `initialization_mode` 会从 `random` 更新为
`blip2_stage1_partial`，并记录 checkpoint 路径与 SHA-256。

迁移器还带有 allowlist：shape mismatch 只能是第 0、2、4、6、8、10 层
cross-attention 的 key/value weight；checkpoint-only 只能是 MLM 或已知 BLIP-2 辅助
模块；target-only 只能是本项目新增 LayerNorm/projector。出现任何其他项目会直接失败。

## 检查结果与迁移统计

```text
checkpoint tensor 数                         346
checkpoint tensor elements                  210,178,169
可加载 tensor 数                             318
已加载 tensor elements                      172,709,120
已加载 parameter elements                   172,708,608
相对当前 bridge 参数迁移率                    97.230351%
相对 checkpoint tensor elements 使用率       82.172721%
shape mismatch tensor 数                     12
shape mismatch 源 elements                  12,976,128
shape mismatch 目标 elements                 1,769,472
checkpoint-only tensor 数                    16
checkpoint-only elements                    24,492,921
目标中保留原初始化 parameter elements         4,919,680
```

按模块：

| 模块 | 加载 tensor | mismatch | checkpoint-only | 目标迁移率 |
|---|---:|---:|---:|---:|
| Query Tokens | 1 | 0 | 0 | 100% |
| token/position embeddings | 5 | 0 | 0 | 100% |
| self-attention | 120 | 0 | 0 | 100% |
| text FFN | 72 | 0 | 0 | 100% |
| query FFN | 72 | 0 | 0 | 100% |
| cross-attention layer 0 | 8 | 2 | 0 | 80.0623% |
| cross-attention layer 2 | 8 | 2 | 0 | 80.0623% |
| cross-attention layer 4 | 8 | 2 | 0 | 80.0623% |
| cross-attention layer 6 | 8 | 2 | 0 | 80.0623% |
| cross-attention layer 8 | 8 | 2 | 0 | 80.0623% |
| cross-attention layer 10 | 8 | 2 | 0 | 80.0623% |
| MLM head | 0 | 0 | 7 | 不适用 |
| BLIP-2 辅助/视觉分支 | 0 | 0 | 9 | 不适用 |
| Byte LayerNorm + projector | 0 | 0 | 0 | 0% |

每层 cross-attention 加载 query weight/bias、key bias、value bias、output dense
weight/bias 和 output LayerNorm weight/bias，共 8 个 tensor。以下两个 weight 不能加载：

```text
crossattention.self.key.weight    checkpoint [768,1408]，目标 [768,192]
crossattention.self.value.weight  checkpoint [768,1408]，目标 [768,192]
```

六层共 12 个 mismatch。这些目标 tensor 保留 LAVIS Q-Former 的标准随机初始化
`Normal(mean=0, std=0.02)`；对应 bias shape 兼容并已加载。本阶段没有新增
`192 -> 1408` adapter。

其余未加载项：

- 7 个 MLM head tensor：当前 bridge 为 hidden-state bridge，已明确移除 MLM head；
- 9 个 BLIP-2 tensor：`temp`、`ln_vision.*`、`vision_proj.*`、`text_proj.*`、
  `itm_head.*`，均没有当前目标模块；
- `byte_layernorm.*`：项目新增，保持 PyTorch LayerNorm 默认 weight=1、bias=0；
- `llm_projector.*`：项目新增，保持 PyTorch Linear 默认 Kaiming-uniform 初始化。

Query Tokens `[1,32,768]` 完整加载。word embedding 为 `[30523,768]`，与当前词表
完全一致，因此不需要也没有执行 `[DEC]` 行截断或补齐；position embedding
`[512,768]` 也完整加载。没有发现已知项目之外的不兼容。

## 回归验证

两张真实 JPEG 的结果：

```text
JPEG byte batch       [2,45260]
ByteFormer features   [2,177,192]
有效 Byte tokens       [166,177]
Query features        [2,32,768]
Projected features    [2,32,4096]
NaN/Inf               无
运行精度               FP32
峰值 GPU 显存          1,541,794,816 bytes（约 1.436 GiB）
```

数值检查：

```text
加载前后同一输入      max=3.41805172，mean=0.51547742
改变 Byte padding 值  max=0，mean=0
增加 Byte padding     max=0，mean=0
增加文本 padding      max=4.61935997e-6，mean=4.71609923e-7
重复初始化            max=0，mean=0
```

加载前后差异证明预训练权重确实改变了输出。重复加载同一 checkpoint 后结果完全一致；
有意保留随机初始化的 tensor 不会被迁移器修改。Query Tokens、Q-Former
self-attention、cross-attention、query FFN 和 projector 均有非零梯度；ByteFormer 完全
冻结且无梯度。最后一层 text-only FFN 的六个零梯度参数是 query-only loss 的预期结果。

完整机器可读结果位于服务器的：

```text
/data/students/Zhang_jinsong/spacebyte/result/byteformer/blip2_qformer_init/inspection.json
/data/students/Zhang_jinsong/spacebyte/result/byteformer/blip2_qformer_init/verification.json
```

## 使用方法

检查 checkpoint：

```bash
python scripts/inspect_blip2_qformer_checkpoint.py \
  --checkpoint /data/students/Zhang_jinsong/spacebyte/models/blip2_stage1/blip2_pretrained.pth \
  --output /data/students/Zhang_jinsong/spacebyte/result/byteformer/blip2_qformer_init/inspection.json
```

真实 JPEG 回归验证：

```bash
python scripts/verify_blip2_qformer_initialization.py \
  --blip2-checkpoint /data/students/Zhang_jinsong/spacebyte/models/blip2_stage1/blip2_pretrained.pth \
  --byteformer-config corenet/projects/byteformer/imagenet_jpeg_q100/conv_kernel_size=8.yaml \
  --byteformer-checkpoint /data/students/Zhang_jinsong/spacebyte/models/byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt \
  --image-dir /data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense/images/flickr \
  --device cuda:0 \
  --output /data/students/Zhang_jinsong/spacebyte/result/byteformer/blip2_qformer_init/verification.json
```
