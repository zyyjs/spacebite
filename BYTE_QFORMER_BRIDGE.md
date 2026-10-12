# Random-initialized Byte Q-Former bridge

本分支实现一个独立、instruction-aware 的桥接模块：

```text
JPEG bytes
  -> 冻结的官方 ByteFormerTokenAdapter
  -> byte features [B,T,192] + byte mask [B,T]
  -> trainable LayerNorm(192)
  -> 32 query tokens + instruction tokens
  -> 12-layer LAVIS Q-Former
  -> query features [B,32,768]
  -> Linear(768,4096)
  -> projected features [B,32,4096]
```

`initialization_mode` 明确为 `random`。这里只复用了固定 LAVIS 子模块提交中的
`Qformer.py` 实现，没有下载或加载 BLIP-2、X-InstructBLIP、Vicuna、EVA-CLIP 的任何
模型权重，也没有实例化完整 X-InstructBLIP。官方 ByteFormer checkpoint 保持不变。

## 独立环境

现有 ByteFormer 环境没有被修改。桥接验证使用：

```text
/data/students/Zhang_jinsong/spacebyte/envs/spacebyte-qformer
```

关键依赖的完整版本为：

```text
Python             3.10.22
PyTorch            2.3.0+cu121
torchvision        0.18.0+cu121
torchaudio         2.3.0+cu121
torchtext          0.18.0
Transformers       4.33.2
tokenizers         0.13.3
huggingface-hub    0.36.2
safetensors        0.8.0
pytest             8.1.1
CoreNet            b205775948b27f71c55630984c42b98fff4297d9
LAVIS              05cb670f47296522e24e7727e366b4a997d28582
CUDA wheels        cu121
```

可复现环境定义见 `environment-qformer.yml`。为避免 `lavis/__init__.py` 启动完整模型
注册流程及不需要的模态依赖，bridge 使用 `importlib` 直接加载固定路径下的
`lavis/lavis/models/blip2_models/Qformer.py`。这仍然使用原始 LAVIS Q-Former 类，且不
修改 LAVIS 子模块。

## 实际 Q-Former 配置

```text
encoder_width          192
hidden_size            768
num_hidden_layers      12
num_attention_heads    12
intermediate_size      3072
add_cross_attention    true
cross_attention_freq   2
cross-attention layers 0, 2, 4, 6, 8, 10
query_length           32
vocab_size             30523
LLM projection         768 -> 4096
precision              FP32
```

Q-Former 的 MLM head 不参与桥接且已在随机初始化后置为 `None`，因此不会保留一组永远
收不到梯度的无关参数。instruction 的 word/position embeddings、自注意力、query
cross-attention 和 FFN 均保留。

## 验证

```bash
conda activate /data/students/Zhang_jinsong/spacebyte/envs/spacebyte-qformer

python scripts/verify_byte_qformer_bridge.py \
  --byteformer-config corenet/projects/byteformer/imagenet_jpeg_q100/conv_kernel_size=8.yaml \
  --byteformer-checkpoint /data/students/Zhang_jinsong/spacebyte/models/byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt \
  --image-dir /data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense/images/flickr \
  --device cuda:0
```

脚本输出机器可读 JSON。2026-10-12 的两张真实 JPEG 验证得到：

```text
JPEG byte lengths          42448, 45260
ByteFormer features        [2,177,192]
valid ByteFormer tokens    166, 177
query features             [2,32,768]
projected features         [2,32,4096]
combined query/text mask   [2,38]
NaN/Inf                    none
byte padding value error   max=0, mean=0
added byte padding error   max=0, mean=0
text padding error         max=1.4901161e-6, mean=2.4992565e-7
peak allocated GPU memory  1,537,236,480 bytes (about 1.432 GiB)
```

文本追加 padding 后 CUDA 可能为不同矩阵形状选择不同 kernel；上述 `1e-6` 量级是
FP32 舍入差异，masked 文本没有参与 attention。

参数量：

| 模块 | 总参数 | 可训练参数 |
|---|---:|---:|
| 官方 ByteFormer（含原 1000 类头） | 15,920,680 | 0 |
| Q-Former（不含 MLM head） | 174,453,504 | 174,453,504 |
| Query tokens | 24,576 | 24,576 |
| Byte LayerNorm | 384 | 384 |
| LLM projector | 3,149,824 | 3,149,824 |
| 完整 bridge | 177,628,288 | 177,628,288 |

Query tokens、LayerNorm、self-attention、cross-attention、query FFN 和 projector 均
检测到非零梯度。loss 只依赖最终 query 输出，因此最后一层的 text-only FFN 处在该
loss 的下游，不应影响 query；若诊断中显示这四个参数为零梯度，这是预期行为：

```text
qformer.bert.encoder.layer.11.intermediate.dense.weight
qformer.bert.encoder.layer.11.intermediate.dense.bias
qformer.bert.encoder.layer.11.output.dense.weight
qformer.bert.encoder.layer.11.output.dense.bias
qformer.bert.encoder.layer.11.output.LayerNorm.weight
qformer.bert.encoder.layer.11.output.LayerNorm.bias
```

## 未来加载 BLIP-2 Stage-1 时的预计不兼容项

本轮没有实际加载该 checkpoint。根据结构，未来需要显式处理：

- BLIP-2 视觉 encoder width 与 ByteFormer 的 192 不同，因此 0、2、4、6、8、10 层
  cross-attention 的 key/value 投影权重输入维度不兼容；
- 本项目的 `LayerNorm(192)` 和 `Linear(768,4096)` 是新增模块，BLIP-2 Stage-1 不会
  提供可直接对应的参数；
- 32 个 query tokens 与隐藏维 768 的 Q-Former 核心参数理论上可兼容，但仍需核对
  checkpoint 键名前缀和实际 query 数；
- word embedding 只有在来源 checkpoint 同样使用加入 `[DEC]` 后的 30523 词表时才
  能直接兼容；普通 BERT 30522 词表会差一行；
- MLM head 本 bridge 已移除，来源 checkpoint 中相应键会成为预期的 unexpected keys；
- 不允许用 `strict=False` 静默忽略上述问题；未来加载器必须逐项列出并处理。
