# Byte/Pixel VLM 权威交接

> 后续 AI 在开展 Spacebyte、SpatialSense、ByteFormer、Pixel 分支或融合研究前，必须先读本文。本文记录截至 Byte-only 稳定性实验完成时的实际状态；若旧 README 或历史报告与本文冲突，以本文和当前源码为准。

## 1. 研究目标与当前边界

最终目标是构建一个 **Pixel + Byte 双视觉分支、GAP 式动态门控、Vicuna 生成**的 VLM。当前实际完成的链路只有 Byte 视觉分支和文本输入：

```text
带红/蓝框的图像 → JPEG Q100 bytes
→ 部分解冻的官方 ByteFormer
→ Byte tokens [B,T,192] + Byte attention mask [B,T]
→ Byte LayerNorm + BLIP-2 初始化的 instruction-aware Q-Former
→ 32 Query tokens [B,32,768]
→ Linear(768,4096)
→ Vicuna-7B-v1.1（冻结）+ 文本 prompt
→ yes/no 自回归生成或 answer-only causal-LM loss
```

当前没有 Pixel Encoder、Pixel Q-Former、GAP 门控、Vicuna LoRA，也没有把损坏增强加入训练。

## 2. SpatialSense 输入与预处理

- 任务是“候选空间关系是否成立”的平衡二分类，每条标注回答 `yes` 或 `no`，不是九分类。
- bbox 原始顺序为 `[y_min,y_max,x_min,x_max]`。
- 保持长宽比，以 bicubic 缩放并 letterbox 到 `224×224`，空白为灰色 `(128,128,128)`。
- 先画 object 蓝框 `(0,96,255)`，再画 subject 红框 `(255,32,32)`，线宽 3。
- 通过 PIL 编码为未优化的 JPEG Q100，再读取整数码流；预处理版本为 `spatialsense-red-subject-blue-object-letterbox224-jpegq100-v1`。
- 固定问题模板为：`Is the object in the red box {relation} the object in the blue box? Answer yes or no.`
- 官方划分为 train 11,238、valid 2,638、test 3,622；test 不参与训练、早停、调参或 checkpoint 选择。

## 3. ByteFormer token adapter

- 官方模型：ImageNet JPEG-Q100 ByteFormer-Tiny，配置 `k=8, w=128`，官方 `Linear(192,1000)` 分类头仍保留。
- token 在 `post_transformer_norm` **之后**、masked mean pooling **之前**、1000 类 classifier **之前**提取。
- 输入 `byte_tensor=[B,L]`，有效字节为 0–255，右侧 padding 为 `-1`。
- 输出 `features=[B,T,192]`；`attention_mask=[B,T]`，其中 1=有效 token、0=padding。
- 实际主干包含 Conv1D、12 个 Transformer blocks，并在 blocks 0、1、3、5、7、9 后进行六次 TokenMerging。
- CoreNet 原写法 `mask[x == -1].fill_(-inf)`只修改高级索引临时张量，不能修改原 mask。项目 adapter 使用可追踪的正确 mask 构造。
- 为避免 shifted-window attention 使 padding 污染有效特征，每条真实 JPEG 按自身长度独立通过官方主干，最后才在 token 层补齐并生成 mask；不能把字节级 mask 简单裁剪或插值到最终 T。
- 官方分类路径重建误差、单样本/混合 batch 有效 token 误差以及增加右侧 padding 后的误差均为 0。

## 4. Byte Q-Former 与 BLIP-2 初始化

当前 Q-Former 配置：

```text
encoder_width=192
hidden_size=768
num_hidden_layers=12
num_attention_heads=12
intermediate_size=3072
cross_attention_freq=2（layers 0,2,4,6,8,10）
query_length=32
vocab_size=30523（含 [DEC]）
projector=Linear(768,4096)
```

官方 BLIP-2 Stage-1 checkpoint 通过显式映射 `query_tokens → query_tokens`、`Qformer.* → qformer.*` 初始化。只复制语义对应且 shape 完全一致的 tensor，迁移 318 个 tensor、172,708,608 个参数 elements；Query Tokens、词/位置 embedding、自注意力、FFN 和兼容的 cross-attention 参数均成功加载。

已知且经 allowlist 审计的不兼容项：

- 六个 cross-attention 层的 key/value weight 共 12 个 tensor：来源为 `[768,1408]`，Byte 目标为 `[768,192]`，保持 LAVIS 标准随机初始化，未 reshape、截断或补零；
- `Byte LayerNorm(192)` 与 `Linear(768,4096)` 是项目新增模块；
- 当前 bridge 移除了 MLM head；BLIP-2 的视觉 LayerNorm、投影、ITM head 和 temperature 没有目标模块。

这些最初不兼容/新增参数在后续 SpatialSense 训练中已经参与优化，不能再视为未训练随机层。

## 5. Vicuna、输入协议与监督

- 模型：`lmsys/vicuna-7b-v1.1`，revision `694d39416b1744d939760f5257c4bb08b2fae400`。
- hidden size 4096、32 layers、32 heads；全部参数冻结，但 loss 路径保留通过 Vicuna 运算到视觉 prefix 的计算图。
- 按固定 X-InstructBLIP 协议增加 `[PAD]`（ID 32000），词表由 32000 resize 到 32001；BOS/EOS/UNK 都是 `</s>`（ID 2）。
- 输入顺序为：`[start + USER:] [image:] [32 projected Byte Queries] [question + ASSISTANT:] [answer + EOS]`。
- Q-Former同时接收自然语言问题作为 instruction tokens。
- 当前 answer-only 目标使用完整词表 causal-LM loss，但每个样本**恰好只监督一个 `yes` 或 `no` token**；Query、图像 cue、prompt、EOS 和 padding 的 label 全部为 `-100`，预测位置严格位于答案 token 前一位。

## 6. 当前可训练/冻结范围

ByteFormer 仅解冻实际按模块身份解析出的：

- `transformer.10`：444,864 参数；
- `transformer.11`：444,864 参数；
- `post_transformer_norm`：384 参数；
- ByteFormer 解冻合计：890,112 参数。

继续训练：完整 Q-Former、32 个 Query Tokens、Byte LayerNorm、六层 Byte-width cross-attention K/V 和 projector。冻结：Vicuna 全部参数，以及 ByteFormer embedding、Conv1D、TokenMerging、blocks 0–9 和 classifier。

稳定性实验参数组保持为：ByteFormer 解冻部分 `1e-6`、BLIP-2 初始化参数 `5e-6`、原不兼容/新增参数 `5e-5`；AdamW、weight decay 0.01、gradient clipping 1.0、FP16 GradScaler、micro batch 1、gradient accumulation 8、seed 20261013。

## 7. 关键实验结论

1. tiny-overfit 在36条固定训练样本上达到100%，只证明链路、标签和梯度连通。
2. 冻结 ByteFormer 的 full-cache pilot 最终 Normal AUROC 0.5159，只比 Zero 高0.0153、比 Shuffled 高0.0041；决策退化为单一类别，未证明利用视觉。
3. 改成 answer-only 后，冻结 ByteFormer 仍退化为全部预测 `no`；valid Normal AUROC 0.5040。
4. 解冻 ByteFormer 最后两个 block 和 post norm 后，epoch 3 的 Normal AUROC 为0.5270，相对更强控制高2.74个百分点，但当时只出现一次。
5. 稳定性续训 epoch 4 达到 Normal AUROC 0.5420，且相对 Zero/Shuffled 分别高4.31/3.84个百分点；epoch 3、4 连续满足预注册稳定条件，因此早停。五次全验证集 shuffle 与配对 bootstrap 最终支持视觉增益，状态为 `confirmed`。

结论：**冻结 ByteFormer 是先前模型难以利用 Byte 视觉信息的重要原因；部分解冻已经产生小幅但统计上可确认的独立 Byte 视觉信号。**这不等于 Pixel 分支或未来融合已经有效。

## 8. 当前最终指标

| 条件 | Accuracy | Macro-F1 | AUROC |
|---|---:|---:|---:|
| Normal | 53.64% | 50.65% | 54.20% |
| Zero-Byte | 50.00% | 33.33% | 49.89% |
| 5次 Shuffled-Byte 平均 | 50.14% | 46.62% | 50.43% |

五次 Shuffled AUROC 样本标准差为0.001324。2000次配对 bootstrap 的95%区间：

- Normal AUROC：`[0.520080, 0.564448]`；
- Normal−Zero AUROC：`[0.013579, 0.072263]`；
- Normal−五次 Shuffled 平均 AUROC：`[0.015331, 0.061869]`。

最终状态：**`confirmed`**。

## 9. 最佳 checkpoint

```text
路径：/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_byteformer_stability_longrun/stability_best.pt
大小：2,142,755,604 bytes（约 1.996 GiB）
SHA-256：31672841e55f92f0ac88c346d9af79dae1801818898fe9b70b4107ad349b6b95
epoch：4
```

该文件含 bridge、部分 ByteFormer、optimizer、scheduler、AMP scaler 和 RNG 状态；不要提交到 Git，也不要覆盖。起始 `partial_best.pt` 只有模型状态，因此稳定性续训明确属于 **`continued fine-tuning with optimizer reinitialized`**，不是严格断点续训。

## 10. Git 状态与历史分支

本文整理前的已验证实现基线：

```text
branch: feat/spatialsense-byteformer-stability-longrun
commit: 60e9e3d0d86c3ff92cded50c055af36bab552a0a
CoreNet: b205775948b27f71c55630984c42b98fff4297d9
LAVIS:   05cb670f47296522e24e7727e366b4a997d28582
```

关键历史分支按开发顺序：

| 分支 | tip commit | 内容 |
|---|---|---|
| `archive/byteformer-spatialsense-9cls` | `ee33b0067c5367b347a41d99350631ece1e44c3b` | 旧九分类归档 |
| `feat/byteformer-official-baseline` | `0144f02635f16669cf5109e4ccc2a03b782bb256` | 官方 ByteFormer 基线 |
| `feat/byteformer-token-adapter` | `bc93a594c4f2b5e70f5517f918f3a327f8445435` | token/mask adapter |
| `feat/byte-qformer-bridge` | `3fa681c319556cdeb7d0825762f39b4b3ef9869b` | Byte Q-Former bridge |
| `feat/byte-qformer-blip2-init` | `4f907f2e6da9c2a723db6233a1d9551cbfdbc8d9` | BLIP-2 Stage-1 初始化 |
| `feat/byte-vicuna-integration` | `5ff5fb28a4c36609017f2c1e53a4f1c77e3c6401` | 冻结 Vicuna 接线 |
| `feat/spatialsense-tiny-overfit` | `751892e1d51bd88f4bb716f08702a7f0d1d128de` | 小样本连通验证 |
| `feat/spatialsense-full-pilot` | `6c90b3cf8ba65b693c4dbb507e4b06a248cff23a` | 冻结 ByteFormer pilot |
| `feat/spatialsense-answer-loss-pilot` | `06aecec137fdc56b1f06151b4d7fc09fcc787dc3` | answer-only 对照 |
| `feat/spatialsense-byteformer-partial-unfreeze` | `1aa2fcbc52ed71f3d9ed80b282cb40cbf47d3480` | 部分解冻3 epochs |
| `feat/spatialsense-byteformer-stability-longrun` | `60e9e3d0d86c3ff92cded50c055af36bab552a0a` | 自动稳定性续训与统计 |

## 11. 复现稳定性实验

以下显式命令由 `scripts/train_spatialsense_stability_longrun.py` 的实际参数默认值和外部 `training_config.json` 核对得到；等价的历史启动方式是仅显式指定 `--device cuda:2`。运行会重新训练，执行前必须确认输出目录策略，不能覆盖现有最佳结果。

```bash
cd /home/Zhang_jinsong/spacebyte

/data/students/Zhang_jinsong/spacebyte/envs/spacebyte-qformer/bin/python \
  scripts/train_spatialsense_stability_longrun.py \
  --dataset-root /data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense \
  --train-manifest /data/students/Zhang_jinsong/spacebyte/cache/spatialsense_full/manifests/train.jsonl \
  --valid-manifest /data/students/Zhang_jinsong/spacebyte/cache/spatialsense_full/manifests/valid.jsonl \
  --vicuna-path /data/students/Zhang_jinsong/spacebyte/models/vicuna-7b-v1.1 \
  --start-checkpoint /data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_byteformer_partial_unfreeze/partial_best.pt \
  --output-dir /data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_byteformer_stability_longrun \
  --report-path /home/Zhang_jinsong/spacebyte/SPATIALSENSE_BYTEFORMER_STABILITY_LONGRUN.md \
  --device cuda:2 --eval-batch-size 4 --num-workers 2 \
  --seed 20261013 --gradient-accumulation 8 \
  --byteformer-lr 1e-6 --pretrained-lr 5e-6 --new-lr 5e-5 \
  --weight-decay 0.01 --gradient-clipping 1.0 \
  --warmup-steps 100 --grad-scaler-init 1024 \
  --smoke-optimizer-steps 20 --max-epoch 10 \
  --bootstrap-iterations 2000 \
  --qformer-tokenizer /data/students/Zhang_jinsong/spacebyte/models/bert-base-uncased-tokenizer \
  --blip2-checkpoint /data/students/Zhang_jinsong/spacebyte/models/blip2_stage1/blip2_pretrained.pth \
  --byteformer-config corenet/projects/byteformer/imagenet_jpeg_q100/conv_kernel_size=8.yaml \
  --byteformer-checkpoint /data/students/Zhang_jinsong/spacebyte/models/byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt \
  > /data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_byteformer_stability_longrun/train.log 2>&1
```

脚本内部会运行回归测试、20-step smoke、epoch 4–10 自动训练/验证、早停、五次全验证集 shuffle、2000次配对 bootstrap，并生成 JSON/Markdown。历史运行在 epoch 4 早停。不要把日志、checkpoint、cache 或原始数据加入 Git。

## 12. 已确认的约束与陷阱

- ByteFormer 一旦参与训练，旧的 frozen-ByteFormer feature cache 就失效；必须从带框 JPEG Q100 字节在线重新计算。
- micro batch 为1时，不能在 batch 内 shuffle；Shuffled 条件必须在**整个验证集**建立固定置换。
- 全验证集置换必须排除 sample 自身以及相同视觉 identity，目标问题/标签不变，只替换视觉来源。
- test 集不得用于训练、早停、选择 checkpoint、调参或当前阶段比较。
- answer-only 每个样本只监督一个 yes/no token；EOS 不参与 loss。
- 稳定性续训的起始 checkpoint 没有 optimizer/scheduler/scaler 状态，因此优化器被重新初始化。
- Zero-Byte 是将 projected Query prefix 缩放为0；Shuffled-Byte 是在验证集范围替换另一条样本的视觉码流，两者含义不同。
- 不得覆盖已有 checkpoint、cache、机器可读结果或训练日志。

## 13. 未完成模块、下一步与禁止事项

尚未完成：

- Pixel Encoder；
- Pixel Q-Former；
- GAP 式动态门控；
- Vicuna LoRA；
- 损坏增强和鲁棒性评估。

**下一步唯一任务：建立 Pixel-only 分支，并用 Normal、Zero-Pixel、Shuffled-Pixel 验证 Pixel 分支的独立有效性。**在 Pixel-only 达到独立视觉有效性之前：

- 不直接加入 GAP 门控；
- 不立即训练 Vicuna LoRA；
- 不使用 test 集；
- 不继续无上限训练 Byte-only 模型。

## 14. 权威来源

本文由当前 README、实际训练入口和以下已提交报告交叉核对：`SPATIALSENSE_TINY_OVERFIT.md`、`SPATIALSENSE_FULL_PILOT.md`、`SPATIALSENSE_ANSWER_ONLY_PILOT.md`、`SPATIALSENSE_PARTIAL_UNFREEZE.md`、`SPATIALSENSE_BYTEFORMER_STABILITY_LONGRUN.md`、`BYTEFORMER_TOKEN_ADAPTER.md`、`BYTE_QFORMER_BRIDGE.md`、`BLIP2_QFORMER_INITIALIZATION.md`、`BYTE_VICUNA_INTEGRATION.md`。最终 checkpoint 的路径、大小和 SHA-256 均来自服务器实际文件。
