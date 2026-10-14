# SpatialSense ByteFormer Stability Long-run

## 1. 实验目的

验证部分解冻 ByteFormer 后，Normal 条件相对 Zero-Byte 与全验证集 Shuffled-Byte 的视觉增益是否稳定且具有统计支持。

## 2. checkpoint恢复方式

起始：`/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_byteformer_partial_unfreeze/partial_best.pt`。恢复方式：**continued fine-tuning with optimizer reinitialized**。

## 3. 训练epoch和停止原因

本轮训练 epoch 4–4；停止原因：`stability_confirmed_two_consecutive_epochs`。

## 4. 每个epoch指标

| Epoch | N Acc | N Macro-F1 | N AUROC | Z Acc | Z Macro-F1 | Z AUROC | S Acc | S Macro-F1 | S AUROC | N-Z | N-S |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.5155 | 0.5155 | 0.5270 | 0.5000 | 0.3333 | 0.4989 | 0.5061 | 0.5061 | 0.4996 | 0.0281 | 0.0274 |
| 4 | 0.5364 | 0.5065 | 0.5420 | 0.5000 | 0.3333 | 0.4989 | 0.5038 | 0.4684 | 0.5036 | 0.0431 | 0.0384 |

## 5. 最佳epoch

Epoch **4**（按 Normal AUROC、Normal Macro-F1、较早epoch依次选择）。

## 6. 五次shuffle

AUROC均值 **0.504305**，样本标准差 **0.001324**。

## 7. Bootstrap置信区间

- Normal AUROC：[0.520080, 0.564448]
- Normal−Zero AUROC：[0.013579, 0.072263]
- Normal−五次Shuffled均值 AUROC：[0.015331, 0.061869]

## 8. 最终结论

`confirmed`

## 9. 峰值显存和耗时

峰值GPU显存：16.086 GiB；总耗时：1.006小时。

## 10. checkpoint和日志

- Best：`/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_byteformer_stability_longrun/stability_best.pt`
- Last：`/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_byteformer_stability_longrun/stability_last.pt`
- 日志：`/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_byteformer_stability_longrun/train.log`

## 11. 测试结果

训练前：`54 passed, 16 warnings in 46.67s`；训练后：`54 passed, 16 warnings in 14.54s`；20-step smoke：通过。

## 12. Git

分支：`feat/spatialsense-byteformer-stability-longrun`；训练启动时commit：`1aa2fcbc52ed71f3d9ed80b282cb40cbf47d3480`；实验报告由本分支最终结果提交生成。
