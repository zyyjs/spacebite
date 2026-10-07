# SpatialSense × ByteFormer 实施状态

更新时间：2026-10-07

## 已完成

- 将 SpatialSense annotation 按有序 subject/object 对聚合为 9 维多标签目标。
- 使用 mask 忽略未标注关系，避免将未知标签误当成负例。
- 在 224×224 JPEG Q100 图片中绘制 subject 红框和 object 蓝框。
- 以真实 JPEG 文件字节作为 ByteFormer 输入，并使用 `-1` 做批内补齐。
- 严格载入官方 ImageNet JPEG Q100、kernel size 8、width 128 权重，再将分类头替换为 9 维输出。
- 实现 weighted sampler、masked BCE、验证指标、checkpoint 和独立评估脚本。
- 完整数据转换和完整缓存验证。
- 单元测试、GPU 冒烟测试、16 样本分类头过拟合测试和 batch size 16 全模型反向传播测试。

## 当前数据规模

| split | 样本数 |
| --- | ---: |
| train | 10,989 |
| valid | 2,564 |
| test | 3,539 |
| 合计 | 17,092 |

完整数据验证报告：

`/data/students/Zhang_jinsong/spacebyte/result/byteformer/data_validation/report.json`

## 预训练前检查结果

- Pytest：5 项全部通过。
- JPEG 文件长度：17,682–74,618 bytes，平均 39,547 bytes。
- kernel size 8 后最长序列：18,653 tokens，小于配置上限 50,000。
- batch size 16、全模型反向传播：loss 有限，梯度有限。
- GPU 峰值：allocated 约 6.00 GiB，reserved 约 7.17 GiB。
- 16 样本分类头训练 loss 最低约 0.023，证明标签、损失和优化链路可学习。

检查产物：

- `result/byteformer/smoke_initial_20261007/`
- `result/byteformer/overfit_head_16samples_20261007/`
- `result/byteformer/batch_probe/report.json`

## 正式基线

正式运行采用 `configs/spatialsense_byteformer_q100.yaml`，输出目录为：

`/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_q100_baseline/`

训练结束后使用验证集 macro balanced accuracy 最优的 `best.pt` 在测试集上评估。
