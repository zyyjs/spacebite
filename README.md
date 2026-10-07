# Spacebyte

用于模型下载、数据准备、微调、评估和推理的项目仓库。

当前实现使用 ImageNet JPEG Q100 ByteFormer 预训练模型，在 SpatialSense 上进行
9 类空间关系的多标签微调。每个样本由一对有序的 subject/object 构成，并只对数据集中
实际标注的关系计算 masked binary cross-entropy。

## 开始使用

ByteFormer 的官方 CoreNet 源码位于 `corenet/` 子模块，其中项目配置见
`corenet/projects/byteformer/`。

首次克隆本仓库时一并获取源码：

```bash
git clone --recurse-submodules https://github.com/zyyjs/spacebite.git
```

对于已克隆的仓库，初始化或更新子模块：

```bash
git submodule update --init --recursive
```

当前服务器上的 ByteFormer 训练环境位于：

```text
/data/students/Zhang_jinsong/spacebyte/envs/byteformer
```

激活环境：

```bash
conda activate /data/students/Zhang_jinsong/spacebyte/envs/byteformer
```

也可以在仓库根目录通过 `environment.yml` 重建环境：

```bash
conda env create --file environment.yml
```

## 数据准备

预处理会将 subject/object 框分别绘制为红色和蓝色，将图片等比例缩放并补边到
224×224，然后以 JPEG quality=100 保存。模型读取的是生成文件的真实 JPEG 字节，
而不是解码后的像素张量。

```bash
python scripts/prepare_spatialsense.py \
  --config configs/spatialsense_byteformer_q100.yaml

python scripts/validate_spatialsense.py \
  --config configs/spatialsense_byteformer_q100.yaml \
  --output /data/students/Zhang_jinsong/spacebyte/result/byteformer/data_validation/report.json
```

## 训练与评估

默认配置在 GPU 0 上使用 batch size 16、BF16、3 个 epoch 的分类头预热和 25 个
epoch 的总训练周期：

```bash
python scripts/train.py \
  --config configs/spatialsense_byteformer_q100.yaml \
  --run-name spatialsense_q100_baseline

python scripts/evaluate.py \
  --config configs/spatialsense_byteformer_q100.yaml \
  --checkpoint /data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_q100_baseline/best.pt \
  --split test
```

训练产物包括 `metrics.jsonl`、`last.pt`、`best.pt` 和解析后的配置，统一写到
`/data/students/Zhang_jinsong/spacebyte/result/byteformer/<run-name>/`。

用于快速检查安装、数据和前向/反向传播的命令：

```bash
pytest -q
python scripts/train.py \
  --config configs/spatialsense_byteformer_q100.yaml \
  --processed-root /data/students/Zhang_jinsong/spacebyte/processed/spatialsense_byteformer_q100_smoke \
  --smoke
```

详细设计与实验阶段见 `TRAINING_PLAN.md`。

## Git 安全约定

数据集、模型权重、训练产物、环境变量和访问令牌不得提交到仓库。需要共享的小型示例数据应先脱敏，并存放在单独的样例目录中。
