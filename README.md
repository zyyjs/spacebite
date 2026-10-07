# Spacebyte

用于模型下载、数据准备、微调、评估和推理的项目仓库。

当前仅保留基础仓库文件；模型和训练方案确定后，再按实际需要创建目录、安装依赖并添加代码。

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

数据格式和微调方案确定后，再补充环境安装及训练命令。

## Git 安全约定

数据集、模型权重、训练产物、环境变量和访问令牌不得提交到仓库。需要共享的小型示例数据应先脱敏，并存放在单独的样例目录中。
