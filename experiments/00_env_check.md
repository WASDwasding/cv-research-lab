# 环境检查

- 日期：2026-09-22
- 机器：联想 Y7000P，Intel Core i7-14650HX，NVIDIA GeForce RTX 4060 Laptop GPU（8188 MiB，驱动 566.24）
- `python` / `torch` 版本：Python 3.12.10，torch 2.6.0+cu124
- CUDA 是否可用：是。`torch.cuda.is_available()` 为 `True`，设备名 `NVIDIA GeForce RTX 4060 Laptop GPU`
- 最小命令（在仓库根目录）：

```
python src/env_check.py --epochs 2 --train-n 4000 --val-n 1000
```

- 数据集：合成 3 档图像 `synthetic-3grade`。本机没有 torchvision，CIFAR-10 没有下载。这批图只证明训练循环能跑完，不进论文主表。
- 结果数字：训练 4000 张，验证 1000 张，batch 128，学习率 0.001，种子 42。第 1 轮验证准确率 0.6390，第 2 轮 **0.6370**。用时 1.0 秒，设备 cuda。
- 下一步：装上 numpy 和 torchvision 后，再在公开集上做第 2 天的划分。主表不要使用这次合成集的数字。
