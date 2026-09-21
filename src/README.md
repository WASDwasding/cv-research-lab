# 代码

训练与评测脚本放这里。在环境检查通过、老师圈题之前，不要堆模型文件。

建议以后：

```
src/
  data/      划分与 Dataset
  models/    骨干与分类头
  train.py
  eval.py
```

依赖用 `requirements.txt` 锁版本；随机种子写进命令行。
