# 研究方向：从图像评估手机和平板的屏幕损伤程度

> 英文：`smartphone and tablet screen damage severity estimation`。中文：屏幕划痕、裂纹、碎裂的损伤分级。  
> 对象在 2026-09-23 收死：只评屏幕玻璃，不评后盖、边框、摄像头。

---

## 1. 一句话

人看一张手机或平板的屏幕照片能判断「伤得重不重」。本课题让模型做同一件事：在**已经知道是手机或平板屏幕**的前提下，给出损伤档次，必要时标出伤在哪里。

应用对应：二手屏幕成色、屏幕玻璃出厂质检。

---

## 2. 学科位置

| 别人问 | 怎么答 |
|--------|------|
| 大方向 | 计算机视觉 |
| 应用方向 | 视觉质检 / 产品状态评估 |
| 默认任务 | 细粒度有序分类（3 档损伤） |
| 可选前置 | 缺陷检测或分割（框/涂出伤痕） |

不是「这是哪一款手机」，而是「这块屏幕现在伤成什么样」。

```
商品图
 ├── 这是什么？          → 分类（本课题不当主问题）
 ├── 是不是同一件？      → 检索（课题 A，非默认）
 ├── 东西在画面哪？      → 检测（课题 C，仅当前置）
 └── 伤得有多重、伤在哪？ → 本课题（默认 B）
```

---

## 3. 四种做法（只锁定一种主输出）

| 类型 | 输出 | 适合现在？ |
|------|------|------------|
| **1 等级（默认）** | 无 / 轻 / 中重 | ✅ 最好标、最好发 EI |
| 2 位置+程度 | 框/掩膜 + 等级 | 老师偏检测时再加 |
| 3 连续分数 | 0–10 | 标注主观，先不做 |
| 4 视频 | 多视角短视频 | 标注和训练都更重，先不做 |

档次有顺序：把「轻」预测成「中」应比预测成「严重」惩罚更小（有序分类）。

---

## 4. 题目边界

> 针对手机和平板的屏幕玻璃，用图像估计损伤等级。  
> 公开集用 SSGD（手机屏幕玻璃，带框）。先做整图 3 档分类，再比较「先框出伤痕再分级」是否更准。  
> 再用真实拍摄的手机或平板屏幕验证反光、脏背景时掉多少点，把失败案例写进论文。

不要写：「用 AI 分析所有产品在图片和视频中的各种损伤」。

---

## 5. 指标（论文至少要有）

- 等级：Accuracy、每档 F1、混淆矩阵；最好看有序误差  
- 若做检测：mAP@0.5  
- 必须有：**基线、≥3 对照、失败案例**（反光当划痕、小裂纹看漏、背景当伤）

---

## 6. 搜论文关键词

中文：手机屏幕缺陷检测、屏幕裂纹分级、屏幕划痕、外观质检。  
英文：`smartphone screen defect detection`，`screen crack`，`damage severity estimation`，`ordinal classification`。

公开数据只下屏幕集，不下钢材 NEU、不下 MVTec：

| 顺序 | 数据集 | 用途 |
|------|--------|------|
| 今天就下 | **SSGD**（Han et al., ICASSP 2023） | 主实验。2504 张手机屏幕玻璃，7 类缺陷，有框。学术使用。仓库 https://github.com/VincentHancoder/SSGD/ ，压缩包在该页的 Baidu 与清华云链接。放到 4060 的 `data/raw/ssgd/` |
| 仅当前面两个链接都下不下来 | **MSD**（Zhang et al., ICASSP 2022） | 备选。1200 张屏幕表面缺陷掩膜（oil / scratch / stain）加 20 张完好图。https://github.com/jianzhang96/MSD |

SSGD 没有完好屏幕。映射表里「无」这一档写 0 张，不把污迹改叫无。完好屏幕留到 10 月 7 日的真实拍摄。

先读的两篇论文：

1. 缺陷检测：Han, Yang, Li, Hu, Li. *SSGD: A smartphone screen glass dataset for defect detection.* ICASSP 2023. arXiv:2303.06673。  
2. 有序分类：Cao, Mirjalili, Raschka. *Rank consistent ordinal regression for neural networks with application to age estimation.* Pattern Recognition Letters, 2020（CORAL）。第 9 天的有序损失按这篇做。

第 3 天再读成色分级：Mohandas, Southern, Fitzpatrick, Hayes. *Deep learning enabled computer vision in remanufacturing and refurbishment applications: defect detection and grading for smart phones.* Journal of Remanufacturing, 2025. DOI:10.1007/s13243-024-00147-2。他们自建的严重度数据不作为本次下载。

---

## 7. 不是什么

不是通用识别、不是照片糊不糊的质量评价、不是生成修复图、不是医疗诊断、不是必须上视频和大模型。
