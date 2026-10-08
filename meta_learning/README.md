# RT-DETR FOMAML 迁移学习模块

> 当前主入口为根目录train_transfer.py/test_transfer.py。此目录保留普通迁移共用实现及可选FOMAML。现有GPT支持集seed=20260924；重新抽样会改变实验，请始终显式传入该seed。base/tasks尚未生成，FOMAML元训练前还需要source manifest和任务准备。

低亮目标域的普通迁移与 FOMAML 对比实验、权重位置及最终指标见
`LOW_CONTRAST_TRANSFER_README.md`。

本目录是独立实验模块，不替换、不修改项目根目录中的普通 RT-DETR 训练和迁移学习流程。它用于比较：

1. 普通迁移：基础 RT-DETR 最优权重直接在少量新场景样本上微调。
2. FOMAML 迁移：基础权重先在多个源域任务上做一阶 MAML，再用完全相同的新场景样本微调。

最终任务不输出类别，所有模型仍然只有一个 `target` 类；元任务表示真实/合成、背景、传感器或场景域，不是目标类别。

## 1. 数据约束

先使用根目录的 `01_prepare_voc_dataset.py` 生成：

```text
datasets/prepared/base       源域数据，只参与基础训练和元训练
datasets/prepared/rtdetr_gpt_low_contrast_combined_v1   新场景数据，train 用于 few-shot，val 选权重/阈值，test 最后测试
```

源域应至少包含两个任务。推荐把真实 VEDAI 和传统合成 VEDAI 分成两个 VOC 源目录，再一起准备 `base`；生成的 `manifest.csv` 会保留来源路径。若它们在同一目录，文件名含 `_aug`、`synth`、`synthetic` 或 `composite` 的样本会自动归为 `synthetic`，其余归为 `real`。

更可靠的方法是提供 CSV：

```csv
image,domain
000001.png,vedai_real
000002.png,vedai_synthetic
```

然后运行：

```powershell
python meta_learning/01_prepare_meta_tasks.py --domain-map D:/path/meta_domains.csv
```

元任务清单只从 `base/train` 生成。few-shot 支持集只从 `rtdetr_gpt_low_contrast_combined_v1/train` 生成，`rtdetr_gpt_low_contrast_combined_v1/test` 不参与训练、模型选择或阈值选择，还需要检查同源或近重复样本是否跨划分。

## 2. 文件作用

| 文件 | 作用 |
|---|---|
| `01_prepare_meta_tasks.py` | 将 `base/train` 按真实、合成或场景来源整理为多个元任务 |
| `02_train_fomaml.py` | 从基础 RT-DETR 权重开始，在各源任务上执行 support 内更新和 query 外更新，保存 FOMAML 初始化 |
| `03_prepare_fewshot_support.py` | 从 `rtdetr_gpt_low_contrast_combined_v1/train` 固定生成嵌套的 5/10/20-shot 支持集 |
| `04_adapt_fewshot.py` | 选择 `standard` 或 `fomaml` 初始化，在同一个 K-shot 集上微调，并用 val 选择最优权重 |
| `05_compare_fewshot.py` | 在 val 选择满足 Pd 的最低 FAR 阈值，再将阈值固定用于 test，输出公平对比 |
| `run_all.py` | 依次执行上述全部步骤，适合数据确认无误后运行 |
| `common.py` | 权重、参数冻结、损失、随机种子与输出等公共函数，不直接运行 |

## 3. 推荐运行顺序

在 PyCharm 中把 Working directory 设置为 `D:\InfraredSmallTargetDetection`，解释器选择已验证的 CUDA 环境。

```powershell
python meta_learning/01_prepare_meta_tasks.py
python meta_learning/02_train_fomaml.py --gpu 0
python meta_learning/03_prepare_fewshot_support.py

python meta_learning/04_adapt_fewshot.py --method standard --shots 5 --gpu 0
python meta_learning/04_adapt_fewshot.py --method fomaml --shots 5 --gpu 0
python meta_learning/04_adapt_fewshot.py --method standard --shots 10 --gpu 0
python meta_learning/04_adapt_fewshot.py --method fomaml --shots 10 --gpu 0
python meta_learning/04_adapt_fewshot.py --method standard --shots 20 --gpu 0
python meta_learning/04_adapt_fewshot.py --method fomaml --shots 20 --gpu 0

python meta_learning/05_compare_fewshot.py --gpu 0
```

服务器使用物理 GPU 1 时把 `--gpu 0` 改为 `--gpu 1`。也可以一次运行：

```powershell
python meta_learning/run_all.py --gpu 1
```

## 4. 初始化与训练细节

FOMAML 和普通迁移默认使用同一个基础权重，查找顺序为：

```text
outputs/rtdetr/base/best.pth
weights/rtdetr/vedai_real_synth_best.pth
```

可用 `--initial-weight D:/path/best.pth` 明确指定。FOMAML 默认只更新 RT-DETR 的 `encoder` 和 `decoder`，冻结 backbone，以降低显存和少样本过拟合风险。每个 episode 对每个源域抽取互不重叠的 support/query 图片：support 做一次内循环 SGD，query 梯度用于更新公共初始化参数；忽略二阶梯度，因此是 FOMAML。

默认超参数位于根目录 `project_config.yaml`：

```text
meta epochs: 20
episodes/epoch: 10
support/query per task: 2/2
inner lr: 1e-4
meta lr: 1e-5
adaptation: 20 epochs, lr 2e-5
```

GTX 1650 Ti 建议保持 batch 1；RTX 4090 可增加 episode 数量，但为保证对比公平，普通迁移和 FOMAML 的适配轮数、学习率、支持样本必须相同。

## 5. 权重和结果

```text
outputs/meta_learning/fomaml/base/best.pth
outputs/meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/standard_005shot/best.pth
outputs/meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/fomaml_005shot/best.pth
outputs/meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/standard_010shot/best.pth
outputs/meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/fomaml_010shot/best.pth
outputs/meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/standard_020shot/best.pth
outputs/meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/fomaml_020shot/best.pth
outputs/meta_learning/evaluation/rtdetr_gpt_low_contrast_combined_v1/comparison.csv
```

每个评测目录还会保存 `validation/summary.json`、`test/summary.json`、阈值扫描、逐图 TP/FP/FN 和预测框 CSV。指标采用类别无关一对一匹配，正确检测条件为 `IoU >= 0.40`：

```text
Pd  = TP / (TP + FN)
FAR = FP / (TP + FP)
```

## 6. 如何判断 FOMAML 有效

不能只看一个 shot 的一次结果。固定数据划分后比较 5/10/20-shot：如果 FOMAML 在多个 shot 下稳定提高 test Pd 或 F1，同时 FAR 不明显变差，才说明它更适合比赛的新场景快速适配。建议后续用 3 至 5 个随机种子重复实验，报告均值和标准差。

FOMAML 适合本项目的迁移学习部分；类/场景增量学习仍建议使用旧样本回放与蒸馏，避免灾难性遗忘。正式比赛若只下发极少量新场景样本，使用 FOMAML 初始化适配；若不允许赛时训练，则直接提交基础模型或离线适配后的固定权重。
