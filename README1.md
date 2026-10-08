# RT-DETR 红外小目标检测

## 数据、模型权重与实验结果下载

代码与说明文档通过 GitHub 分支 `Infrared-Small-Target-Detection` 提供；大体积数据、模型权重和实验结果通过同一个网盘资源包提供。

- 网盘下载地址：[百度网盘下载](https://pan.baidu.com/s/1kORKAJxS6XA2yqmHRWq5dg)
- 提取码：`vtgf`
- 压缩包名称：`InfraredSmallTargetDetection-data.zip`
- 2026-10-08 本地核查：未压缩资源合计约 **3.30 GiB**；实际压缩包大小以网盘为准。

### 资源包包含哪些文件

从原项目根目录同时选择以下三个文件夹，完整压缩到同一个 ZIP 中。使用复制或压缩操作，保留原项目中的文件。

| 文件夹 | 内容 | 未压缩大小（约） |
|---|---|---:|
| `datasets/` | 两套 prepared 数据集，以及数据目录说明；包含图片、VOC XML、COCO JSON 和划分清单 | 2.69 GiB |
| `weights/` | 两个历史 RT-DETR 权重：VEDAI 原图/合成图基础权重及 FOMAML 初始化权重，以及用途说明 | 154 MiB |
| `outputs/` | GPT 低亮场景普通迁移和 FOMAML 的 5/10/20-shot 最优权重、训练日志、元数据、评估结果 | 475 MiB |

请保留这些目录下的全部内容及现有 train/val/test 划分，不要只打包图片或单独抽出 best.pth。datasets/README.md、datasets/CATALOG.md 和 weights/README.md 等小型说明文档也应保留在 GitHub 代码仓库中。

### 下载后的放置方法

解压到克隆得到的代码仓库根目录，即 train_basic.py、project_config.yaml 所在目录。最终相对结构应为：

```text
Image-algorithm-group/
├─ train_basic.py
├─ test_basic.py
├─ project_config.yaml
├─ datasets/
│  └─ prepared/
│     ├─ base/
│     └─ rtdetr_gpt_low_contrast_combined_v1/
├─ weights/
│  └─ rtdetr/
│     ├─ vedai_real_synth_best.pth
│     └─ vedai_real_synth_fomaml_best.pth
└─ outputs/
   └─ meta_learning/
      ├─ adaptation/
      │  └─ rtdetr_gpt_low_contrast_combined_v1/
      │     ├─ standard_005shot/
      │     ├─ standard_010shot/
      │     ├─ standard_020shot/
      │     ├─ fomaml_005shot/
      │     ├─ fomaml_010shot/
      │     └─ fomaml_020shot/
      └─ evaluation/
```

若解压软件额外生成了同名资源包文件夹，请将其内部的 datasets、weights、outputs 三个文件夹复制到代码根目录，不要让数据多嵌套一层。与已有文件合并时，先核对同名文件，避免覆盖后来产生的新实验结果。

### 现有资源与运行限制

- 当前资源包保留的是已有数据和历史实验结果，不包含新一轮随机初始化训练产生的 `outputs/rtdetr/base/best.pth`。该文件需完成基本项训练后生成，或在支持指定权重的入口中显式使用已有历史权重。
- `outputs/meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/standard_010shot/best.pth` 为已保留的普通迁移权重，也是增量代码使用的初始权重。
- 当前资源包不包含增量阶段 1/2 的数据和训练结果，也不包含真实连续五帧评测数据；不能仅凭本资源包直接完成这些实验。
- 根目录的 `meta_learning/` 是代码目录，要上传 GitHub；`outputs/meta_learning/` 是实验产物，放入网盘资源包。两者不要混淆。
- 上传代码时应通过 .gitignore 排除 `datasets/prepared/`、`outputs/` 和模型权重（如 `*.pth`）；网盘资源 ZIP 也不要加入 Git 提交。



项目路径：`D:\InfraredSmallTargetDetection`。仅保留 RT-DETR；根目录只放六个训练/测试入口。模型实现、数据处理和评估函数在子目录中，不要只复制六个入口文件运行。

## 环境

本机 Python：`D:\MySoftware\Anaconda3\envs\aa_yolo\python.exe`。`aa_yolo` 是已有环境名称，并不代表项目仍使用 AA-YOLO。依赖见项目 requirements 文件，不要另用系统 Python。VS Code/PyCharm 选择此解释器，工作目录设为项目根目录。

## 六个入口

| 文件 | 用途 | 默认数据/权重 |
|---|---|---|
| train_basic.py | 基本项：RT-DETR 随机初始化训练；不加载预训练权重 | datasets/prepared/base；输出 outputs/rtdetr/base/best.pth |
| test_basic.py | 基本项验证选阈值、固定阈值测试 | 同上 best.pth；可用 --weights 指定历史权重 |
| train_transfer.py | 普通迁移：默认 GPT 低亮训练集 10-shot；冻结骨干，适配 encoder/decoder | 初始 outputs/rtdetr/base/best.pth；输出 outputs/meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/standard_010shot/best.pth |
| test_transfer.py | 普通迁移验证及测试 | 同一 profile 和 shot 数的迁移权重 |
| train_incremental.py | 从普通迁移权重开始，依次训练增量阶段1、2；使用回放，不使用 FOMAML | 输出 outputs/incremental_after_transfer；前提是另行准备两个阶段数据 |
| test_incremental.py | 评估两阶段增量模型 | 上述阶段权重和已准备的阶段评估集 |

默认本机 GPU0、小 batch。入口将参数传给内部实现。先运行 `python train_basic.py --dry-run` 可查看命令，不会训练；其余五个文件同理。

```powershell
conda activate aa_yolo
cd D:\InfraredSmallTargetDetection
python train_basic.py
python test_basic.py
# 历史 VEDAI + 合成权重仍保留，不冒充本次从零训练的结果：
python test_basic.py --weights weights/rtdetr/vedai_real_synth_best.pth
python train_transfer.py --shots 10
python test_transfer.py --shots 10
```

基本项会从本次自身已有 checkpoint 续训。`--no-resume` 不会覆盖已有训练结果。迁移学习遇到已有同名实验会拒绝覆盖；本次整理保留了历史 standard_010shot 实验。新实验应使用新 profile/独立输出设计，不要为重跑随意删除旧权重。

## 数据

只保留 VEDAI 原图及合成图、GPT 模拟场景图。详见 `datasets/README.md`。VOC XML 为人工检查标注，训练使用由标注生成的 COCO JSON。保留原有 train/val/test 划分，不重新随机划分，不移动外部原始文件。

| profile | train | val | test |
|---|---:|---:|---:|
| base（训练含638张VEDAI原图+253张合成） | 891 | 80 | 79 |
| rtdetr_gpt_low_contrast_combined_v1（GPT低亮） | 283 | 60 | 62 |

增量实验的旧权重、输出和派生划分已按要求删除，本次不会重建。两个增量入口保留算法能力，但当前直接启动会因缺少阶段数据而失败。未来明确阶段1/2的数据来源后，再显式运行 `python -m competition.workflow incremental-prepare`，审查其划分再训练，避免与普通迁移训练集或测试集重叠。

## 代码结构

2026-10-03已继续清理子目录：移除HIT-UAV/road_v3/v4历史结果、配置和权重，导出冒烟输出、文档渲染缓存、重复GPT源数据、YOLO TXT标签、已完成迁移实验的last.pth。VEDAI/GPT图片、XML、COCO标注、全部相关best.pth和有效实验结果保留。

| 目录 | 保留原因 |
|---|---|
| competition | 六个入口共用的数据、指标、推理与回放实现 |
| engines/rtdetr | 网络及训练引擎；include/rtdetr_r50vd.yml是R18配置引用的共用模板，必须保留 |
| training、evaluation | 基本项训练、测试和五帧评估实现 |
| meta_learning | 普通迁移共用实现、GPT支持集及可选FOMAML代码 |
| incremental_after_transfer | 两阶段增量实现，待提供阶段数据 |
| datasets/prepared | 两套有效数据：base与GPT低亮组合 |
| weights、outputs/meta_learning | VEDAI基础/FOMAML初始化、GPT迁移最优权重及结果 |
| vendor/windows_py38 | Windows Python3.8的pycocotools兼容依赖 |
| tools、tests、docs、reports | 数据维护、运行检查、技术文档和清理清单 |

清理详情见 `reports/subfolder_cleanup_20261003.json`。所有移出的内容完整归档到项目外的 `D:\InfraredSmallTargetDetection_archive_20261002\subfolder_cleanup_20261003_130138`，该目录不是运行依赖。项目内完成的迁移实验保留best.pth和日志；如需旧last.pth及优化器状态，可以从归档恢复。

- `competition/workflow.py`：六个入口共用的命令分发，`python -m competition.workflow` 列出维护操作。
- `training/train_rtdetr.py`、`evaluation/test_rtdetr.py`：基本项训练和评估实现；直接追加 `--help` 查看详细参数。
- `engines/rtdetr/src/`：RT-DETR 原始网络、损失、匹配器、数据加载和优化器；必须保留。
- `competition/rtdetr_runtime.py`：生成训练配置及模型推理；`metrics.py`：无类别目标匹配和指标；`postprocess.py`：验证集置信度/NMS选择。
- `tools/prepare_voc_dataset.py`、`competition/prepare.py`：VOC转训练格式；`tools/catalog_datasets.py`：数据清点；`tools/audit_dataset.py`：数据审计；`tools/check_environment.py`：环境检查；`tools/export_rtdetr_onnx.py`：导出。
- `meta_learning/`：迁移实现及可选 FOMAML 实验。`03_prepare_fewshot_support.py` 只从目标 train 抽支持集；`04_adapt_fewshot.py` 适配；`05_compare_fewshot.py` 验证和测试；普通迁移依赖 common.py，不可删除整个目录。
- `incremental_after_transfer/`：阶段数据准备、顺序训练、两阶段测试；`competition/incremental_trainer.py`：共用回放训练器。
- `evaluation/evaluate_five_frames.py`：真实连续五帧评测。不能把独立静态图随意排序冒充序列；example文件不是正式序列标注。
- `tests/`：结构、数据存在性、指标和CPU模型冒烟检查，不训练。

## 评估与验证

单类检测，不要求输出语义类别。正确匹配使用 IoU >= 0.4；Pd=TP/(TP+FN)，FAR=FP/(TP+FP)。权重及阈值只在验证集选择，测试集不参与调参。本次仅整理代码和数据，不产生新实验指标，历史指标与元数据保留在 outputs/reports 中，不能替代真实比赛域验证。

```powershell
python -B -m unittest discover -s tests -p test_project_layout.py
python -B tests/smoke_test.py --forward
```

模型为 R18-vd、100 queries、3层 decoder，输入宽1280/高1024。比赛时延需在 Jetson AGX Orin 64GB 三模型同时运行条件下按完整接口计时；本机验证不代表满足45ms。

## 每个目录和文件的用途

以下说明对应2026-10-03清理后的实际文件。所有路径相对于 `D:\InfraredSmallTargetDetection`。图片、XML和重复的实验结果用文件模式说明；例如 `*.xml` 表示每张图片对应的标注文件，不需要逐张运行。

### 根目录及 .vscode

| 文件 | 具体作用 | 使用方式 |
|---|---|---|
| train_basic.py | 基本项训练入口，分发到 training/train_rtdetr.py；默认从零训练 | 日常训练直接运行 |
| test_basic.py | 基本项评估入口，先验证选置信度，再固定用于测试 | 训练完成后运行；历史权重用 --weights 指定 |
| train_transfer.py | 普通迁移入口，默认在 GPT train 的10-shot支持集上适配 | 已有基本项初始化和支持集后运行 |
| test_transfer.py | 普通迁移评估入口，默认只评估 standard 方法 | 已有迁移最优权重后运行 |
| train_incremental.py | 两阶段增量训练入口，顺序调用阶段1和阶段2 | 先准备阶段数据；当前阶段数据尚未生成 |
| test_incremental.py | 增量评估入口，比较迁移初始模型与两个阶段模型 | 两阶段权重生成后运行 |
| project_config.yaml | 全工程默认设置：IoU、Pd目标、GPU、轮数、batch及元学习参数 | 修改默认值；入口指定的命令行参数优先 |
| requirements.txt | 通用依赖版本，例如NumPy、Pillow、OpenCV、SciPy、TensorBoard和COCO工具 | CUDA版PyTorch另外安装；不要与服务器依赖文件混装 |
| requirements_pc_cuda118.txt | 本机参考PyTorch版本：torch 2.0.1、torchvision 0.15.2、torchaudio 2.0.2，cu118 | 与通用依赖配合使用；不是完整的独立依赖清单 |
| requirements_server_cuda128.txt | Linux/新GPU服务器参考环境：torch 2.7.1、torchvision 0.22.1及配套依赖，cu128 | 用于相应服务器环境；不是Jetson专用环境文件 |
| README.md | 项目总说明、运行顺序、数据和逐文件用途 | 优先阅读本文件 |
| .gitignore | Git忽略规则，控制哪些运行产物不进入版本管理 | 无需运行 |
| .vscode/settings.json | VS Code中Conda环境管理器和包管理器的选择设置 | 不会自动保证选中aa_yolo；解释器仍需在VS Code选择 |

六个入口是轻量调用文件，算法实现位于下列目录中。修改模型应查阅 `engines/rtdetr/src/`，修改数据、评估或训练流程应查阅对应公共模块。

### competition：公共功能

| 文件 | 具体作用 |
|---|---|
| __init__.py | 将competition目录声明为Python包，支持跨入口导入 |
| workflow.py | 将六个入口映射到内部实现；传递参数、设置工作目录/PYTHONPATH，支持--dry-run |
| paths.py | 定位项目根目录、读取project_config.yaml、拼接prepared数据路径并检查文件是否存在 |
| voc.py | 解析VOC XML中的图片信息和矩形框，为数据准备提供统一样本格式 |
| prepare.py | 将VOC源数据整理为单类数据集，复制/链接图片、保存XML和COCO JSON，按指定方式生成划分与来源记录 |
| metrics.py | 类别无关的一对一目标匹配、IoU计算、TP/FP/FN、Pd/FAR/F1、阈值选择及逐图结果保存 |
| rtdetr_runtime.py | 生成单类R18训练配置；加载checkpoint并推理train/val/test中的指定split；保存预测、指标和计时 |
| postprocess.py | 类别无关NMS处理及验证集confidence/NMS阈值搜索；供迁移和增量测试调用 |
| platform_deps.py | 配置RT-DETR源码导入路径；Windows Python3.8缺少可用pycocotools时启用vendor兼容包 |
| incremental_trainer.py | 共用增量阶段训练器：加载上一阶段权重、冻结backbone、训练encoder/decoder、验证选best并记录进度；回放样本来自准备好的混合训练集 |

除有命令行参数的工具模块外，这些文件由入口导入，不需要逐个点击运行。当前增量实现为有监督微调加样本回放，没有执行FOMAML内外循环，也没有默认启用蒸馏。

### training、evaluation和tools

| 路径 | 具体作用 | 常用运行方式 |
|---|---|---|
| training/train_rtdetr.py | 基本项训练实现；生成配置、处理随机初始化/显式初始权重/断点续训，再调用训练引擎 | 通常通过train_basic.py；直接运行可用--help查看详细参数 |
| evaluation/test_rtdetr.py | 基本项评估实现；加载指定权重，验证选confidence后测试并输出指标 | 通常通过test_basic.py |
| evaluation/evaluate_five_frames.py | 读取已有预测和真实五帧分组，评估每组最后一帧；自身不会运行时序网络或跟踪器 | 有真实时序分组时使用--groups指定CSV |
| tools/check_environment.py | 检查Python、Torch/CUDA、常用依赖、源码路径及COCO兼容依赖 | python tools/check_environment.py --require-cuda |
| tools/prepare_voc_dataset.py | VOC数据准备命令行入口，调用competition.prepare | python tools/prepare_voc_dataset.py --help |
| tools/catalog_datasets.py | 清点固定split、生成数据目录报告；可显式导入base、补XML或计算图片哈希 | python tools/catalog_datasets.py --hash-images |
| tools/audit_dataset.py | 检查缺图、损坏图、尺寸、越界框和跨split完全相同图片 | python tools/audit_dataset.py --profile base --hash-overlap |
| tools/export_rtdetr_onnx.py | 加载RT-DETR模型/权重，导出ONNX，用于后续部署；不是TensorRT引擎构建器 | python tools/export_rtdetr_onnx.py --help |

数据准备、导出和审计属于辅助操作，正常训练已有数据不需要每次重新准备。五帧程序需要真实序列，不能把GPT独立静态图排序后当作真实五帧指标。

### engines/rtdetr/configs：模型和训练配置

| 文件 | 具体作用 |
|---|---|
| runtime.yml | 引擎日志、输出、AMP/EMA等共用运行设置 |
| dataset/competition_base.yml | VEDAI基本项单类COCO数据路径与DataLoader设置 |
| dataset/competition_rtdetr_gpt_low_contrast_combined_v1.yml | GPT低亮组合的单类COCO数据路径与DataLoader设置 |
| rtdetr/competition_r18_base.yml | 基本项实际R18配置：骨干深度18、100queries、3层decoder、1024高/1280宽及优化器/增强设置 |
| rtdetr/competition_r18_rtdetr_gpt_low_contrast_combined_v1.yml | GPT profile的实际R18配置，供迁移适配与评估构建网络和DataLoader |
| rtdetr/include/rtdetr_r50vd.yml | RT-DETR网络/损失/匹配器共用基础模板；实际R18配置覆盖其骨干深度、输入尺寸、queries和decoder层数 |
| rtdetr/include/dataloader.yml | 共用数据增强、DataLoader和collate配置模板 |
| rtdetr/include/optimizer.yml | 共用AdamW、学习率调度、参数组配置模板 |

`rtdetr_r50vd.yml`文件名包含R50，但当前模型是否为R18应看合并后的配置；本项目实际配置覆盖为R18。`competition.rtdetr_runtime.build_config()`会按参数生成或更新两个profile的配置；这些文件不是不可变的历史超参数记录。追溯某次旧实验时，应同时查checkpoint元数据、training_complete.json及log.jsonl。

### engines/rtdetr/tools和src：网络与训练引擎

`engines/rtdetr/tools/train.py` 是底层引擎训练入口，接收配置、恢复/微调权重和随机种子等参数，调用solver；一般由 `training/train_rtdetr.py` 调用。

以下源码文件通常无需单独运行。所有 `__init__.py` 用于导入/注册包内组件；即使没有被六个入口直接引用，也可能在加载 `src` 时被间接导入。

| 子目录/文件 | 具体作用 |
|---|---|
| src/__init__.py | 导入data、nn、optim、zoo组件，供配置注册系统构建对象 |
| src/core/__init__.py | 导出配置类、注册和配置构建函数 |
| src/core/config.py | BaseConfig及模型、优化器、DataLoader等配置对象接口 |
| src/core/yaml_config.py | YAMLConfig：按合并配置延迟构建模型、损失、优化器、EMA和DataLoader |
| src/core/yaml_utils.py | 组件注册、按名称创建对象、读取__include__继承配置并合并字典 |
| src/data/__init__.py | 导入并注册COCO/CIFAR10数据集、变换和DataLoader |
| src/data/dataloader.py | DataLoader、collate及批次拼接定义 |
| src/data/transforms.py | 注册图像/标注增强、Resize、张量转换和框格式变换 |
| src/data/functional.py | 与图像、目标框相关的辅助变换函数 |
| src/data/torchvision_compat.py | 兼容torchvision旧datapoints与新tv_tensors接口，统一框格式和图像转换 |
| src/data/coco/__init__.py | 导出COCO数据读取、评估及辅助函数 |
| src/data/coco/coco_dataset.py | 读取instances_*.json与图片，将标注转为模型需要的target字典 |
| src/data/coco/coco_eval.py | 汇总预测并调用COCO评估，得到mAP50:95/mAP50等；不是比赛Pd/FAR的定义实现 |
| src/data/coco/coco_utils.py | COCO类别映射、数据集API转换等辅助功能 |
| src/data/cifar10/__init__.py | 训练框架保留的CIFAR10数据集注册；当前检测配置使用CocoDetection |
| src/nn/__init__.py | 导入网络架构、基础损失接口和backbone注册模块 |
| src/nn/backbone/__init__.py | 导出PResNet及框架保留的其他骨干 |
| src/nn/backbone/presnet.py | 当前实际使用的PResNet/R18-vd骨干，提取多尺度特征 |
| src/nn/backbone/common.py | 卷积/归一化层、冻结BatchNorm和激活函数等公共层 |
| src/nn/backbone/utils.py | 按层获取中间特征的公共工具 |
| src/nn/backbone/test_resnet.py | 框架中的MResNet备选骨干定义；名字含test但不是本项目的单元测试入口 |
| src/nn/backbone/regnet.py | RegNet备选骨干定义；当前配置未选用 |
| src/nn/backbone/dla.py | DLA备选骨干及其层结构；当前配置未选用 |
| src/nn/arch/__init__.py | 导入框架保留的Classification架构 |
| src/nn/arch/classification.py | 通用分类网络和分类头接口；当前配置的顶层模型为RTDETR |
| src/nn/criterion/__init__.py | 框架基础损失接口/注册入口 |
| src/nn/criterion/utils.py | 将目标字典整理为损失所需张量的辅助工具 |
| src/zoo/__init__.py | 导入检测模型族，完成RT-DETR组件注册 |
| src/zoo/rtdetr/__init__.py | 导出RTDETR、HybridEncoder、decoder、postprocessor、criterion和matcher |
| src/zoo/rtdetr/rtdetr.py | RTDETR整体网络，将backbone、encoder、decoder串联；定义forward和部署转换 |
| src/zoo/rtdetr/hybrid_encoder.py | HybridEncoder：对骨干多尺度特征进行注意力处理和跨尺度融合 |
| src/zoo/rtdetr/rtdetr_decoder.py | RTDETRTransformer、query选择、解码层和各层目标框/置信度预测 |
| src/zoo/rtdetr/denoising.py | 训练期构造带噪目标query，提供去噪监督；推理时不需要人工输入GT |
| src/zoo/rtdetr/rtdetr_criterion.py | SetCriterion，计算分类/置信度、L1框回归、GIoU及辅助/去噪损失 |
| src/zoo/rtdetr/matcher.py | HungarianMatcher，将训练预测与真实目标进行匈牙利匹配 |
| src/zoo/rtdetr/rtdetr_postprocessor.py | 将归一化预测框转为原图像素坐标，并输出top-k框和分数 |
| src/zoo/rtdetr/box_ops.py | 框坐标转换、IoU/GIoU等网络训练所需操作 |
| src/zoo/rtdetr/utils.py | 反sigmoid、可变形注意力等encoder/decoder辅助运算 |
| src/solver/__init__.py | 导出和注册solver类 |
| src/solver/solver.py | 通用训练器基类，管理模型、设备、优化器和checkpoint状态 |
| src/solver/det_solver.py | 检测任务训练/验证总循环、最佳状态及checkpoint保存 |
| src/solver/det_engine.py | 单个epoch的训练和COCO验证函数，基本项、迁移和增量共用 |
| src/optim/__init__.py | 导出优化器、调度器、EMA和AMP组件 |
| src/optim/optim.py | 注册/构建PyTorch优化器和学习率调度器 |
| src/optim/ema.py | 模型参数指数滑动平均EMA及其更新/保存 |
| src/optim/amp.py | 自动混合精度的梯度缩放等训练支持 |
| src/misc/__init__.py | 导出日志及可视化工具 |
| src/misc/dist.py | 分布式初始化、进程同步、模型包装和随机种子工具 |
| src/misc/logger.py | 平滑统计loss/耗时、训练进度和多进程指标汇总 |
| src/misc/visualizer.py | 展示训练样本和框等调试可视化工具 |

网络仍使用单类前景置信度损失来判断“是否有目标”。比赛不要求语义类别输出，不表示可以删除criterion中的分类/置信度分支。训练期Hungarian匹配与 `competition/metrics.py` 的比赛测试匹配作用不同。

### meta_learning：普通迁移及可选FOMAML

| 文件 | 具体作用 |
|---|---|
| __init__.py | 将meta_learning声明为包，便于增量训练器调用公共函数 |
| common.py | 随机种子、checkpoint读取与保存、匹配权重、冻结backbone、构建配置、损失计算和批次搬移；普通迁移及增量都需要 |
| 01_prepare_meta_tasks.py | 从base/train的manifest.csv按来源/显式domain-map划分源域任务，生成tasks/base清单 |
| 02_train_fomaml.py | 从基本项权重出发，在VEDAI真实/合成等源任务上执行support内更新、query一阶元更新，训练可快速适配的初始化 |
| 03_prepare_fewshot_support.py | 只从目标train生成嵌套5/10/20-shot支持集，保存COCO子集及样本顺序 |
| 04_adapt_fewshot.py | 加载standard或fomaml初始化，在指定支持集上训练encoder/decoder，验证选best |
| 05_compare_fewshot.py | 比较指定shot/method模型；验证选confidence/NMS，固定后测test并写比较表 |
| 06_evaluate_zero_shot.py | 基础VEDAI权重直接测试GPT目标域的零样本基线，仍由val选阈值 |
| run_all.py | 可选的完整FOMAML对照流水线：准备任务、元训练、支持集、两种适配和评估；不是日常普通迁移入口 |
| README.md | 元学习任务设置、源域/目标域定义及运行步骤 |
| LOW_CONTRAST_TRANSFER_README.md | 已完成的GPT低亮普通迁移/FOMAML对照实验，含划分、超参数、权重和指标 |

当前 `train_transfer.py` 使用standard普通迁移；要运行FOMAML请显式使用该目录脚本。现有base目录没有manifest.csv，tasks清单也未生成；重新进行元训练前须准备可追溯的来源manifest/domain-map。普通迁移和现有FOMAML权重测试不需要重新生成tasks。

`support/rtdetr_gpt_low_contrast_combined_v1/` 内文件：

| 文件 | 具体作用 |
|---|---|
| support_005.json | 5张目标train支持图和对应COCO目标框 |
| support_010.json | 10张支持图及目标框，包含上述5张 |
| support_020.json | 20张支持图及目标框，包含上述10张 |
| support_manifest.csv | 支持候选样本排序、图片ID/名字、目标数及是否入选各shot的记录 |
| summary.json | 抽样seed、源标注路径、各shot图像/目标数量 |

支持图片本身位于 `datasets/prepared/rtdetr_gpt_low_contrast_combined_v1/images/train`，没有重复复制。现有支持集seed为20260924；重新抽样可能覆盖支持清单并改变旧实验含义，需要固定相同参数。

### incremental_after_transfer：迁移后的两阶段域增量

| 文件 | 具体作用 |
|---|---|
| prepare.py | 旧pilot的阶段数据准备：项目内旧低亮GPT域、外部GPT V3和D:\导出后，构成新域样本加旧train回放；不自动用于现场比赛的新数据 |
| train.py | 阶段1从standard_010shot/best.pth初始化，阶段2从阶段1best初始化；调用competition.incremental_trainer |
| test.py | 比较迁移初始模型/阶段1/阶段2，在各模型已见域val上选阈值，再输出old/v3/new域test结果 |
| README.md | 阶段顺序、数据依赖、初始权重和增量状态说明 |

旧pilot只保留代码，`inc_after_transfer_stage1`、`inc_after_transfer_stage2`数据和两阶段输出目前均不存在。该流程维持一个target类，变化的是风格/场景域。现场下发数据后应先按实际域重新配置prepare，再运行两个根入口。

### datasets：图片和标注

| 路径/文件 | 具体作用 |
|---|---|
| README.md | 数据来源、保留数量、固定划分和去重说明 |
| prepared/base/ | 基本项VEDAI原图及传统合成数据；train891/val80/test79 |
| prepared/rtdetr_gpt_low_contrast_combined_v1/ | GPT低亮模拟域；train283/val60/test62，共405张 |
| 每个profile的images/train、images/val、images/test/*.png | 分别用于训练、选权重/阈值、最终测试的图像 |
| 每个profile的voc_xml/train、voc_xml/val、voc_xml/test/*.xml | 与图片一一对应的VOC矩形框标注，保留坐标用于检查/对接 |
| 每个profile的annotations/instances_train.json | RT-DETR实际读取的COCO训练标注 |
| 每个profile的annotations/instances_val.json | COCO验证标注，用于选择权重和阈值 |
| 每个profile的annotations/instances_test.json | COCO测试标注，用于固定模型与阈值后的评估 |
| 每个profile的train.txt、val.txt、test.txt | 原有划分的图像清单；当前COCO加载器主要读取JSON |
| base/provenance.json | VEDAI导入来源、原JSON路径、划分未变和存储方式的记录 |
| GPT profile的manifest.csv | 图像来源、split及目标数等样本来源记录 |
| GPT profile的dataset_summary.json | 数据准备时的来源、单类设置、split统计等记录 |
| GPT profile的experiment_profile.json | GPT组合实验的划分比例、seed、图像尺寸和评估规则等设置 |

部分来源记录仍为服务器或外部原始路径，用于追溯历史；当前加载图片和标注使用project内prepared路径。metadata内的自动序列备注是历史记录，项目已清理自动五帧分组文件。图片可能是硬链接，修改内容也可能影响外部原图，日常训练应只读。

### weights：保留的初始化权重

| 文件 | 具体作用 |
|---|---|
| README.md | 初始化权重和训练输出best路径说明 |
| rtdetr/vedai_real_synth_best.pth | 历史VEDAI原图+合成图训练的最优权重；普通迁移初始化及基本项历史测试使用 |
| rtdetr/vedai_real_synth_fomaml_best.pth | VEDAI源域FOMAML元训练后的初始化；还需在新域支持集适配才能得到对应迁移模型 |

两个文件不是新basic scratch训练的产物；新基本项权重由训练保存到 `outputs/rtdetr/base/best.pth`。

### outputs：迁移权重、日志和评估结果

当前保留的训练输出为 `meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/`。`standard`代表普通迁移，`fomaml`代表从元训练初始化适配；005/010/020表示支持图片数。

| 子目录 | 含义 |
|---|---|
| standard_005shot、standard_010shot、standard_020shot | 普通迁移5/10/20张目标域支持图的训练产物 |
| fomaml_005shot、fomaml_010shot、fomaml_020shot | FOMAML初始化在同样5/10/20张支持图上适配的训练产物 |

六个训练目录中的文件用途一致：

| 文件模式 | 具体作用 |
|---|---|
| best.pth | 验证选中的最优模型参数及checkpoint元数据，用于测试；可包含初始化路径、参数与训练模块信息 |
| log.jsonl | 逐轮训练loss、验证mAP50:95/mAP50等记录，用于查看训练过程和分析过拟合 |
| training_complete.json | 完成状态、最佳epoch、最佳验证mAP、最优权重路径和训练时长 |
| tensorboard/events.out.tfevents.* | TensorBoard事件文件，保存可视化训练曲线 |

可运行 `tensorboard --logdir outputs/meta_learning/adaptation` 查看曲线。训练超参数可从checkpoint中的metadata、project_config及原命令追溯；只看best.pth文件名不能推断数据来源。

评估结果目录为 `meta_learning/evaluation/rtdetr_gpt_low_contrast_combined_v1/`，包含同名六个迁移子目录及 `zero_shot/`。零样本表示基础VEDAI权重未在GPT支持集上适配。

| 文件模式 | 具体作用 |
|---|---|
| comparison.csv、comparison.json | 普通迁移与FOMAML的5/10/20-shot总体对比 |
| comparison_with_zero_shot.csv、comparison_with_zero_shot.json | 加入zero-shot基线的完整对比表 |
| 每个方法目录的val_predictions.csv | 验证集预测框与置信度，用于阈值/NMS选择及复查 |
| 每个方法目录的test_predictions.csv | 测试集预测框与置信度，不等于全部行都通过最终阈值 |
| validation/summary.json | 验证选中阈值对应指标、confidence/NMS设置等 |
| validation/threshold_sweep.csv | 验证置信度扫描中的Pd/FAR等，便于分析取舍 |
| validation/nms_grid.json | 不同NMS设置的验证搜索结果 |
| validation/per_image.csv | 验证逐图TP/FP/FN等统计 |
| test/summary.json | 最终固定验证阈值后的TP/FP/FN、Pd/FAR/F1等，是查最终测试指标的主要文件 |
| test/per_image.csv | 测试逐图目标与误检统计，定位漏检/虚警样本 |
| zero_shot/result.json | 零样本评估的配置、来源权重及结果汇总 |

后续训练/测试会创建尚不存在的输出路径：

| 路径/文件模式 | 何时生成及用途 |
|---|---|
| rtdetr/base/best.pth | 基本项训练验证选中最优权重 |
| rtdetr/base/checkpoint.pth | 基本项训练断点，供自身继续训练 |
| rtdetr/base/competition_evaluation/ | 基本项val/test预测、阈值和测试指标 |
| incremental_after_transfer/stage1、stage2/ | 增量训练各阶段best/last、log.jsonl、progress.json、training_complete.json |
| incremental_after_transfer/evaluation/ | 增量初始及两阶段模型的逐域评估与comparison.csv |
| meta_learning/tasks/base/ | 元任务准备后的domain_manifest.csv与summary.json；实际位置在项目meta_learning目录内，不在outputs内 |
| meta_learning/fomaml/base/ | 重新元训练后的best/last和日志；此处指outputs/meta_learning/fomaml/base |

### docs和reports：技术文档与清理/数据记录

| 文件 | 具体作用 |
|---|---|
| docs/RTDETR_INFRARED_SMALL_TARGET_MODEL_TECHNICAL_DOCUMENT.md | 模型结构、损失、配置、训练和历史提交包的技术说明 |
| docs/rtdetr_project_architecture.png | 模型/项目架构示意图，配合技术文档阅读 |
| reports/EXPERIMENT_RESULTS.md | 早期AA-YOLO/HIT-UAV/道路实验的历史结果汇总；相关模型或数据已清理，不能当作当前保留模型的成绩 |
| reports/experiment_results.csv | 上述历史实验表的CSV版本 |
| reports/dataset_audit_competition_road_v4.json | 已移除road_v4数据的旧结构审计记录 |
| reports/dataset_audit_rtdetr_low_contrast_v1.json | 已移除混合low_contrast_v1数据的旧审计记录，不是当前405张组合数据的审计 |
| reports/subfolder_cleanup_20261003.json | 本次子目录移出清单、归档路径、保留数量和运行验证结果 |
| reports/organization_20261002/SUMMARY.md | 项目整理状态说明 |
| reports/organization_20261002/dataset_catalog.csv | 当前保留profile/split数量和目标数清单 |
| reports/organization_20261002/dataset_catalog.json | 清单及精确重复/缺文件等审计的结构化结果 |
| reports/organization_20261002/incremental_deletion_manifest.json | 已删除增量权重、结果和派生数据的历史清单 |
| reports/organization_20261002/rtdetr_only_cleanup.json | 只保留RT-DETR、合并为六入口及清理其他数据的历史清单 |

当前GPT迁移指标请优先查 `outputs/meta_learning/evaluation/rtdetr_gpt_low_contrast_combined_v1/`；reports中早期失败实验保留为历史记录。

### tests：运行检查

| 文件 | 具体作用 |
|---|---|
| test_project_layout.py | 检查入口路径、随机初始化、普通迁移参数、图像/XML存在性及增量初始化依赖 |
| smoke_test.py | 检查指标计算和R18模型构建；--forward额外运行一次CPU1024x1280前向，不启动训练 |

### vendor/windows_py38/pycocotools：Windows兼容依赖

| 文件 | 具体作用 |
|---|---|
| __init__.py | pycocotools包标识 |
| coco.py | COCO标注API：读取图片/目标信息和查询标注 |
| cocoeval.py | COCO mAP等评估计算 |
| mask.py | 掩码/RLE编码操作的Python接口；框检测使用COCO框架时也会导入 |
| _mask.cp38-win_amd64.pyd | Python3.8/Windows64位编译扩展，供mask.py调用 |

此兼容包由platform_deps.py按平台选择。Linux服务器及其他Python版本需要安装与其平台匹配的pycocotools，不能直接使用此pyd文件。

### 查找入口、权重和指标的顺序

1. 要训练/测试：先看根目录对应的六个入口和上面的运行顺序。
2. 要调整网络：看engines/rtdetr/src/zoo/rtdetr及nn/backbone/presnet.py。
3. 要确认图像/标签：看datasets/prepared/<profile>/images和voc_xml；模型实际读取annotations中的JSON。
4. 要找迁移最优权重：看outputs/meta_learning/adaptation/<profile>/<method>_<shots>shot/best.pth。
5. 要找最终指标：看outputs/meta_learning/evaluation/<profile>/<method>_<shots>shot/test/summary.json。
6. 要追溯超参数和训练过程：读取best.pth元数据、training_complete.json、log.jsonl及TensorBoard事件。
