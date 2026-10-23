# RT-DETR v2 统一检测工程

按《项目技术路线书_RT-DETRv2》第5、6页整理。工程主线为 **RT-DETR v2**，当前完整接入的业务任务是 **红外小目标检测，模型为 RT-DETRv2-S**。类别只有 `target`，输出原图像素坐标框、分数和固定标签0。

公共入口采用继承与多态，任务和算法分别配置。统一工程不要求各任务使用相同输入尺寸、类别、权重、损失或阈值。

## 任务状态

| 任务 | 状态 | 输入尺寸 高×宽 | 类别与后处理 |
|---|---|---|---|
| `infrared_small_target` | 已接通并验证训练、恢复、评估、推理、导出 | **1024×1280** | 单类；保留已有置信度与可选NMS策略 |
| `infrared_recognition` | 预留，待该任务负责人补充 | **待确定** | 独立业务类别、权重、损失及评估 |
| `visible_recognition` | 保留交接版实现，统一入口暂不启用 | 交接训练配置960×960；原部署配置另行保留 | 6类；DEIM配套损失；集合后处理 |

可见光交接版的代码与配置来自 `D:/multitask_detection_handover_20261015(1)/multitask_detection`，原件未修改。其历史指标属于原项目，未在本机重新训练或复验。相关说明保存在 `docs/handover_visible/`；`tools/visible_*.py` 是旧入口兼容存档，不是当前红外入口。

## 目录

```text
configs/
  tasks/         三个业务任务，各自的数据、类别、评估口径
  models/        算法及网络组件配置，RT-DETRv2-S与任务解耦
  training/      独立训练配方、优化器、EMA及初始化
  deployment/    尺寸、后端、权重、引擎、阈值
src/
  core/          配置、任务注册、公共工具
  data/          COCO索引、原始尺寸、变换记录、采样
  models/
    backbones/ encoders/ decoders/ heads/
    adapters/    TaskAdapter → DetectionTaskAdapter → DetrTaskAdapter → 业务子类
  losses/        与算法配套的损失和匹配
  optim/         公共AdamW分组与EMA
  engine/        公共训练、验证、结果与断点管理
  metrics/       框级识别与小目标业务指标
  deployment/    公共运行时、PyTorch/TensorRT后端、导出
tools/           统一命令入口
third_party/     固定源码快照与许可证，避免与本工程src命名冲突
experiments/     按任务/算法/阶段/实验隔离
docs/ tests/
```

具体继承关系与差异边界见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)。PDF条目映射见 [PDF_COMPLIANCE.md](docs/PDF_COMPLIANCE.md)。

## 本机运行

本机验证环境：Python3.8.20、torch2.0.1+cu118、torchvision0.15、GTX1650Ti。使用已有环境，不要求升级可见光环境。

在本工程根目录执行 PowerShell：

```powershell
$py = 'D:\MySoftware\Anaconda3\envs\aa_yolo\python.exe'
& $py -B tools/list_tasks.py
& $py -B tools/train.py --task infrared_small_target --show-config
& $py -B tools/train.py --task infrared_small_target --dry-run
& $py -B tools/eval.py --task infrared_small_target --split test --backend pytorch --out experiments/infrared_small_target/manual_check/pytorch.json
```

默认数据使用 `D:/InfraredSmallTargetDetection/datasets/prepared/texture_20261014`，只读引用，未复制或重划分。可通过 `--data-root` 或环境变量 `INFRARED_DATA_ROOT` 指定自己的COCO数据。必须按任务配置填写 `category_id_to_label`，不能混用其他任务的编号。

默认推理权重为导入的已训练 v2-S：`experiments/infrared_small_target/texture_v2_s_imported/weights/best.pth`。同时保存同权重的TensorRT引擎及SHA256构建记录。它们不是短训练自检生成的模型。

### TensorRT推理

如果环境已安装匹配的TensorRT，直接使用。若使用本机独立依赖目录：

```powershell
$env:TENSORRT_ROOT = 'C:\Users\NewAdmin\Documents\ChatGPT\红外小目标\rtdetr_latency_lab_20261009\deps'
& $py -B tools/eval.py --task infrared_small_target --backend tensorrt --split test --bench 200 --out experiments/infrared_small_target/manual_check/tensorrt.json
& $py -B tools/infer.py --task infrared_small_target --backend tensorrt --image '你的图片.png'
```

依赖路径由环境变量显式提供；网络构建与训练不导入旧实验代码。引擎会校验权重和引擎哈希、GPU型号与输入尺寸。更换设备或GPU时需重新构建。

### 训练与恢复

```powershell
& $py -B tools/train.py --task infrared_small_target --name basic_01
& $py -B tools/train.py --task infrared_small_target --name basic_01 --resume experiments/infrared_small_target/rtdetrv2_s/basic/basic_01/weights/last.pth
```

训练入口真实执行训练。默认复用统一R18基础配方：官方COCO初始化、FP32、FrozenBN、EMA、AdamW、原图尺寸及水平翻转，最多72轮。已有导入模型的历史自适应训练记录独立归档；此整理版本不声称逐步复现历史的早停轨迹。

每个实验分别保存配置、类别映射、模型、EMA、优化器、学习率调度器、随机状态、验证阈值和数据标注哈希。`last.pth`是完整恢复断点，`best.pth`为所选EMA推理权重。新实验名已存在时拒绝覆盖。

`--max-steps`和`--val-limit`仅用于短流程验证；产生的权重带 `smoke_test` 标记，不能当正式性能模型。正式训练不要使用这两个参数。

### 验证集选阈值

```powershell
& $py -B tools/calibrate.py --task infrared_small_target --ckpt '你的best.pth' --pd-target 0.92 --output-dir experiments/infrared_small_target/calibration_01
# 使用生成的frozen_task.yaml进行固定阈值测试
& $py -B tools/eval.py --config experiments/infrared_small_target/calibration_01/frozen_task.yaml --split test --out experiments/infrared_small_target/calibration_01/test.json
```

阈值选择入口只读val。推理权重改变时须重新在val选工作点，不能沿用旧权重的阈值。现有固定测试集已被反复分析，工程核对属于回归检查。

### 导出与构建

```powershell
# onnx属于可选依赖；本机可复用已有依赖目录，也可在独立环境安装requirements-export.txt
$env:PYTHONPATH = $env:TENSORRT_ROOT
& $py -B tools/export.py --task infrared_small_target --action onnx --output-dir experiments/infrared_small_target/export_01
& $py -B tools/export.py --task infrared_small_target --action build --output-dir experiments/infrared_small_target/export_01
```

导出保留1280×1024、100 queries、3层decoder，复用已验证的部署分支融合、Conv/BN融合和Top-k框头优化。新导出引擎应独立做精度回归；默认仍使用已核对的导入引擎。

## 评估口径

小目标使用IoU≥0.4、置信度降序、类别无关、一对一匹配。

- Pd / Recall = TP / (TP + FN)。
- Precision = TP / (TP + FP)。
- Fa / FAR = FP / (TP + FP)，分母为检测目标数。
- 零分母返回JSON null，并明确为未定义。
- 按原图框的等效边长sqrt(w×h)统计各尺寸组的GT、TP、FN和Pd；组区间左闭右开。
- 辅助记录FP/图、负图误警图片比例、显存分配统计及其覆盖范围。

均时从已解码CPU RGB输入到CPU Detection就绪，包含预处理、H2D、网络、解码、D2H、阈值和NMS；排除磁盘解码、初始化、预热及指标匹配。是本机单模型结果，未验证Orin三模型并发。完整记录见 [VALIDATION.md](docs/VALIDATION.md)。

运行检查：`& $py -B -m unittest discover -s tests -v`。
