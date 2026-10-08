# 红外目标识别代码（ZXS）

负责人：ZXS；GitHub：`zhangjinyuan0507-dev`。协作分支：`Infrared-Target-Recognition`。

本目录从现有研究工程中提取红外车辆检测与识别代码，提供 DroneVehicle 标注转换、标签检查、YOLOv8s 训练、单张/批量推理、真值导出和比赛口径评测。输入为红外图像，输出为车辆框、类别和置信度；不是只做整图分类，也不是红外/可见光融合模型。

## 类别顺序

类别 ID 必须保持不变，沿用本项目已经核验的 DroneVehicle 映射：

| ID | 类别 |
|---|---|
| 0 | car |
| 1 | freight_car |
| 2 | truck |
| 3 | bus |
| 4 | van |

## 文件说明

```text
configs/infrared_recognition.yaml      只加载红外识别专家的推理配置
configs/pretrained_yolov8s.manifest.json 历史初始化权重来源及校验值
scripts/convert_dronevehicle_to_yolo.py XML 多边形转 YOLO 水平框
scripts/build_ir_dataset_yaml.py       生成独立 train/val 数据 YAML
scripts/audit_yolo_dataset.py          标签审计和可视化
scripts/train_detector.py              训练入口，不修改源 data.yaml
scripts/unified_inference.py           单张推理，任务由配置决定
scripts/predict_yolo_dataset.py        批量预测并导出 JSONL
scripts/export_yolo_ground_truth.py    YOLO 真值转 JSONL
scripts/evaluate_competition_metrics.py IoU/类别匹配评测
scripts/sweep_competition_thresholds.py 验证集阈值选择
tests/smoke_ir_pipeline.py             无需真实数据/权重的工程冒烟检查
```

## 安装环境

原工程训练环境为 Python 3.8、PyTorch 2.0.1、TorchVision 0.15.2、Ultralytics 8.0.196。建议新建独立环境，不覆盖现有环境。以下 PowerShell 命令均在本目录运行：

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

CUDA 安装示例用于具备兼容驱动的 NVIDIA GPU。仅检查工程链路时可改用 CPU 版本并传入 `--device cpu --fp32`；安装方式参考 [PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)。

## 数据下载与准备

使用 [DroneVehicle 官方数据页面](https://github.com/VisDrone/DroneVehicle#dataset)。下列地址和提取码来自官方 README（2026-10-08 核对页面，未下载整个数据集）：

| 划分 | 下载地址 | 提取码 |
|---|---|---|
| 训练 | https://pan.baidu.com/s/1ptZCJ1mKYqFnMnsgqEyoGg | ngar |
| 验证 | https://pan.baidu.com/s/1e6e9mESZecpME4IEdU8t3Q | jnj6 |
| 测试 | https://pan.baidu.com/s/1JlXO4jEUQgkR1Vco1hfKhg | tqwc |

将原始压缩包分别解压到新的目录，例如 `datasets/raw/DroneVehicle/train` 和 `datasets/raw/DroneVehicle/val`。转换器会在每个划分下查找官方目录：训练使用 `trainimg/trainlabel/trainimgr/trainlabelr`，验证使用对应的 `val*` 目录；允许存在外层嵌套目录，但不允许同名目录有多个候选。`img/label` 为 RGB，`imgr/labelr` 为红外。

转换器保留原工程的双模态转换能力，需要两个模态的图像及 XML；本任务只用派生的 `infrared` 数据。原始文件只读，不上传图片、标签全集或压缩包。

```powershell
python scripts/convert_dronevehicle_to_yolo.py --split train=datasets/raw/DroneVehicle/train --split val=datasets/raw/DroneVehicle/val --output datasets/processed/dronevehicle_zxs_v1
python scripts/build_ir_dataset_yaml.py --dataset-root datasets/processed/dronevehicle_zxs_v1/infrared --output datasets/processed/dronevehicle_zxs_v1/infrared/data.yaml
python scripts/audit_yolo_dataset.py --data datasets/processed/dronevehicle_zxs_v1/infrared/data.yaml --output experiments/audit/ir_zxs_v1 --splits train val --visualizations-per-split 30
```

默认去掉官方图像四周 100 像素白边，840×712 图像变为 640×512；多边形转成裁剪后的水平外接框。未知 `*` 对象和仅有点标注的对象跳过，并记录统计。输出目录必须使用新名称；转换与配置生成不会覆盖已有结果。先查看审计报告及可视化框，再训练。

训练和验证使用两个独立官方划分，不能把测试集作为验证集。这里的 YAML 由转换结果生成，类别顺序从转换器读取并校验。

## 初始化权重与训练

YOLOv8 参考实现和权重说明见 [Ultralytics 官方文档](https://docs.ultralytics.com/models/yolov8/)。原工程使用的 YOLOv8s 初始化权重来源为：

https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

历史文件大小及 SHA-256 记录在 `configs/pretrained_yolov8s.manifest.json`。本次仅提交该清单，不提交 `.pt`。将初始化权重自行放到 `models/yolov8s.pt`；下载文件可以先运行 `Get-FileHash models/yolov8s.pt -Algorithm SHA256` 核对清单。旧链接若失效，请从官方页面获取兼容权重并自行记录版本，不要把不同文件当成同一个初始化实验。

```powershell
python scripts/train_detector.py --model models/yolov8s.pt --data datasets/processed/dronevehicle_zxs_v1/infrared/data.yaml --project experiments/training --name ir_zxs_v1 --epochs 100 --imgsz 640 --batch 4 --device 0 --workers 0 --no-amp
```

示例使用 `--no-amp`，避免首次运行还需要 AMP 校验权重。若启用 AMP，需要另备本地 `yolov8n.pt`，去掉 `--no-amp` 并增加 `--amp-reference models/yolov8n.pt`。训练脚本要求本地模型文件存在，不自动下载。

训练输出在 `experiments/training/ir_zxs_v1/`；运行时数据配置在 `experiments/training/_runtime_data/`。再次运行请更换 `--name`。本次提交不包含训练完成的红外五类 `best.pt`，也不提供尚未建立的权重网盘链接。

## 单张与批量推理

完成训练后，修改 `configs/infrared_recognition.yaml` 的 `weight` 为自己训练得到的 `best.pt` 相对路径，例如 `../experiments/training/ir_zxs_v1/weights/best.pt`；也可把权重自行复制到默认位置 `weights/infrared_recognition/best.pt`。配置只包含 `infrared_recognition`，不需要另外两个专家权重。必须使用上述五类红外权重，不能用 COCO 初始化权重替代训练结果。

```powershell
python scripts/unified_inference.py --registry configs/infrared_recognition.yaml --task infrared_recognition --image datasets/processed/dronevehicle_zxs_v1/infrared/images/val/你的图片.jpg --device 0 --output experiments/predictions/ir_single_v1.json
python scripts/predict_yolo_dataset.py --registry configs/infrared_recognition.yaml --task infrared_recognition --data datasets/processed/dronevehicle_zxs_v1/infrared/data.yaml --split val --device 0 --output experiments/evaluation/ir_predictions_v1.jsonl
```

第一条命令中的图片名应替换成真实文件名。单张推理输出像素坐标 `bbox=[x1,y1,x2,y2]`、`class_id`、`class_name` 和 `score`；批量结果为逐图 JSONL。

## 评测与阈值选择

```powershell
python scripts/export_yolo_ground_truth.py --data datasets/processed/dronevehicle_zxs_v1/infrared/data.yaml --split val --output experiments/evaluation/ir_ground_truth_v1.jsonl
python scripts/evaluate_competition_metrics.py --ground-truth experiments/evaluation/ir_ground_truth_v1.jsonl --predictions experiments/evaluation/ir_predictions_v1.jsonl --iou 0.40 --confidence 0.001 --output experiments/evaluation/ir_metrics_v1.json
python scripts/sweep_competition_thresholds.py --ground-truth experiments/evaluation/ir_ground_truth_v1.jsonl --predictions experiments/evaluation/ir_predictions_v1.jsonl --task-metric recognition_rate --target 0.93 --iou 0.40 --output experiments/evaluation/ir_thresholds_v1.json
```

识别率为类别正确且 IoU≥0.40 的匹配数 / 真值目标总数，与 Ultralytics 的 mAP 不同。`--target 0.93` 是验证集选阈值的目标，不代表已经达到该精度；是否达到见输出 `target_met`。推理配置中的 0.001 用于保留阈值扫描候选。正式推理应使用验证集选出的阈值；测试集不用于选阈值。所有输出文件使用新名称，拒绝覆盖。

## 工程检查与验证边界

```powershell
python tests/smoke_ir_pipeline.py --output experiments/smoke/ir_zxs_v1
python tests/smoke_ir_pipeline.py --output experiments/smoke/ir_zxs_model_v1 --model-smoke
python scripts/evaluate_competition_metrics.py --self-test
```

冒烟检查生成小规模合成 RGB/红外图像与 XML，验证原始输入不变、100 像素裁边、五类顺序、独立 train/val、拒绝覆盖、标签审计、真值导出和评测。`--model-smoke` 额外使用随机初始化的五类模型在 CPU 上验证单专家加载和单张/批量预测，不下载权重，不训练正式模型。

本次工程检查使用本机已有 Python 3.10.9、PyTorch 2.9.1 CPU、Ultralytics 8.3.228；它不替代上述 Python 3.8 / Ultralytics 8.0.196 训练环境复现。此提交未重新进行正式训练、真实数据精度或 Jetson 时延验证，不据此宣称满足现场指标。

本目录只提交源码、配置、依赖、文档和检查脚本。数据、权重、实验产物和虚拟环境受 `.gitignore` 排除；使用公开数据和第三方依赖时遵守其原有使用条款。第三方算法依赖通过安装获取，不在本次上传中复制第三方源码。
