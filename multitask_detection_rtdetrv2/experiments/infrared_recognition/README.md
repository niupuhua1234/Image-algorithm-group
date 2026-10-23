# 红外目标识别集成

本目录保存 `D:/project/RGBT/RT-DETR-main` 的红外识别实验、权重和 TensorRT 筛选入口。原 RGBT 工程不复制、不修改；现有统一工程的 `infrared_small_target` 和 `visible_recognition` 分支也不修改。

## 来源与配置

- 来源代码：`D:/project/RGBT/RT-DETR-main`
- 标准任务配置：`configs/tasks/infrared_recognition.yaml`；[task.yaml](task.yaml) 仅保留旧命令兼容
- 类别：`car1`、`car2`、`car3`、`car4`
- 输入：`1024×1280`（高×宽）
- RT-DETRv2-R18VD，3 层 decoder，300 queries
- 评估主 IoU：`0.40`
- 训练精度：CUDA AMP（来源工程 `use_amp: True`）
- 训练增强：RandomIoUCrop(0.10，140 epoch 后关闭)、水平/垂直翻转、PhotometricDistort(0.7)、Resize
- 训练选择：`best.pth` 按 R40 → R50 → R60 → AP40；`best_AP40.pth` 单独保存 AP40 最佳
- 每 10 epoch 保存 `checkpoint0009.pth` 等候选；训练结束后以 TensorRT FP16 验证集复核候选
- 训练初始化权重：本工程 `source_import/weights/best.pth`（已从你的 `VEDAIbest-400best-200/best.pth` 导入）
- 默认部署权重：本工程 `finetune/weights/best.pth`
- 模型结构与损失配置：`configs/models/rtdetrv2_r18_infrared.yaml`
- TensorRT：已支持四类 `xyxy + C 个 sigmoid score` 输出；FP16 选择引擎中 LayerNorm 保持 FP32

## 运行

在统一工程根目录执行：

```powershell
$py = 'D:\software\Anaconda\envs\RT-DETR\python.exe'
& $py -B experiments/infrared_recognition/train.py --show-config
& $py -B experiments/infrared_recognition/train.py --dry-run --device cuda
# 正式训练会自动按来源 AP40/Recall 规则选权重、标定部署阈值并生成 deployment/frozen_task.yaml。
& $py -B experiments/infrared_recognition/train.py --name finetune --device cuda
& $py -B tools/eval.py --config experiments/infrared_recognition/finetune/deployment/frozen_task.yaml --split val --device cuda --out experiments/infrared_recognition/validation_val.json
& $py -B tools/eval.py --config experiments/infrared_recognition/finetune/deployment/frozen_task.yaml --split test --device cuda --out experiments/infrared_recognition/validation_test.json
& $py -B tools/infer.py --config experiments/infrared_recognition/finetune/deployment/frozen_task.yaml --image '你的图片.png' --device cuda
# 可选：只在当前 TensorRT/CUDA 环境稳定时，对已完成训练重新执行 TensorRT 候选复核。
& $py -B experiments/infrared_recognition/select_trt.py --run-dir experiments/infrared_recognition/finetune --force-enable --device cuda
# 一般不需要手动运行；只在要复核已有训练结果时执行。
& $py -B experiments/infrared_recognition/calibrate_threshold.py --run-dir experiments/infrared_recognition/finetune --device cuda
```

评估与推理统一使用 `tools/eval.py`、`tools/infer.py`；运行时通过 `--task infrared_recognition` 或红外冻结配置 `--config` 指定任务，入口内部会加载任务适配器。原 `eval.py`、`infer.py` 兼容入口已删除。正式训练完成并生成冻结配置后，上述统一入口会使用本次选中的权重和标定阈值。

如需改数据位置：

```powershell
$env:INFRARED_RECOGNITION_DATA_ROOT = '你的数据集根目录'
```

原工程的 `best_AP40.pth` 是另一套 AP 优先初始化权重。若要使用它微调，可将 `train.initialize` 改为该权重在本工程中的导入路径。训练完成前，默认部署路径不存在，因此不能直接运行评估或推理。

正式训练默认以本工程已导入的 `source_import/weights/best.pth` 初始化。以 `--name finetune` 完成训练后，产物直接位于 `experiments/infrared_recognition/finetune/`；其中 `weights/best.pth`（召回优先）会自动成为默认评估和推理权重，`weights/best_AP40.pth` 为 AP40 优先权重。不要把初始化权重当作本次训练的最终结果。

也可以从统一入口训练；该任务仍会执行红外专用的断点、输出整理和中断保护流程：

```powershell
& $py -B tools/train.py --task infrared_recognition
```

训练中断后，使用原运行名称和 `--resume` 从最近完成的一轮继续：

```powershell
& $py -B tools/train.py --task infrared_recognition --name finetune --resume
```

入口会在正式输出目录或中断留下的临时目录中查找 `weights/last.pth`。该文件保存模型、优化器、学习率调度、EMA 和随机状态；尚未完成的一轮会重新训练。继续训练时需保持原任务、模型、训练配置及训练/验证标注一致。如果中断前还没完成第一轮，就没有可恢复的轮次。

## 部署阈值标定

`deployment.conf_threshold: 0.60` 只是在尚未标定时的临时默认值。红外专用 `train.py` 在每次非冒烟正式训练结束后会自动执行 `calibrate_threshold.py`：它只在 **val** 上执行一次零阈值推理，然后复用缓存预测扫描 0.01～0.99 的阈值。选择规则是 **四类宏平均 F1 最大**；完全相同时依次取宏平均召回更高、宏平均精度更高、阈值更高的结果。

结果保存在本次 run 的 `deployment/threshold_calibration.json`，可复核每个候选阈值；同时生成 `deployment/frozen_task.yaml`，其中固定了本次 `best.pth` 的 SHA256 和选中的 `conf_threshold`。后续推理与测试使用这个冻结配置，而不是修改全局配置或其它任务。

## 设计边界

本接入复用统一工程已经固定的 RT-DETRv2 模型、VFL/L1/GIoU 损失、COCO 数据契约、Detection 输出和通用评估记录。红外任务单独启用来源配方的 AMP、增强、LinearWarmup(16 step)、COCO AP40/R40/R50/R60 选择和 TensorRT FP16 候选复核；这些能力不改变其它任务的训练配方。训练评估保留 RT-DETR 的 300 个候选框；默认业务推理仍使用 0.60 阈值。
