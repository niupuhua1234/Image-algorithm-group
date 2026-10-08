# 本项目 RT-DETR 红外小目标检测模型技术文档

> 2026-10-03路径已更新。本文涉及的自包含提交包为历史独立导出包，不是本工程内的子目录；训练/测试请以根README和六个入口为准。

## 1 文档结论

本项目实际采用的是一套面向 `1280 × 1024` 单帧红外图像的单类别 RT-DETR-R18 检测器。模型只判断“是否存在目标”和目标矩形框位置，不输出车辆、人员等语义类别。所有训练数据中的类别均折叠为统一的 `target` 类。

基础提交模型由以下模块组成：

```text
1280×1024 RGB 图像
    ↓
PResNet18-vd 主干网络
    ↓  stride 8/16/32，通道 128/256/512
HybridEncoder
    ↓  统一 256 通道，AIFI + FPN + PAN
RT-DETR Transformer Decoder
    ↓  100 queries，3 层 decoder，多尺度可变形注意力
RTDETRPostProcessor
    ↓
目标框 xyxy + confidence，不输出类别名称
```

模型共 `20,073,172` 个参数。自包含提交目录中的 `model.py` 与 `best.pt` 均包含 510 个状态张量，已通过 `strict=True` 完成 `510/510` 严格匹配。

基础权重在其 VEDAI 真实图像与传统合成图像体系的 79 张独立测试图上得到：

| 指标 | 结果 |
|---|---:|
| 目标数 | 179 |
| TP / FP / FN | 168 / 432 / 11 |
| Pd | 93.85% |
| Precision | 28.00% |
| FAR | 72.00% |
| F1 | 43.13% |
| GTX 1650 Ti 全流程均值 | 168.98 ms/帧 |

该结果说明基础权重召回较高，但虚警明显偏高；本机速度也不能代表 Jetson AGX Orin 的 TensorRT 正式速度。当前模型尚不能据此判定满足比赛要求。

## 2 开源来源

### 2.1 官方 RT-DETR

本项目源码主体来自 [lyuwenyu/RT-DETR 官方 GitHub 仓库](https://github.com/lyuwenyu/RT-DETR)。该仓库是论文 [DETRs Beat YOLOs on Real-time Object Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html) 的官方实现，提供 Paddle 和 PyTorch 版本。

官方公开信息包括：

- RT-DETR 使用高效混合编码器和 IoU-aware query selection 实现实时端到端检测。
- 官方 RT-DETR-R18 在 `640 × 640` 输入下约为 20M 参数。
- 官方表格给出的 T4 TensorRT FP16 速度为 217 FPS。
- 官方 PyTorch 版本支持自定义数据训练。
- 官方仓库给出 ONNX Runtime、TensorRT 和 OpenVINO 等部署方向。
- 官方仓库后来增加了 sliced inference，用于改善小目标检测。

官方速度不能直接作为本项目速度。本项目输入为 `1280 × 1024`，像素数约是 `640 × 640` 的 3.2 倍；目标平台还是 Jetson AGX Orin，并要求三类模型同时运行。

### 2.2 红外小目标 RT-DETR 公开改进参考

[SF-DETR](https://github.com/Jim975231/SF-DETR) 是一个直接面向复杂背景红外小目标的 RT-DETR 改进项目，公开了三个主要模块：

| 模块 | 公开源码 | 功能 |
|---|---|---|
| SFSCBlock | [SFSCBlock.py](https://github.com/Jim975231/SF-DETR/blob/main/ultralytics/nn/IRSTD_modules/SFSCBlock.py) | Scharr 边缘、频域门控和空间分支联合增强 |
| SRAIM | [SRAIM.py](https://github.com/Jim975231/SF-DETR/blob/main/ultralytics/nn/IRSTD_modules/SRAIM.py) | 频谱残差异常显著性与 Transformer 尺度内交互 |
| CFARepC3 | [CFARepC3.py](https://github.com/Jim975231/SF-DETR/blob/main/ultralytics/nn/IRSTD_modules/CFARepC3.py) | 轻量频率细节注入与坐标注意力跨尺度融合 |

这些模块目前**没有并入本项目的 `best.pt`**。本项目当前权重仍是标准 RT-DETR-R18 单类别定制版。SF-DETR 只能作为后续优化方向，不能把其论文或仓库结果写成本项目已取得的结果。

## 3 比赛任务与模型边界

### 3.1 输入输出

| 项目 | 本项目定义 |
|---|---|
| 输入图像 | `1280 × 1024`，转为三通道 RGB |
| 模型输入 | `1 × 3 × 1024 × 1280`，`float32`，像素归一化到 `[0,1]` |
| 标注来源 | VOC XML 矩形框 |
| 内部训练标注 | COCO JSON，所有类别折叠为 `category_id=0` |
| 输出 | `x1, y1, x2, y2, confidence` |
| 类别输出 | 不输出具体类别 |
| 正确检测条件 | 预测框和未匹配真值框的 `IoU ≥ 0.40` |

不要求类别识别并不等于不需要标注。模型仍需利用 XML 矩形框学习位置，只是所有目标共享一个前景类别。

### 3.2 比赛指标

项目实现采用置信度降序的一对一匹配：一个预测只能匹配一个真值，一个真值也只能被一个预测匹配。

```text
Pd        = TP / (TP + FN)
Precision = TP / (TP + FP)
FAR       = FP / (TP + FP)
F1        = 2 × Pd × Precision / (Pd + Precision)
```

内部目标是 `Pd ≥ 90%`，同时尽量降低 FAR。正式设备要求是在 Jetson AGX Orin 64GB 上三类模型同时运行时，全流程单帧时间小于 45 ms。

## 4 本项目模型总体结构

模型实例在 [`InfraredSmallTargetDetection_basic_item_submission/model.py`](../InfraredSmallTargetDetection_basic_item_submission/model.py) 的 `build_model()` 中构建：

```python
backbone = PResNet(
    depth=18,
    variant="d",
    return_idx=[1, 2, 3],
    freeze_at=-1,
    freeze_norm=True,
    pretrained=False,
)

encoder = HybridEncoder(
    in_channels=[128, 256, 512],
    feat_strides=[8, 16, 32],
    hidden_dim=256,
    nhead=8,
    dim_feedforward=1024,
    use_encoder_idx=[2],
    num_encoder_layers=1,
    expansion=0.5,
    eval_spatial_size=[1024, 1280],
)

decoder = RTDETRTransformer(
    num_classes=1,
    hidden_dim=256,
    num_queries=100,
    feat_channels=[256, 256, 256],
    feat_strides=[8, 16, 32],
    num_levels=3,
    num_decoder_layers=3,
    num_denoising=100,
    eval_spatial_size=[1024, 1280],
)
```

### 4.1 与官方常用配置的区别

| 参数 | 官方常用 RT-DETR 配置 | 本项目 |
|---|---:|---:|
| Backbone | R50 等 | R18 |
| 输入 | 640 × 640 | 1280 × 1024 |
| 类别数 | 80 | 1 |
| Queries | 300 | 100 |
| Decoder 层 | 6 | 3 |
| HybridEncoder expansion | 1.0 | 0.5 |
| 输出候选数 | 300 | 100 |

减小 queries、decoder 层数和 encoder expansion，是为了控制高分辨率输入下的计算量。比赛场景目标通常稀疏，100 queries 足以覆盖单帧目标数量，但在极端密集场景下仍需重新验证。

## 5 PResNet18-vd 主干网络

PResNet18-vd 使用 BasicBlock，四个阶段的块数为 `[2, 2, 2, 2]`。`variant='d'` 使用三个 `3 × 3` 卷积作为 stem，并在部分降采样捷径中采用平均池化加 `1 × 1` 卷积。

`return_idx=[1,2,3]` 返回三层特征：

| 特征层 | 步长 | 通道数 | 1280 × 1024 输入下的空间尺寸 |
|---|---:|---:|---:|
| P3 | 8 | 128 | 160 × 128 |
| P4 | 16 | 256 | 80 × 64 |
| P5 | 32 | 512 | 40 × 32 |

对于约 `20 × 12` 像素的小目标，映射到 stride 8 特征后约占 `2.5 × 1.5` 个单元。目标非常容易在下采样中丢失，这也是项目后来单独实验 P2-Lite 高分辨率分支的原因。

`freeze_norm=True` 固定 BatchNorm 统计量，可以减轻小 batch 训练时统计不稳定。基础训练设置 `freeze_at=-1`，主干可训练；少样本迁移和类增量脚本可单独冻结主干，仅更新 encoder 和 decoder。

## 6 HybridEncoder

### 6.1 尺度内交互

三路主干特征先通过 `1 × 1` 卷积统一到 256 通道。本项目只对 `use_encoder_idx=[2]`，即 stride 32 的最高层特征执行一层 TransformerEncoder。

该层参数为：

- hidden dimension：256；
- attention heads：8；
- feed-forward dimension：1024；
- activation：GELU；
- dropout：0；
- position encoding：二维正余弦位置编码。

只在最高层做注意力可以保留全局建模能力，同时避免在高分辨率特征上执行完整自注意力造成高开销。

### 6.2 跨尺度融合

尺度内交互后，HybridEncoder 依次执行：

1. 自顶向下 FPN：高层语义上采样，与低层细节拼接；
2. CSPRepLayer 融合；
3. 自底向上 PAN：低层特征降采样，再与高层融合；
4. 输出三层 256 通道特征。

本项目将 `expansion` 从常见的 1.0 调为 0.5，以降低融合块内部通道和计算量。

核心实现位于：

- [`engines/rtdetr/src/zoo/rtdetr/hybrid_encoder.py`](../engines/rtdetr/src/zoo/rtdetr/hybrid_encoder.py)
- 自包含版本位于 [`model.py`](../InfraredSmallTargetDetection_basic_item_submission/model.py)

## 7 RT-DETR Transformer Decoder

### 7.1 IoU-aware query selection

编码器输出经过分类头和边界框头后，模型选择得分最高的 100 个位置作为 decoder 查询的初始参考框。它不是完全依赖固定 query embedding，而是从当前图像内容中挑选候选区域。

### 7.2 多尺度可变形注意力

每层 decoder 包括：

1. query 自注意力；
2. 多尺度可变形交叉注意力；
3. 前馈网络；
4. 分类头和边界框回归头；
5. 逐层迭代更新参考框。

可变形注意力只在每个尺度的少量采样点上聚合特征，比对所有特征位置执行全局交叉注意力更快。

### 7.3 去噪训练

训练时模型额外创建 100 个 denoising queries，对真值类别和框加入扰动，并通过 attention mask 隔离不同去噪组。这样可以为 decoder 提供更稳定的正样本学习信号。推理时不创建去噪查询，因此不会增加正式输出数量。

实现文件：

- [`rtdetr_decoder.py`](../engines/rtdetr/src/zoo/rtdetr/rtdetr_decoder.py)
- [`denoising.py`](../engines/rtdetr/src/zoo/rtdetr/denoising.py)

## 8 匹配与损失函数

### 8.1 HungarianMatcher

模型的 100 个预测 query 与真值框进行 Hungarian 全局一对一匹配。匹配代价为：

```text
Cost = 2 × Cost_class + 5 × Cost_bbox + 2 × Cost_giou
```

实现位于 [`matcher.py`](../engines/rtdetr/src/zoo/rtdetr/matcher.py)。

### 8.2 SetCriterion

| 损失 | 权重 | 作用 |
|---|---:|---|
| Varifocal Loss | 1 | 学习目标存在概率和定位质量，处理大量背景 query |
| L1 Box Loss | 5 | 约束归一化中心坐标和宽高误差 |
| GIoU Loss | 2 | 提供框几何重叠和相对位置优化信号 |
| Auxiliary Loss | 同主损失 | 监督中间 decoder 层 |
| Denoising Loss | 同主损失 | 监督带噪真值查询 |

实现位于 [`rtdetr_criterion.py`](../engines/rtdetr/src/zoo/rtdetr/rtdetr_criterion.py)。

## 9 数据与训练配置

### 9.1 基础权重数据来源

基础权重 `weights/rtdetr/vedai_real_synth_best.pth` 使用 VEDAI 真实红外图像和传统方式生成的合成图像训练。独立域整理中记录的 D0 划分为：

| 划分 | 图像数 |
|---|---:|
| Train | 891 |
| Validation | 80 |
| Test | 79 |

该权重不包含后来加入的 GPT 图、HIT-UAV 迁移数据或类增量阶段数据。

### 9.2 基础训练超参数

| 参数 | 设置 |
|---|---|
| Epochs | 72 |
| Batch size | 16 |
| Optimizer | AdamW |
| 主学习率 | `1e-4` |
| Backbone 学习率 | `1e-5` |
| Weight decay | `1e-4` |
| Gradient clipping | `0.1` |
| LR scheduler | MultiStepLR，milestones 60/68，gamma 0.1 |
| AMP | 启用 |
| EMA | 启用 |
| 数据增强 | RandomHorizontalFlip `p=0.5` |
| 输入尺寸 | 固定 `1024 × 1280` |

提交目录 `best.pt` 的元数据记录其来自训练 checkpoint 中的 `ema.module`，选中状态对应 epoch 14，用于推理。

### 9.3 训练入口

[`training/train_rtdetr.py`](../training/train_rtdetr.py) 负责训练：

- 当前基本项默认随机初始化；
- 只有显式传入 `--initial-weight` 才加载外部权重；
- 若输出目录存在 `checkpoint.pth`，默认断点续训；
- 完成后要求存在 `best.pth`。

示例：

```powershell
cd D:\InfraredSmallTargetDetection
python training/train_rtdetr.py --profile base --gpu 0 --epochs 72 --batch-size 16
```

## 10 推理与后处理

### 10.1 自包含推理包

目录 `InfraredSmallTargetDetection_basic_item_submission` 包含：

| 文件 | 作用 |
|---|---|
| `model.py` | 完整模型结构，不依赖 `engines/rtdetr/src` |
| `best.pt` | 约 80.6 MB 的 EMA 最优权重 |
| `infer.py` | 图像读取、推理、输出和可选 VOC 指标计算 |

严格一致性检查：

```powershell
cd D:\InfraredSmallTargetDetection\InfraredSmallTargetDetection_basic_item_submission
python model.py --weight best.pt --device cpu
```

预期结果：

```text
missing_keys = 0
unexpected_keys = 0
All 510 parameter/buffer shapes match
```

### 10.2 推理流程

[`infer.py`](../InfraredSmallTargetDetection_basic_item_submission/infer.py) 的流程为：

1. 检查图像是否为 `1280 × 1024`；
2. 转换为 RGB；
3. 转为 `NCHW float32` 并除以 255；
4. 模型产生分类 logits 和归一化边界框；
5. 后处理映射为原图 `xyxy`；
6. 使用固定 confidence 过滤；
7. 输出 `predictions.csv`；
8. 提供 XML 时计算 `metrics.json`。

默认推理命令：

```powershell
python infer.py --input test_set\images --xml-dir test_set\xml --device cuda:0
```

### 10.3 Confidence 与 NMS

基础 `infer.py` 的固定阈值为：

```text
confidence = 0.08029208332300186
match IoU = 0.40
```

基础提交推理不执行额外 NMS，只对 RT-DETR 的 top-100 候选做置信度筛选。项目通用实验另外提供 [`competition/postprocess.py`](../competition/postprocess.py)，在验证集上联合搜索 confidence 与 NMS IoU，用于处理跨域时的重复框和虚警。

工作点选择规则为：

1. 优先找到验证集 `Pd ≥ 0.90` 的点；
2. 在可行点中最小化 FAR；
3. 再比较 Pd、FP 和 confidence；
4. 固定参数后只测试一次 test。

不能根据测试集结果反复调整 confidence 或 NMS，否则会形成测试集泄漏。

## 11 当前实验结果解释

### 11.1 基础域测试

基础自包含模型在 VEDAI 测试集上的结果：

```json
{
  "images": 79,
  "targets": 179,
  "tp": 168,
  "fp": 432,
  "fn": 11,
  "pd": 0.9385474860,
  "precision": 0.28,
  "false_alarm_rate": 0.72,
  "f1": 0.4313222080,
  "mean_pipeline_ms": 168.9817
}
```

结果说明：

- Pd 已超过 90%；
- 低阈值保留了大量候选，导致 FAR 高达 72%；
- 当前主要矛盾是虚警，而不是基础域召回；
- 速度来自 GTX 1650 Ti 的 PyTorch 全流程，不能用来判断 Orin TensorRT 是否达标。

### 11.2 跨域问题

VEDAI、HIT-UAV、GPT 低亮图和真实比赛数据可能在以下方面不同：

- 红外极性和目标亮度；
- 局部目标背景对比度；
- 道路比例与纹理；
- 飞行高度和视角；
- 目标框大小与长宽比；
- 图像噪声、模糊和热成像响应；
- 负样本中的石块、路标、灌木和高亮纹理。

因此，在源域上 Pd 高并不等于比赛域效果高。继续混入大量风格差异较大的数据，也可能同时抬高 FN 和 FP。

## 12 Jetson AGX Orin 部署

### 12.1 导出流程

[`tools/export_rtdetr_onnx.py`](../tools/export_rtdetr_onnx.py) 将模型导出为固定输入 ONNX：

```text
input images:         1 × 3 × 1024 × 1280 float32
input original_sizes: 1 × 2 int64
outputs:              labels, boxes, scores
opset:                16
```

脚本将 `scaled_dot_product_attention` 替换为等价的 `MatMul → Softmax → MatMul`，用于绕过 PyTorch 2.0 旧 ONNX 导出器的算子限制。

推荐部署顺序：

1. 用 `model.py + best.pt` 做严格权重检查；
2. 导出固定尺寸 ONNX；
3. 使用 ONNX checker 检查；
4. 在 Jetson 上构建 TensorRT FP16 engine；
5. batch 固定为 1；
6. 预分配 CUDA 输入输出；
7. 预热后分别测模型时间和全流程时间；
8. 三个比赛模型同时运行时重新测量；
9. 在同一推理程序中验证 Pd 和 FAR。

INT8 只能在具备代表性比赛域校准集时尝试。红外弱小目标对量化误差敏感，量化后必须重新评估 Pd 和 FAR。

### 12.2 45 ms 时延风险

全流程时间应包括：

- 图像接收或磁盘读取；
- 解码与 RGB 转换；
- CPU 到 GPU 拷贝；
- TensorRT 推理；
- 后处理和可选 NMS；
- 输出格式转换。

只报告 GPU kernel 时间不满足比赛定义。三个模型并发时还会竞争 GPU 算力、显存带宽、CPU 和数据传输资源。

## 13 代码文件映射

| 文件 | 作用 |
|---|---|
| [`InfraredSmallTargetDetection_basic_item_submission/model.py`](../InfraredSmallTargetDetection_basic_item_submission/model.py) | 自包含 RT-DETR-R18 模型结构 |
| [`InfraredSmallTargetDetection_basic_item_submission/infer.py`](../InfraredSmallTargetDetection_basic_item_submission/infer.py) | 提交推理程序 |
| [`competition/rtdetr_runtime.py`](../competition/rtdetr_runtime.py) | 生成模型与数据配置，执行统一推理 |
| [`competition/metrics.py`](../competition/metrics.py) | 类别无关 IoU=0.40 一对一评测 |
| [`engines/rtdetr/src/nn/backbone/presnet.py`](../engines/rtdetr/src/nn/backbone/presnet.py) | PResNet 主干 |
| [`engines/rtdetr/src/zoo/rtdetr/hybrid_encoder.py`](../engines/rtdetr/src/zoo/rtdetr/hybrid_encoder.py) | HybridEncoder |
| [`engines/rtdetr/src/zoo/rtdetr/rtdetr_decoder.py`](../engines/rtdetr/src/zoo/rtdetr/rtdetr_decoder.py) | Transformer decoder |
| [`engines/rtdetr/src/zoo/rtdetr/rtdetr_criterion.py`](../engines/rtdetr/src/zoo/rtdetr/rtdetr_criterion.py) | VFL、L1、GIoU 和辅助损失 |
| [`engines/rtdetr/src/zoo/rtdetr/matcher.py`](../engines/rtdetr/src/zoo/rtdetr/matcher.py) | HungarianMatcher |
| [`training/train_rtdetr.py`](../training/train_rtdetr.py) | 通用训练和断点续训入口 |
| [`evaluation/test_rtdetr.py`](../evaluation/test_rtdetr.py) | 验证选阈值后的测试入口 |
| [`competition/postprocess.py`](../competition/postprocess.py) | confidence 和 NMS 搜索 |
| [`tools/export_rtdetr_onnx.py`](../tools/export_rtdetr_onnx.py) | 固定形状 ONNX 导出 |

## 14 主要风险与优化建议

| 问题 | 当前证据 | 优先措施 |
|---|---|---|
| 小目标下采样损失 | 20×12 目标在 stride 8 上仅约 2.5×1.5 单元 | 对比标准模型与已有 P2-Lite；必要时局部切片 |
| FAR 高 | 基础域 FAR 72% | 增加真实困难负样本；验证集联合选择 confidence/NMS |
| 域差距 | 不同代理域结果波动大 | 数据按成像域分层；少量现场样本迁移 |
| 弱对比目标召回 | 暗目标更容易成为 FN | 按局部对比度分组评测；消融红外增强模块 |
| TensorRT 兼容性 | 频域和注意力可能引入不支持算子 | 每次只加入一个模块并先验证 ONNX/TensorRT |
| 三模型并发时延 | 单模型 5090 或 1650 Ti 速度不可直接使用 | 必须在 Orin 三模型同时运行时测全流程 |

推荐顺序：

1. 先完成 FP/FN 的目标尺寸、局部对比度、道路区域和背景类型分组分析；
2. 优先优化数据质量、真实困难负样本以及验证集工作点；
3. 对比标准 R18 与现有 P2-Lite，不先堆叠多个新模块；
4. 若低对比目标仍是主要 FN，再单独移植 SRAIM 或 CFARepC3 做消融；
5. 每个候选模型同时检查 Pd、FAR、ONNX 导出和 Orin FP16 时延；
6. 最终只保留满足接口和时延要求的方案。

不建议一次性完整移植 SF-DETR 三个模块。这样无法判断指标变化来自哪个模块，而且 `torch.fft` 对 ONNX/TensorRT 的兼容性和高分辨率开销需要单独验证。

## 15 复现命令

### 15.1 环境检查

```powershell
cd D:\InfraredSmallTargetDetection
python tools/check_environment.py --require-cuda
```

### 15.2 基础推理

```powershell
cd D:\InfraredSmallTargetDetection\InfraredSmallTargetDetection_basic_item_submission
python infer.py --input test_set\images --xml-dir test_set\xml --device cuda:0
```

### 15.3 通用训练

```powershell
cd D:\InfraredSmallTargetDetection
python training/train_rtdetr.py --profile base --gpu 0 --epochs 72 --batch-size 16
```

服务器使用 GPU6 时：

```bash
python training/train_rtdetr.py --profile base --gpu 6 --epochs 72 --batch-size 16
```

### 15.4 测试

```powershell
python evaluation/test_rtdetr.py --profile base --gpu 0
```

### 15.5 ONNX 导出

```powershell
python tools/export_rtdetr_onnx.py --profile base --gpu 0
```

## 16 文档适用范围

本文档描述的是当前项目基础提交模型：

```text
RT-DETR-R18
单类别 target
1280 × 1024
100 queries
3 decoder layers
PResNet18-vd + HybridEncoder
VEDAI 真实图像 + 传统合成图像权重
```

普通迁移、FOMAML、类增量、HIT-UAV、GPT 低亮图、P2-Lite V4 和 SPIRE-Box 属于项目中的独立实验线，不应混入本基础模型结构或权重说明。若提交权重发生变化，应重新执行严格状态字典匹配，并同步更新数据来源、阈值、测试结果和部署文件。

