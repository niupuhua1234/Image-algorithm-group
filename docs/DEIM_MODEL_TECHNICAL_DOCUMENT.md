# DEIM R18 车辆检测模型 · 技术文档

| 项 | 内容 |
|---|---|
| 文档版本 | v1.0 |
| 模型版本 | **car_union v4 / stage1 best**(`best_stg1.pth`) |
| 训练日期 | 2026-09-10 ~ 09-12 |
| 权重日期 | 2026-09-11 19:12(epoch 31) |
| 适用场景 | 无人机 / 航空影像车辆检测(可见光 RGB) |
| 实测平台 | RTX 5080 16GB · torch 2.11.0+cu128 · TensorRT 11.2.1.2 |

---

## 1. 概述

### 1.1 模型定位

本模型是 **DEIM(DETR with Improved Matching)** 框架下的 RT-DETRv2 变体,采用 **ResNet-18** 轻量骨干,在自建 **CAR-UNION 6 类**车辆数据集上从 COCO 预训练权重微调而来。

**选择 DEIM R18 的理由**:在本项目的横向对比中(VEDAI 官方 9 类),DEIM R18 达到 **96.99%**,显著高于 RT-DETR-L(91.78%)与 YOLO26m(90.14%),而参数量仅 **19.9M**(不到 RT-DETR-L 40M+ 的一半)。

### 1.2 核心指标速览

| 指标 | 数值 |
|---|---|
| 参数量 | **20,088,424**(20.09 M) |
| 模型权重体积 | 322 MB(FP32,含 EMA + optimizer 状态) |
| 类别数 | **6**:car / truck / bus / van / pickup / tank |
| 训练输入 | 960 × 960 |
| 推理输入 | 1024 × 1024(默认) |
| **验收准确率** | **90.36%**(50939 / 56375)@conf 0.005 |
| 验证集 mAP@.50:.95 | 0.4048(best epoch 31) |
| 验证集 mAP@.50 | 0.6693 |
| 单块时延(PyTorch @1024) | **14.7 ms** |
| 单块时延(TensorRT @1024) | **5.57 ms** |
| 峰值显存 @1024 | 271 MB |

---

## 2. 模型架构

### 2.1 整体结构

```
输入图像 [B, 3, H, W]  (RGB, 0~1)
    │
    ├─▶ Backbone: PResNet-18-d          → 多尺度特征 [stride 8 / 16 / 32]
    │                                    通道数 128 / 256 / 512
    │
    ├─▶ Encoder: HybridEncoder          → 256 通道的融合特征(3 个尺度)
    │      (AIFI 自注意力 + CCFM 跨尺度融合)
    │
    └─▶ Decoder: RTDETRTransformerv2    → 300 个查询的迭代细化
           (3 层, 可变形注意力 + 去噪训练)
              │
              ├─▶ pred_logits [B, 300, 6]    分类得分(未过 sigmoid)
              └─▶ pred_boxes  [B, 300, 4]    归一化 cxcywh
                     │
                     └─▶ PostProcessor → top-300 → 最终检测结果
```

### 2.2 Backbone:PResNet-18-d

| 参数 | 值 | 说明 |
|---|---|---|
| depth | 18 | ResNet-18 |
| **return_idx** | **[1, 2, 3]** | 输出 stage 2/3/4 的特征 |
| 输出步长 | **stride 8 / 16 / 32** | 对应 3 个尺度 |
| 输出通道 | **128 / 256 / 512** | |
| freeze_at | **-1**(不冻结) | v4 全程全参数训练 |
| freeze_norm | **False** | BN 参与训练 |
| 预训练 | 官方 COCO 120ep 权重 | |

**参数量**:11,199,968(11.2 M,占总量 55.8%)

> `-d` 后缀表示该变体使用 Deep stem(前两层 stride 调整),用于在保持轻量的同时提升小目标特征的保留能力。

### 2.3 Encoder:HybridEncoder

| 参数 | 值 |
|---|---|
| in_channels | [128, 256, 512] |
| **hidden_dim** | **256** |
| **use_encoder_idx** | **[2]**(仅对 stride-32 层做自注意力) |
| num_encoder_layers | 1 |
| nhead | 8 |
| dim_feedforward | 1024 |
| dropout | 0.0 |
| **expansion** | **0.5**(CCFM 中的通道压缩比) |

**结构**:AIFI(尺度内自注意力,只作用于 stride-32)+ CCFM(跨尺度特征融合,PAN 式双向路径)。

**参数量**:4,965,120(4.97 M,占 24.7%)

**设计意图**:只在最深层做自注意力(计算量最小、感受野最大),浅层保持卷积以保留小目标的位置精度。

### 2.4 Decoder:RTDETRTransformerv2

| 参数 | 值 | 说明 |
|---|---|---|
| **num_layers** | **3** | 解码器层数(R18 变体用 3 层,非 R50 的 6 层) |
| **num_queries** | **300** | 每张图最多 300 个检测 |
| **num_denoising** | **100** | 去噪训练查询组数 |
| num_levels | 3 | 使用 3 个尺度的特征 |
| **num_points** | **[4, 4, 4]** | 可变形注意力每层采样点数 |
| hidden_dim | 256 | |
| **query_pos_method** | **as_reg** | 查询位置编码由回归分支生成 |
| activation / mlp_act | silu | |
| **eval_idx** | **2** | 推理时使用第 3 层输出 |

**参数量**:3,923,336(3.92 M,占 19.5%)

### 2.5 参数量分布

| 模块 | 参数量 | 占比 |
|---|---|---|
| Backbone(PResNet-18-d) | 11,199,968 | 55.8% |
| Encoder(HybridEncoder) | 4,965,120 | 24.7% |
| Decoder(RTDETRTransformerv2) | 3,923,336 | 19.5% |
| **合计** | **20,088,424** | 100% |

### 2.6 DEIM 的两个关键改进

**① Dense O2O(Dense One-to-One)**

传统 DETR 系用一对一匹配,一张图只有一条监督信号,收敛慢。DEIM 通过 **Mosaic / MixUp 人为增加每图目标数**,让同一张图产生更多匹配对,从而**在不增加推理成本的前提下加速收敛**。

本模型配置:`Mosaic probability=1.0`、`MixUp prob=0.5`(epoch 4~24)。

**② MAL(Matchability-Aware Loss)**

用匹配质量(而非仅分类正确性)加权分类损失,解决"低质量匹配的框被过度惩罚"的问题。

本模型损失权重:

| 损失项 | 权重 |
|---|---|
| `loss_mal`(分类) | 1 |
| `loss_bbox`(L1 回归) | 5 |
| `loss_giou`(GIoU) | 2 |
| gamma | 1.5 |

---

## 3. 输入输出规格

### 3.1 输入

| 项 | 规格 |
|---|---|
| 形状 | `[B, 3, H, W]`,推理默认 `[1, 3, 1024, 1024]` |
| 数据类型 | `torch.float32` |
| 通道顺序 | **RGB**(非 BGR) |
| **值域** | **0.0 ~ 1.0**(已除以 255,**不做** ImageNet 均值方差归一化) |
| 尺寸要求 | 任意,但建议 32 的倍数;训练用 960,推理默认 1024 |
| 预处理 | 直接 `resize` 到目标尺寸(双线性),不保持长宽比 |

**预处理代码**:

```python
import numpy as np
import torch
from PIL import Image

def preprocess(image_path, imgsz=1024):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((imgsz, imgsz), Image.BILINEAR)
    x = np.asarray(img).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return torch.from_numpy(x).cuda(), img.size  # img.size = 原始尺寸
```

### 3.2 输出

模型 forward 返回 **dict**:

| 键 | 形状 | 说明 |
|---|---|---|
| `pred_logits` | `[B, 300, 6]` | 分类 logits(**未过 sigmoid**) |
| `pred_boxes` | `[B, 300, 4]` | **归一化** `cxcywh`(值域 0~1) |
| 其他键 | — | 训练时的 aux/denoising 输出,推理时忽略 |

### 3.3 坐标系统

```
pred_boxes (归一化 cxcywh)  ──box_convert──▶  xyxy (归一化)
                                            │
                                            × imgsz
                                            ▼
                                     像素坐标 (推理空间)
                                            │
                                            ÷ scale        其中 scale = imgsz / max(W, H)
                                            ▼
                                      原始图像素坐标
```

**注意**:推理时若图像被 resize 过,预测框需**反向缩放**回原图尺寸。

---

## 4. 后处理流程

DEIM 输出的是 300 个查询的原始预测,**不需要 NMS**(一对一匹配天然无重复),只需阈值过滤。

```python
import torch
import torchvision

def postprocess(out, imgsz=1024, conf=0.005, num_classes=6):
    """DEIM 后处理: sigmoid → topk → 阈值过滤 → 坐标转换"""
    # 1) 分类得分
    scores = torch.sigmoid(out['pred_logits'])           # [1, 300, 6]

    # 2) 取 score 最高的 300 个 (query, class) 组合
    scores, index = torch.topk(scores.flatten(1), 300, dim=-1)
    labels = index % num_classes                          # 类别
    index = index // num_classes                          # 查询索引

    # 3) 坐标: 归一化 cxcywh → 像素 xyxy
    boxes = torchvision.ops.box_convert(out['pred_boxes'], 'cxcywh', 'xyxy')
    boxes = boxes * imgsz                                 # 缩放到像素
    boxes = boxes.gather(1, index.unsqueeze(-1).repeat(1, 1, 4))[0]

    # 4) 置信度过滤
    keep = scores[0] >= conf
    return labels[0][keep], boxes[keep], scores[0][keep]
```

**关键参数**:

| 参数 | 说明 |
|---|---|
| `conf` | 置信度阈值。**验收口径用 0.005**;工业场景建议 0.20~0.30 |
| `num_classes` | 6(本模型) |
| top-k | 300(等于 `num_queries`,即全部查询) |

---

## 5. 训练配置

### 5.1 数据集:CAR-UNION v4

| 划分 | 切片块数 | GT 目标数 |
|---|---|---|
| train | 37,847 | 532,429 |
| val | 8,727 | — |
| **test** | **3,976** | **56,375** |

**切片规则**:1920×1080 原图 → 1280×1280 块,x 方向 stride 640(50% 重叠),每图 2 块。

**6 类**:car / truck / bus / van / pickup / tank

> 注:数据来源为多数据集合并(VEDAI / COWC / DOTA 等),已做去重与泄漏核验。tank 类为稀有类,采样时 100% 保留。

### 5.2 超参数

| 类别 | 参数 | 值 |
|---|---|---|
| **优化器** | 类型 | AdamW |
| | **lr** | **2e-4** |
| | betas | [0.9, 0.999] |
| | weight_decay | 1e-4(norm/bn 层为 0) |
| **LR 调度** | 类型 | **flatcosine** |
| | warmup_iter | 2000 步 |
| | lr_gamma | 0.5(lr 下限 = 1e-4) |
| | **flat_epoch** | **24** |
| | no_aug_epoch | 2 |
| **训练** | **epoches** | **45** |
| | total_batch_size | 8 |
| | 输入尺寸 | 960 × 960 |
| | num_workers | 6 |
| | AMP | 启用(`--use-amp`) |
| | 梯度裁剪 | `clip_max_norm`(默认配置) |
| | stop_epoch | 45(全程 stage1) |
| **验证** | batch | 32 |

**LR 曲线形态**(45 epoch = 14,940 步):

```
步数:  0 ──2000──▶ 2000 ─────12,000─────▶ 12,000 ──2000──▶ 14,000 ──940──▶ 14,940
       warmup        flat(恒定 2e-4)        cosine 衰减到 1e-4      no_aug(1e-4)
       0 → 2e-4      2e-4                                            无增强
```

### 5.3 数据增强

**增强策略切换点**:`policy: epoch [4, 24, 43]`(四阶段)

| 阶段 | epoch 区间 | 启用增强 |
|---|---|---|
| 1 | 0 ~ 3 | 仅基础(Resize + HFlip) |
| 2 | 4 ~ 23 | **Mosaic + PhotometricDistort + ZoomOut + IoUCrop** |
| 3 | 24 ~ 42 | 关闭 Mosaic,保留 ZoomOut + IoUCrop |
| 4 | 43 ~ 44 | 全部关闭(no_aug) |

**各增强参数**:

| 增强 | 参数 |
|---|---|
| **Mosaic** | output_size **320**,scaling_range **[0.5, 1.5]**,rotation 10°,probability **1.0** |
| MixUp | probability 0.5,epochs [4, 24] |
| RandomPhotometricDistort | p = 0.5 |
| RandomZoomOut | fill = 0 |
| **RandomIoUCrop** | **p = 0.8** |
| RandomHorizontalFlip | 启用 |
| Resize | 960 × 960 |

> ⚠️ **已知问题**:`Mosaic output_size=320` 意味着 4 张 960 图先被压到 320 画布再放大,**小目标细节会损失**(实测在 DOTA 19px 目标上影响显著)。若你的场景以小目标为主,建议调到 640 并降低概率。

### 5.4 训练结果

| 指标 | 数值 |
|---|---|
| 总 epoch | 45 |
| **最优 epoch** | **31** |
| val mAP@.50:.95 | **0.4048** |
| val mAP@.50 | **0.6693** |
| 训练耗时 | 约 35.6 小时(RTX 5080) |

---

## 6. 性能实测

### 6.1 精度(test 集 3,976 块 / 56,375 GT)

| 类别 | 正确 / GT | 准确率 |
|---|---|---|
| car | 42,407 / 47,453 | **89.37%** |
| truck | 6,161 / 6,519 | **94.51%** |
| bus | 1,072 / 1,080 | **99.26%** |
| van | 775 / 793 | **97.73%** |
| pickup | 484 / 490 | **98.78%** |
| tank | 40 / 40 | **100.00%** |
| **合计** | **50,939 / 56,375** | **90.36%** |

**conf 敏感性**:

| conf | 准确率 | 说明 |
|---|---|---|
| 0.001 | 90.36% | 与训练侧验收一致 |
| **0.005** | **90.36%** | 验收口径默认值 |
| 0.01 | 90.35% | |
| 0.05 | 89.91% | |
| 0.20 | 约 84% | 若关心误报率则用此档 |

### 6.2 时延与显存(FP32,batch=1,RTX 5080)

| 推理尺寸 | 平均时延 | 中位 | p95 | 峰值显存 |
|---|---|---|---|---|
| 640 | 11.97 ms | 11.88 | 14.23 | 174 MB |
| 800 | 12.66 ms | 12.67 | 14.36 | 222 MB |
| 960 | 13.54 ms | 13.59 | 15.02 | 249 MB |
| **1024**(默认) | **14.70 ms** | 14.63 | 16.42 | 271 MB |
| 1280 | 18.33 ms | 18.36 | 19.51 | 372 MB |
| 1408 | 19.70 ms | 19.60 | 21.29 | 435 MB |

**关键观察**:**分辨率从 640 提到 1408(面积 ×4.8),时延只从 12ms 涨到 19.7ms(+64%)**。说明耗时**主要不在骨干卷积**,而在 decoder 的 300 查询迭代与后处理等固定开销 —— 这也意味着**提高分辨率来检测小目标是划算的**。

**TensorRT 加速**(完全定形导出,batch=1):

| 实现 | 时延 | 加速比 |
|---|---|---|
| PyTorch @1024 | 14.70 ms | 1.0× |
| **TensorRT @1024** | **5.57 ms** | **2.6×** |

> TRT engine **与 GPU 架构绑定**,换机器必须重新导出。导出时必须**完全定形**(固定 batch 与 H/W、无 `dynamic_axes`),否则 Gather 无法常量折叠,会报 `Could not find any implementation for node ForeignNode[.../decoder/Gather_N]`;RT-DETR 的 dynamic engine 推理还会崩(`cudaError700`)。

### 6.3 精度-速度操作点建议

| 场景 | 推荐配置 | 准确率 | 单块时延 |
|---|---|---|---|
| **验收 / 离线批处理** | PT 或 TRT @1024,conf 0.005 | **90.36%** | 14.7 / 5.6 ms |
| 实时(<40ms/帧) | TRT @1024,**一帧 2 块** | 90.36% | **11.1 ms** |
| 抑制误报 | conf 0.20 ~ 0.30 | 约 84% ~ 87% | 同左 |

---

## 7. 部署指南

### 7.1 环境

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy Pillow
# 可选
pip install tensorrt   # 版本需与 CUDA 一致
```

实测组合:`torch 2.11.0+cu128` / `torchvision 0.26.0+cu128` / `TensorRT 11.2.1.2`

### 7.2 加载模型

```python
import os, sys
sys.path.insert(0, './deim')
os.chdir('./deim')

import patch_tw                      # ⚠️ 必需: torchvision 0.26 兼容补丁
patch_tw.apply()

import torch
from engine.core import YAMLConfig

cfg = YAMLConfig('configs/deim_rtdetrv2/deim_car_union_v4.yml')
model = cfg.model
model.encoder.eval_spatial_size = None      # ⚠️ 必需: 关闭固定输入尺寸假设
model.decoder.eval_spatial_size = None      # ⚠️ 同上
model = model.cuda().eval()

ck = torch.load('../weights/best_stg1.pth', map_location='cuda')
sd = ck['ema']['module'] if isinstance(ck.get('ema'), dict) and 'module' in ck['ema'] else ck.get('model', ck)
msd = model.state_dict()
model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
```

**三个必做项**:

| 步骤 | 原因 |
|---|---|
| `patch_tw.apply()` | torchvision 0.26 改了 `Transform.forward` 接口,DEIM 用旧式 `_transform`,不打补丁会 `NotImplementedError` |
| `eval_spatial_size = None` | 否则模型会按训练尺寸固定位置编码,输入尺寸不匹配时框会错位 |
| 优先用 `ema.module` | 训练用 EMA 权重,直接加载 `model` 会掉点 |

### 7.3 完整推理示例

```python
import numpy as np, torch, torchvision
from PIL import Image

IMGSZ, CONF, NC = 1024, 0.005, 6
NAMES = ['car', 'truck', 'bus', 'van', 'pickup', 'tank']

def detect(model, image_path):
    img = Image.open(image_path).convert('RGB')
    W, H = img.size
    scale = IMGSZ / max(W, H)
    inp = img.resize((IMGSZ, IMGSZ), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(inp).astype(np.float32).transpose(2, 0, 1)[None] / 255.0).cuda()

    with torch.no_grad():
        out = model(x)

    scores = torch.sigmoid(out['pred_logits'])
    scores, index = torch.topk(scores.flatten(1), 300, dim=-1)
    labels, qidx = index % NC, index // NC
    boxes = torchvision.ops.box_convert(out['pred_boxes'], 'cxcywh', 'xyxy')
    boxes = boxes * IMGSZ
    boxes = boxes.gather(1, qidx.unsqueeze(-1).repeat(1, 1, 4))[0]

    keep = scores[0] >= CONF
    b, l, s = boxes[keep], labels[0][keep], scores[0][keep]

    b = b / scale                              # 映射回原图坐标
    return [(NAMES[int(li)], bb.tolist(), float(si)) for bb, li, si in zip(b, l, s)]

for name, box, score in detect(model, 'test.jpg'):
    print(f'{name:8s} {score:.3f}  [{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]')
```

### 7.4 TensorRT 加速

```python
# 必须在目标机器上执行(engine 与 GPU 架构绑定)
x = torch.zeros((1, 3, 1024, 1024))          # 完全定形, 不要用 dynamic_axes
torch.onnx.export(model, (x,), 'model.onnx',
                  input_names=['images'],
                  output_names=['pred_logits', 'pred_boxes'],
                  opset_version=18, dynamo=False, do_constant_folding=True)
# 再用 trt.Builder 编译 -> 5.57 ms/块
```

---

## 8. 验收口径

### 8.1 指标定义(项目统一口径)

```
正确识别 = 预测类别与真实类别一致  且  IoU ≥ 0.40
准确率   = 正确识别数 / GT 目标总数        (recall 式计数)
```

**匹配方式**:**贪心匹配** —— 按预测框顺序,每个框找 IoU 最大的**未匹配** GT;命中且类别一致则记正确,每个 GT 只能被匹配一次。

### 8.2 与 mAP 的区别(重要)

| 维度 | mAP(COCO 标准) | 本验收口径 |
|---|---|---|
| 误检 | **惩罚**(降低 Precision) | **不惩罚** |
| 漏检 | 惩罚 | **惩罚**(分母是全部 GT) |
| 阈值 | IoU 0.5:0.95 平均 | **固定 IoU ≥ 0.40** |
| 用途 | 学术对比 | **交付验收** |

**实践后果**:本模型在 test 上的 FP(误检)达 **1,037,554**,Precision 仅 **4.68%**。这是 **300 查询 × 3976 块 ≈ 119 万候选框且阈值极低(0.005)** 的必然结果,**不是模型缺陷** —— 在该口径下低阈值只增加匹配机会。

**若你的场景关心误报**,请提高 conf:

| conf | Precision(参考) |
|---|---|
| 0.05 | 15.6% |
| 0.10 | 29.9% |
| 0.20 | 53.4% |
| 0.30 | 67.5% |
| 0.50 | 82.96% |

---

## 9. 已知限制

| 限制 | 说明 | 缓解方式 |
|---|---|---|
| **小目标能力受限于骨干 stride** | 目标短边 < 16px 时召回明显下降(在 DOTA 19px 目标上,整块推理仅 71.17%) | **滑窗切片推理**:把 1280 块切 640/853 窗口放大后推理,实测 **+16pp** |
| **全部 6 类在同一数据集上均衡** | 若目标域只有 car/truck,其余 4 类会灾难性遗忘(实测 van −14.88pp) | 多域联合训练 / 类别回放 |
| **误检量大** | 见 8.2 节,Precision 4.68% | 提高 conf 到 0.20~0.30 |
| **Mosaic output_size 偏小** | 320 会压缩小目标细节 | 训练时改 640 并降概率 |
| **依赖 DEIM 框架** | 不是标准 ONNX 即插即用,需带 `engine/` 代码 | 用本包内的 `export_trt.py` 导出 engine 后可脱离框架 |

---

## 10. 附录

### 10.1 文件清单

| 文件 | 说明 |
|---|---|
| `weights/best_stg1.pth` | 模型权重(EMA,322 MB) |
| `deim/configs/deim_rtdetrv2/deim_car_union_v4.yml` | 本模型训练/推理配置 |
| `deim/configs/deim_rtdetrv2/deim_r18vd_120e_coco.yml` | 父配置(R18 架构) |
| `deim/configs/base/rt_deim.yml` | DEIM 专属配置(损失、调度、增强) |
| `deim/configs/base/rtdetrv2_r50vd.yml` | 基线架构定义(queries/decoder) |
| `deim/engine/` | 模型与推理引擎 |
| `eval_deploy.py` | 验收脚本(路径自适应) |
| `export_trt.py` | TRT 导出脚本 |
| `test/images/` + `test/annotations.json` | 测试集(3,976 块 / 56,375 GT) |

### 10.2 权重内部结构

```python
{
  'date': '2026-09-11T19:12:20.965193',
  'last_epoch': 31,
  'model': {...},          # 538 个键, 原始模型权重
  'ema': {'module': {...}, 'updates': ...},   # ⚠️ 推理用这个
  'criterion': {}, 'postprocessor': {},
  'scaler': {...}, 'optimizer': {...}
}
```

**推理时取 `ema.module`** —— 实测该权重比 `model` 更稳(训练后期 EMA 平滑了噪声)。

### 10.3 版本与来源

| 项 | 值 |
|---|---|
| 框架 | DEIM(DETR with Improved Matching) |
| 基座 | RT-DETRv2 |
| 骨干 | PResNet-18-d(COCO 120ep 预训练) |
| 训练脚本 | `deim/train.py` |
| 权重来源 | `deim/output/deim_car_union_v4/best_stg1.pth` |
| 原评估脚本 | `scripts/eval_deim_car_union.py` |

---

*文档基于实测数据生成,所有指标均可在本部署包内用 `python eval_deploy.py` 复现。*
