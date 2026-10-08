# 红外可见光识别 · DEIM 车辆检测模型

本分支存放 **DEIM(DETR with Improved Matching)车辆检测模型**的完整代码、配置与实验报告,面向无人机 / 航空影像可见光 RGB 图像。

---

## 一、模型成绩

| 指标 | 数值 |
|---|---|
| 模型 | **DEIM R18**(PResNet-18-d + HybridEncoder + RTDETRTransformer v2) |
| 参数量 | **20.09 M** |
| 类别数 | **6**:car / truck / bus / van / pickup / tank |
| 数据集 | CAR-UNION v4(多数据集合并,已去重与泄漏核验) |
| **验收准确率** | **90.36%**(50939 / 56375) |
| 验证集 mAP@.50:.95 | 0.4048 |
| 验证集 mAP@.50 | 0.6693 |
| 单块时延(PyTorch @1024) | 14.7 ms |
| 单块时延(TensorRT @1024) | **5.57 ms** |
| 峰值显存 @1024 | 271 MB |

**验收口径**(项目统一):正确识别 = 类别一致 **且** IoU ≥ 0.40(贪心匹配);准确率 = 正确数 / GT 总数(**recall 式计数,不惩罚误检**)。

**分类别准确率**:

| 类别 | 正确 / GT | 准确率 |
|---|---|---|
| car | 42,407 / 47,453 | 89.37% |
| truck | 6,161 / 6,519 | 94.51% |
| bus | 1,072 / 1,080 | 99.26% |
| van | 775 / 793 | 97.73% |
| pickup | 484 / 490 | 98.78% |
| tank | 40 / 40 | 100.00% |

---

## 二、目录结构

```
.
├── README.md                      本文件
├── requirements.txt               依赖清单
├── .gitignore
├── docs/                          技术文档
│   └── DEIM_MODEL_TECHNICAL_DOCUMENT.md    架构 / 超参 / 实测 / 部署(完整)
├── engines/                       模型引擎
│   └── deim/                      DEIM 框架(完整可训练/可推理)
│       ├── engine/                模型定义(backbone / encoder / decoder / 损失 / 匹配)
│       ├── configs/               配置(含 include 链)
│       │   └── deim_rtdetrv2/deim_car_union_v4.yml    ← 本模型配置
│       ├── tools/                 工具脚本
│       ├── train.py               训练入口
│       └── patch_tw.py            torchvision 0.26 兼容补丁(必需)
├── evaluation/                    评估与导出
│   ├── eval_deploy.py             验收脚本(路径自适应)
│   └── export_trt.py              TensorRT 导出脚本
└── reports/                       实验报告
    └── EXPERIMENT_RESULTS.md      训练 / 验收 / 瓶颈分析 / 改进方向
```

> **注意**:训练与推理**不需要**改动任何绝对路径 —— `evaluation/*.py` 基于自身位置自动定位 `engines/deim/` 与数据目录。

---

## 三、数据与权重下载

> ⚠️ **权重与数据集体积过大,未纳入本仓库**(GitHub 单文件限制 100 MB、仓库建议 < 1 GB),请从下方地址获取。

| 资源 | 体积 | 内容 | 下载地址 |
|---|---|---|---|
| **模型权重** | 307 MB | `best_stg1.pth`(EMA 权重,epoch 31) | `TODO: 待补充网盘链接` |
| **数据集 CAR-UNION v4** | **7.35 GB** | train 50,024 块 / val 8,727 块 / test 3,976 块 + COCO 标注 | `TODO: 待补充网盘链接` |
| **测试集切片(快速验证用)** | 615 MB | test 3,976 块 + `annotations.json` | `TODO: 待补充网盘链接` |

**说明**:
- **只做推理 / 验收** → 下载「模型权重 + 测试集切片」(共 **922 MB**)即可;
- **要做训练** → 下载完整「数据集 CAR-UNION v4」;
- 测试集切片是数据集中 test 部分的子集,单独提供是为了免去下载 7.35 GB。

**数据格式**:
- 图片:切片后的 1280×1280 块(原图 1920×1080,x 方向 stride 640,每图 2 块)
- 标注:训练用 COCO JSON,评估用 `annotations.json`

**放置位置**(下载后按此结构摆放,脚本即可直接运行):

```
<仓库根目录>/
├── engines/deim/          ← 仓库自带
├── evaluation/            ← 仓库自带
├── weights/
│   └── best_stg1.pth      ← 从网盘下载
└── test/
    ├── images/            ← 从网盘下载(3976 张)
    └── annotations.json   ← 从网盘下载
```

---

## 四、环境依赖

```bash
# 实测可用组合
torch==2.11.0+cu128
torchvision==0.26.0+cu128
numpy / Pillow / PyYAML / tqdm
faster-coco-eval
# 可选(TensorRT 加速)
tensorrt==11.2.1.2
```

安装(按本机 CUDA 版本调整):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> **重要**:`engines/deim/patch_tw.py` 是**必需的** —— 它修补 torchvision 0.26 的 `Transform.forward` 接口变更(DEIM 使用旧式 `_transform` 接口)。若报 `NotImplementedError: _transform`,就是这个补丁没生效。`eval_deploy.py` 会自动加载它。

---

## 五、快速开始

### 5.1 验收测试

```bash
# 在仓库根目录执行
python evaluation/eval_deploy.py                 # 默认 conf 0.005(验收口径)
python evaluation/eval_deploy.py --scan          # 扫描多个置信度阈值
python evaluation/eval_deploy.py --imgsz 1280    # 改推理分辨率

# 手动指定路径
python evaluation/eval_deploy.py \
    --ckpt weights/best_stg1.pth \
    --json test/annotations.json \
    --imgroot test/images
```

**预期输出**:

```
test 块数=3976 GT目标=56375
conf=0.005: acc=0.9036 (50939/56375)  avg_ms=15.2  mean_score=0.080
  car       : 42407/47453 = 89.37%
  truck     : 6161/6519 = 94.51%
  ...
```

### 5.2 TensorRT 加速

> ⚠️ **engine 与 GPU 架构绑定,不能跨机器拷贝** —— 必须在目标机器上重新导出。

```bash
python evaluation/export_trt.py --imgsz 1024
```

导出时**必须完全定形**(固定 batch 与 H/W、无 `dynamic_axes`),否则会报
`Could not find any implementation for node ForeignNode[.../decoder/Gather_N]`。

### 5.3 训练

```bash
cd engines/deim
python train.py -c configs/deim_rtdetrv2/deim_car_union_v4.yml \
    -d cuda:0 --use-amp -u epoches=45
```

**关键超参**:

| 参数 | 值 |
|---|---|
| lr | 2e-4(AdamW) |
| weight_decay | 1e-4(norm/bn 层为 0) |
| epoches | 45 |
| LR 调度 | flatcosine(warmup 2000 步,flat_epoch 24,no_aug 2) |
| batch | 8 |
| 输入尺寸 | 960 × 960 |
| 增强策略 | `policy: epoch [4, 24, 43]` |

---

## 六、已知限制

| 限制 | 说明 | 缓解方式 |
|---|---|---|
| 小目标能力受骨干 stride 限制 | 目标 < 32px 时召回下降(car 有 50.6% < 32px) | **滑窗切片推理**,实测在 DOTA 目标域 +16pp |
| 300 queries 上限 | 单块最多 963 个目标,>300 的 9 张块占 9.4% GT,物理上检不全 | 滑窗切片(每窗目标数下降)或提高 `num_queries` |
| 误检量大 | 本口径不惩罚误检,test 上 Precision 仅 4.68% | 提高 conf 到 0.20~0.30(Precision 53%~68%) |
| Mosaic output_size 偏小 | `320` 会压缩小目标细节 | 训练时改 640 并降低概率 |

**详细分析与改进路线见**:
- `reports/EXPERIMENT_RESULTS.md`(实验数据、瓶颈量化、改进方案)
- `docs/DEIM_MODEL_TECHNICAL_DOCUMENT.md`(架构、输入输出、后处理、部署)

---

## 七、版本与来源

| 项 | 值 |
|---|---|
| 框架 | DEIM(DETR with Improved Matching) |
| 基座 | RT-DETRv2 |
| 骨干 | PResNet-18-d(COCO 120ep 预训练) |
| 训练日期 | 2026-09-10 ~ 09-12 |
| 权重时间 | 2026-09-11 19:12(epoch 31) |
| 训练平台 | RTX 5080 16GB |
